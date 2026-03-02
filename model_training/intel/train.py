"""
Intel Training Script
======================

Frozen OpenVINO encoder (CPU) → trainable LSTM decoder (Intel GPU).

Feature caching (default): Encodes all clips ONCE with OpenVINO, saves
to disk, then every epoch loads cached tensors at GPU speed.
  - First run:   ~12 min (one-time encoding)
  - Every epoch:  ~2-3 min (pure GPU)

Without caching (--no-feature-cache): Encodes on the fly every epoch.
  - Every epoch:  ~12 min (CPU-bottlenecked)

Usage (run from project root, e.g. D:\\movie_highlighter):

    python -m model_training.intel.train
    python -m model_training.intel.train --data-path /path/to/dataset
    python -m model_training.intel.train --resume checkpoints_intel/checkpoint_latest.pth
    python -m model_training.intel.train --epochs 30 --batch-size 4
    python -m model_training.intel.train --no-feature-cache
    python -m model_training.intel.train --rebuild-feature-cache
    python -m model_training.intel.train --force-cpu

Options:
    --data-path              Path to dataset folder (default: dataset/)
    --epochs                 Number of training epochs (default: 25)
    --batch-size             Batch size (default: 2)
    --lr                     Learning rate (default: 1e-4)
    --resume                 Resume from checkpoint path
    --no-viz                 Skip sample visualizations before training
    --viz                    Create sample visualizations before training
    --no-cache               Disable ROI cache (slow — runs YOLO every epoch)
    --rebuild-cache          Force rebuild ROI cache even if one exists
    --no-feature-cache       Disable feature caching (encode every epoch)
    --rebuild-feature-cache  Force rebuild feature cache
    --num-workers            DataLoader workers (default: 4, 0 = single-process)
    --force-cpu              Force CPU training (skip Intel GPU)
"""

import os
import sys
import argparse

# ---- Make imports work regardless of how the script is launched ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

# Shared
from model_training.shared.dataset import (
    VideoDataset,
    validate_and_split_dataset,
    apply_dataset_split,
    precompute_roi_cache,
)
from model_training.shared.training_utils import (
    set_seed,
    compute_class_weights,
    save_checkpoint,
    load_checkpoint,
    create_production_model,
    ActionRecognitionModel,
)
from model_training.shared.detection import PoseExtractor
from model_training.shared.visualization import create_sample_visualizations
from model_training.shared.feature_cache import (
    precompute_feature_cache,
    CachedFeatureDataset,
)

# Intel-specific
from model_training.intel.config import CONFIG
from model_training.intel.model import IntelFeatureExtractor, EncoderLSTM

# =============================
# Intel GPU Detection
# =============================
HAS_IPEX = False
HAS_INTEL_GPU = False

try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        HAS_INTEL_GPU = True
        _gpu_name = torch.xpu.get_device_name(0) if hasattr(torch.xpu, 'get_device_name') else 'Intel GPU'
        print(f"✅ Intel GPU detected: {_gpu_name}")
except (ImportError, OSError):
    pass


# =============================
# Validation (cached mode)
# =============================
def validate_cached(model, val_loader, device, criterion, use_amp=False):
    """Validate with pre-encoded features (no encoder needed)."""
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    class_correct, class_total, class_preds = {}, {}, {}

    with torch.no_grad():
        for feats, labels in val_loader:
            feats = feats.to(device)
            labels = labels.to(device)

            if use_amp:
                with torch.amp.autocast('xpu', dtype=torch.bfloat16):
                    outputs, _ = model(feats)
                loss = criterion(outputs.float(), labels)
            else:
                outputs, _ = model(feats)
                loss = criterion(outputs, labels)

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item() * labels.size(0)

            for lbl, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                lbl = int(lbl)
                class_total[lbl] = class_total.get(lbl, 0) + 1
                class_preds.setdefault(lbl, []).append(int(pred))
                if lbl == pred:
                    class_correct[lbl] = class_correct.get(lbl, 0) + 1

    acc = correct / total if total else 0
    avg_loss = running_loss / total if total else float("inf")
    per_class = {
        lbl: class_correct.get(lbl, 0) / class_total[lbl]
        for lbl in class_total
    }
    return avg_loss, acc, per_class, class_preds


# =============================
# Validation (live encoder mode)
# =============================
def validate_live(encoder, model, val_loader, device, criterion, use_amp=False):
    """Validate with live OpenVINO encoding (fallback when no cache)."""
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    class_correct, class_total, class_preds = {}, {}, {}

    with torch.no_grad():
        for frames, labels in val_loader:
            labels = labels.to(device)
            feats = encoder.encode(frames.cpu()).to(device)

            if use_amp:
                with torch.amp.autocast('xpu', dtype=torch.bfloat16):
                    outputs, _ = model(feats)
                loss = criterion(outputs.float(), labels)
            else:
                outputs, _ = model(feats)
                loss = criterion(outputs, labels)

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item() * labels.size(0)

            for lbl, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                lbl = int(lbl)
                class_total[lbl] = class_total.get(lbl, 0) + 1
                class_preds.setdefault(lbl, []).append(int(pred))
                if lbl == pred:
                    class_correct[lbl] = class_correct.get(lbl, 0) + 1

    acc = correct / total if total else 0
    avg_loss = running_loss / total if total else float("inf")
    per_class = {
        lbl: class_correct.get(lbl, 0) / class_total[lbl]
        for lbl in class_total
    }
    return avg_loss, acc, per_class, class_preds


# =============================
# Training Loop (cached)
# =============================
def train_classifier_cached(train_loader, val_loader, feature_dim, num_classes,
                            label_to_idx, idx_to_label):
    """Train LSTM from cached features — pure GPU, no encoder needed."""
    device = torch.device(CONFIG["device"])
    use_intel_gpu = device.type == 'xpu' and HAS_INTEL_GPU

    print(f"Feature dimension: {feature_dim}")

    model = EncoderLSTM(
        feature_dim=feature_dim,
        hidden_dim=CONFIG.get("hidden_dim", 256),
        num_classes=num_classes,
        num_layers=CONFIG.get("num_layers", 2),
        dropout=CONFIG.get("dropout", 0.3),
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"📊 EncoderLSTM: {total_p:,} parameters")

    # Loss
    if CONFIG.get("use_class_weights", True):
        weights = compute_class_weights(train_loader.dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # LR
    is_resuming = CONFIG.get("checkpoint_path") and os.path.exists(CONFIG["checkpoint_path"])
    lr = CONFIG["finetune_learning_rate"] if is_resuming else CONFIG["base_learning_rate"]
    print(f"{'🔄 Resume' if is_resuming else '🆕 Fresh'} LR: {lr}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ---- Intel GPU: IPEX optimize ----
    use_amp = False
    if use_intel_gpu and HAS_IPEX:
        try:
            model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
            use_amp = True
            print("✅ IPEX optimizations applied (bfloat16)")
        except Exception as e:
            print(f"⚠️  IPEX optimize failed: {e}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG.get("base_epochs", 25), eta_min=1e-6
    )

    start_epoch, best_acc, best_loss = 0, 0.0, float("inf")
    best_state = None

    if is_resuming:
        ckpt = load_checkpoint(CONFIG["checkpoint_path"], model, optimizer, device)
        if ckpt:
            if ckpt.get("transfer_learning"):
                start_epoch = 0
            else:
                start_epoch = ckpt.get("epoch", 0) + 1
                best_acc = ckpt.get("best_val_acc", 0.0)
                best_loss = ckpt.get("best_val_loss", float("inf"))

    max_epochs = (
        start_epoch + CONFIG.get("max_finetune_epochs", 15)
        if is_resuming else CONFIG.get("base_epochs", 25)
    )
    patience_ctr = 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}")
        for feats, labels in pbar:
            feats = feats.to(device)
            labels = labels.to(device)

            if use_amp:
                with torch.amp.autocast('xpu', dtype=torch.bfloat16):
                    outputs, _ = model(feats)
                loss = criterion(outputs.float(), labels)
            else:
                outputs, _ = model(feats)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            run_loss += loss.item() * feats.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{correct / total:.4f}")

        scheduler.step()
        t_loss = run_loss / total if total else float("inf")
        t_acc = correct / total if total else 0
        print(f"\n  Train Loss: {t_loss:.4f} | Acc: {t_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Validation
        if len(val_loader) > 0:
            v_loss, v_acc, pc_acc, _ = validate_cached(
                model, val_loader, device, criterion, use_amp=use_amp
            )
            print(f"  Val   Loss: {v_loss:.4f} | Acc: {v_acc:.4f}")
            for li in sorted(pc_acc):
                print(f"    {'✓' if pc_acc[li] > 0 else '⚠️'} {idx_to_label[li]}: {pc_acc[li]:.4f}")

            if v_loss < best_loss - CONFIG.get("min_delta", 0.001):
                best_loss, best_acc = v_loss, v_acc
                best_state = model.state_dict().copy()
                patience_ctr = 0
                print("   ⭐ Improved!")
            else:
                patience_ctr += 1
                print(f"   No improvement ({patience_ctr}/{CONFIG['early_stopping_patience']})")
                if patience_ctr >= CONFIG["early_stopping_patience"]:
                    print("\n🛑 Early stopping")
                    break

        # Checkpoint
        ckpt_every = CONFIG.get("save_checkpoint_every")
        if ckpt_every and (epoch + 1) % ckpt_every == 0:
            ckpt_dir = CONFIG.get("checkpoint_dir", "checkpoints_intel")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_checkpoint(
                model, optimizer, epoch, best_acc, label_to_idx, idx_to_label,
                feature_dim, os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
                best_val_loss=best_loss,
                extra={"sequence_length": CONFIG["sequence_length"]},
            )

    if best_state:
        model.load_state_dict(best_state)
        print(f"\n✅ Best model loaded (loss={best_loss:.4f}, acc={best_acc:.4f})")

    # Save full model
    wrapped = ActionRecognitionModel(
        model=model, label_to_idx=label_to_idx, idx_to_label=idx_to_label,
        feature_dim=feature_dim, sequence_length=CONFIG["sequence_length"],
        model_type="EncoderLSTM",
        extra_meta={"hidden_dim": CONFIG.get("hidden_dim", 256),
                    "num_layers": CONFIG.get("num_layers", 2)},
    )
    wrapped.save(CONFIG["model_save_path"])
    return wrapped, use_amp


# =============================
# Training Loop (live encoder)
# =============================
def train_classifier_live(encoder, train_loader, val_loader, num_classes,
                          label_to_idx, idx_to_label):
    """Train with live OpenVINO encoding each batch (fallback)."""
    device = torch.device(CONFIG["device"])
    use_intel_gpu = device.type == 'xpu' and HAS_INTEL_GPU

    # Discover feature dim
    with torch.no_grad():
        sample, _ = next(iter(train_loader))
        dummy = encoder.encode(sample[0:1].cpu())
        feature_dim = dummy.shape[-1]
    print(f"Feature dimension: {feature_dim}")

    model = EncoderLSTM(
        feature_dim=feature_dim,
        hidden_dim=CONFIG.get("hidden_dim", 256),
        num_classes=num_classes,
        num_layers=CONFIG.get("num_layers", 2),
        dropout=CONFIG.get("dropout", 0.3),
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"📊 EncoderLSTM: {total_p:,} parameters")

    # Loss
    if CONFIG.get("use_class_weights", True):
        weights = compute_class_weights(train_loader.dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # LR
    is_resuming = CONFIG.get("checkpoint_path") and os.path.exists(CONFIG["checkpoint_path"])
    lr = CONFIG["finetune_learning_rate"] if is_resuming else CONFIG["base_learning_rate"]
    print(f"{'🔄 Resume' if is_resuming else '🆕 Fresh'} LR: {lr}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ---- Intel GPU: IPEX optimize ----
    use_amp = False
    if use_intel_gpu and HAS_IPEX:
        try:
            model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
            use_amp = True
            print("✅ IPEX optimizations applied (bfloat16)")
        except Exception as e:
            print(f"⚠️  IPEX optimize failed: {e}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG.get("base_epochs", 25), eta_min=1e-6
    )

    start_epoch, best_acc, best_loss = 0, 0.0, float("inf")
    best_state = None

    if is_resuming:
        ckpt = load_checkpoint(CONFIG["checkpoint_path"], model, optimizer, device)
        if ckpt:
            if ckpt.get("transfer_learning"):
                start_epoch = 0
            else:
                start_epoch = ckpt.get("epoch", 0) + 1
                best_acc = ckpt.get("best_val_acc", 0.0)
                best_loss = ckpt.get("best_val_loss", float("inf"))

    max_epochs = (
        start_epoch + CONFIG.get("max_finetune_epochs", 15)
        if is_resuming else CONFIG.get("base_epochs", 25)
    )
    patience_ctr = 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}")
        for frames, labels in pbar:
            labels = labels.to(device)

            with torch.no_grad():
                feats = encoder.encode(frames.cpu())
            feats = feats.to(device)

            if use_amp:
                with torch.amp.autocast('xpu', dtype=torch.bfloat16):
                    outputs, _ = model(feats)
                loss = criterion(outputs.float(), labels)
            else:
                outputs, _ = model(feats)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            run_loss += loss.item() * frames.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{correct / total:.4f}")

        scheduler.step()
        t_loss = run_loss / total if total else float("inf")
        t_acc = correct / total if total else 0
        print(f"\n  Train Loss: {t_loss:.4f} | Acc: {t_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Validation
        if len(val_loader) > 0:
            v_loss, v_acc, pc_acc, _ = validate_live(
                encoder, model, val_loader, device, criterion, use_amp=use_amp
            )
            print(f"  Val   Loss: {v_loss:.4f} | Acc: {v_acc:.4f}")
            for li in sorted(pc_acc):
                print(f"    {'✓' if pc_acc[li] > 0 else '⚠️'} {idx_to_label[li]}: {pc_acc[li]:.4f}")

            if v_loss < best_loss - CONFIG.get("min_delta", 0.001):
                best_loss, best_acc = v_loss, v_acc
                best_state = model.state_dict().copy()
                patience_ctr = 0
                print("   ⭐ Improved!")
            else:
                patience_ctr += 1
                print(f"   No improvement ({patience_ctr}/{CONFIG['early_stopping_patience']})")
                if patience_ctr >= CONFIG["early_stopping_patience"]:
                    print("\n🛑 Early stopping")
                    break

        # Checkpoint
        ckpt_every = CONFIG.get("save_checkpoint_every")
        if ckpt_every and (epoch + 1) % ckpt_every == 0:
            ckpt_dir = CONFIG.get("checkpoint_dir", "checkpoints_intel")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_checkpoint(
                model, optimizer, epoch, best_acc, label_to_idx, idx_to_label,
                feature_dim, os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
                best_val_loss=best_loss,
                extra={"sequence_length": CONFIG["sequence_length"]},
            )

    if best_state:
        model.load_state_dict(best_state)
        print(f"\n✅ Best model loaded (loss={best_loss:.4f}, acc={best_acc:.4f})")

    wrapped = ActionRecognitionModel(
        model=model, label_to_idx=label_to_idx, idx_to_label=idx_to_label,
        feature_dim=feature_dim, sequence_length=CONFIG["sequence_length"],
        model_type="EncoderLSTM",
        extra_meta={"hidden_dim": CONFIG.get("hidden_dim", 256),
                    "num_layers": CONFIG.get("num_layers", 2)},
    )
    wrapped.save(CONFIG["model_save_path"])
    return wrapped, use_amp


# =============================
# Main
# =============================
def main():
    parser = argparse.ArgumentParser(description="Intel OpenVINO encoder → LSTM training")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip sample visualizations before training")
    parser.add_argument("--viz", action="store_true",
                        help="Create sample visualizations before training")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable ROI cache (slow — runs YOLO every epoch)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rebuild ROI cache even if one exists")
    parser.add_argument("--no-feature-cache", action="store_true",
                        help="Disable feature caching (encode every epoch)")
    parser.add_argument("--rebuild-feature-cache", action="store_true",
                        help="Force rebuild feature cache")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (default: 4, 0 = single-process)")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU training (skip Intel GPU)")
    args = parser.parse_args()

    # Override config from CLI
    if args.data_path:
        CONFIG["data_path"] = args.data_path
    if args.resume:
        CONFIG["checkpoint_path"] = args.resume
    if args.epochs:
        CONFIG["base_epochs"] = args.epochs
    if args.batch_size:
        CONFIG["batch_size"] = args.batch_size
    if args.lr:
        CONFIG["base_learning_rate"] = args.lr
    if args.no_viz:
        CONFIG["create_visualizations"] = False
    if args.viz:
        CONFIG["create_visualizations"] = True
    if args.num_workers is not None:
        CONFIG["num_workers"] = args.num_workers

    # ---- Device selection ----
    if args.force_cpu:
        CONFIG["device"] = "cpu"
        print("🎯 Forced CPU training")
    elif HAS_INTEL_GPU:
        CONFIG["device"] = "xpu"
        print("✅ LSTM will train on Intel GPU")
    elif torch.cuda.is_available():
        CONFIG["device"] = "cuda"
        print(f"✅ Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        CONFIG["device"] = "cpu"
        print("🎯 Training on CPU")

    set_seed(42)
    print("=" * 60)
    print("🧠 INTEL ENCODER (CPU) → LSTM TRAINING (" + CONFIG["device"].upper() + ")")
    print("=" * 60)

    # Check encoder
    if not os.path.exists(CONFIG["encoder_xml"]):
        print(f"❌ Encoder not found: {CONFIG['encoder_xml']}")
        sys.exit(1)

    # Datasets
    data_path = CONFIG["data_path"]
    train_ds = VideoDataset(os.path.join(data_path, "train"), CONFIG)
    val_ds = VideoDataset(os.path.join(data_path, "val"), CONFIG)

    if len(train_ds) == 0:
        print("❌ No training samples found")
        sys.exit(1)

    print(f"\n📁 Train: {len(train_ds)} | Val: {len(val_ds)}")

    ok, valid_actions, new_train, new_val = validate_and_split_dataset(train_ds, val_ds, CONFIG)
    if not ok:
        sys.exit(1)
    apply_dataset_split(train_ds, val_ds, valid_actions, new_train, new_val)

    label_to_idx, idx_to_label = train_ds.get_label_mapping()
    print(f"\n📝 Labels: {label_to_idx}")

    # ==============================
    # PRE-COMPUTE ROI CACHE
    # ==============================
    roi_cache = None
    pose_ext = None

    if CONFIG.get("use_adaptive_cropping") and CONFIG.get("use_pose_guided_crop"):
        pose_ext = PoseExtractor(CONFIG.get("pose_model", "yolo11n-pose.pt"),
                                  CONFIG.get("pose_conf_threshold", 0.3))

    if not args.no_cache:
        all_ds = VideoDataset.__new__(VideoDataset)
        all_ds.samples = train_ds.samples + val_ds.samples
        all_ds.config = CONFIG

        if args.rebuild_cache:
            import hashlib as _h
            h = _h.md5()
            h.update(str(CONFIG.get("sequence_length", 16)).encode())
            h.update(str(CONFIG.get("default_stride", 4)).encode())
            h.update(str(CONFIG.get("crop_size", (224, 224))).encode())
            h.update(str(len(all_ds.samples)).encode())
            cache_file = os.path.join(
                CONFIG.get("checkpoint_dir", "."),
                f"roi_cache_{h.hexdigest()[:8]}.pkl",
            )
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"🗑️  Deleted old cache: {cache_file}")

        roi_cache = precompute_roi_cache(all_ds, CONFIG, pose_extractor=pose_ext)
        train_ds.roi_cache = roi_cache
        val_ds.roi_cache = roi_cache
    else:
        print("\n⚠️  ROI cache DISABLED — training will be slow (YOLO runs every epoch)")

    # ==============================
    # Sample Visualizations
    # ==============================
    if CONFIG.get("create_visualizations") and pose_ext:
        create_sample_visualizations(
            train_ds, pose_ext,
            num_samples=CONFIG.get("num_visualization_samples", 2),
            sample_rate=CONFIG.get("visualization_sample_rate", 5),
        )

    # ==============================
    # Encoder (always CPU)
    # ==============================
    encoder = IntelFeatureExtractor(CONFIG["encoder_xml"], CONFIG["encoder_bin"])

    # ==============================
    # FEATURE CACHING
    # ==============================
    use_feature_cache = not args.no_feature_cache

    if use_feature_cache:
        print("\n" + "=" * 60)
        print("📦 FEATURE CACHING")
        print("=" * 60)

        cache_dir = CONFIG.get("checkpoint_dir", "checkpoints_intel")
        force_rebuild = args.rebuild_feature_cache

        # Cache train and val separately
        print("\n--- Training set ---")
        train_cache_path = precompute_feature_cache(
            train_ds, encoder, CONFIG,
            cache_dir=cache_dir, force_rebuild=force_rebuild,
        )

        print("\n--- Validation set ---")
        val_cache_path = precompute_feature_cache(
            val_ds, encoder, CONFIG,
            cache_dir=cache_dir, force_rebuild=force_rebuild,
        )

        # Create cached datasets
        train_cached = CachedFeatureDataset(
            train_cache_path,
            label_to_idx=label_to_idx, idx_to_label=idx_to_label,
        )
        val_cached = CachedFeatureDataset(
            val_cache_path,
            label_to_idx=label_to_idx, idx_to_label=idx_to_label,
        )

        feature_dim = train_cached.feature_dim

        # DataLoaders — pure tensor loading, can use many workers
        nw = CONFIG.get("num_workers", 4)
        pin = CONFIG["device"] in ("cuda", "xpu")
        pf = CONFIG.get("prefetch_factor", 2) if nw > 0 else None
        pw = CONFIG.get("persistent_workers", True) and nw > 0

        print(f"\n📊 Cached DataLoader: {nw} workers, pin_memory={pin}")

        train_loader = DataLoader(train_cached, batch_size=CONFIG["batch_size"],
                                  shuffle=True, num_workers=nw, pin_memory=pin,
                                  prefetch_factor=pf, persistent_workers=pw)
        val_loader = DataLoader(val_cached, batch_size=CONFIG["batch_size"],
                                shuffle=False, num_workers=nw, pin_memory=pin,
                                prefetch_factor=pf, persistent_workers=pw)

        # Train (cached — no encoder in the loop)
        print(f"\n🚀 Training from cached features...\n")
        wrapped, use_amp = train_classifier_cached(
            train_loader, val_loader,
            feature_dim=feature_dim,
            num_classes=len(valid_actions),
            label_to_idx=label_to_idx,
            idx_to_label=idx_to_label,
        )

        # For post-training validation, use cached validate
        def _val_fn():
            prod_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            device = torch.device(CONFIG["device"])
            _, _, pc, preds = validate_cached(
                wrapped.model, val_loader, device, prod_criterion, use_amp=use_amp
            )
            return pc, preds

    else:
        # ==============================
        # LIVE ENCODING (no cache)
        # ==============================
        nw = CONFIG.get("num_workers", 4) if roi_cache is not None else 0
        pin = CONFIG["device"] in ("cuda", "xpu")
        pf = CONFIG.get("prefetch_factor", 2) if nw > 0 else None
        pw = CONFIG.get("persistent_workers", True) and nw > 0

        print(f"\n📊 DataLoader: {nw} workers, pin_memory={pin}")

        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                                  shuffle=True, num_workers=nw, pin_memory=pin,
                                  prefetch_factor=pf, persistent_workers=pw)
        val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"],
                                shuffle=False, num_workers=nw, pin_memory=pin,
                                prefetch_factor=pf, persistent_workers=pw)

        print(f"\n🚀 Training (live encoding)...\n")
        wrapped, use_amp = train_classifier_live(
            encoder, train_loader, val_loader,
            num_classes=len(valid_actions),
            label_to_idx=label_to_idx,
            idx_to_label=idx_to_label,
        )

        # For post-training validation, use live validate
        def _val_fn():
            prod_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            device = torch.device(CONFIG["device"])
            _, _, pc, preds = validate_live(
                encoder, wrapped.model, val_loader, device, prod_criterion, use_amp=use_amp
            )
            return pc, preds

    # ==============================================================
    # POST-TRAINING: Create filtered mappings
    # ==============================================================
    print("\n" + "=" * 70)
    print("📋 BASE MODEL FILTERING (remove 0% accuracy)")
    print("=" * 70)
    base_keep, base_remove, per_class_acc = create_production_model(
        wrapped, _val_fn, min_val_accuracy=0.001,
    )

    if base_remove:
        wrapped.save_filtered_mapping(
            CONFIG["model_save_path"], base_keep,
            per_class_acc=per_class_acc, suffix="",
        )

    # --- Production filtering: remove <30% accuracy classes ---
    min_prod_acc = CONFIG.get("min_production_accuracy", 0.3)
    prod_keep = [idx for idx in sorted(wrapped.idx_to_label.keys())
                 if per_class_acc.get(idx, 0.0) >= min_prod_acc]
    prod_remove = [idx for idx in sorted(wrapped.idx_to_label.keys())
                   if per_class_acc.get(idx, 0.0) < min_prod_acc]

    print(f"\n📋 PRODUCTION FILTERING (remove <{min_prod_acc:.0%} accuracy)")
    print(f"   Keep: {len(prod_keep)} | Remove: {len(prod_remove)}")

    wrapped.save_filtered_mapping(
        CONFIG["model_save_path"], prod_keep,
        per_class_acc=per_class_acc, suffix="_production",
    )

    # --- Summary ---
    base_mapping = CONFIG["model_save_path"].replace(".pth", "_mapping.json")
    prod_mapping = CONFIG["model_save_path"].replace(".pth", "_production_mapping.json")

    print(f"\n✅ Done!")
    print(f"  Weights:             {CONFIG['model_save_path']}")
    print(f"  Base mapping:        {base_mapping} ({len(base_keep)} classes)")
    print(f"  Production mapping:  {prod_mapping} ({len(prod_keep)} classes)")
    if base_remove:
        print(f"\n  Removed from base (0% accuracy):")
        for ri in base_remove:
            print(f"    ❌ {wrapped.idx_to_label.get(ri, f'class_{ri}')}")
    if len(prod_remove) > len(base_remove):
        extra_removed = [ri for ri in prod_remove if ri not in base_remove]
        if extra_removed:
            print(f"\n  Additionally removed for production (<{min_prod_acc:.0%}):")
            for ri in extra_removed:
                acc = per_class_acc.get(ri, 0.0)
                print(f"    ❌ {wrapped.idx_to_label.get(ri, f'class_{ri}')} ({acc:.1%})")


if __name__ == "__main__":
    main()