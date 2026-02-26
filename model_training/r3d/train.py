"""
R3D Training Script
====================

End-to-end 3D CNN fine-tuning (r3d_18 / mc3_18 / r2plus1d_18).

Usage (run from project root, e.g. D:\\movie_highlighter):

    python -m model_training.r3d.train
    python -m model_training.r3d.train --model r2plus1d_18 --epochs 40
    python -m model_training.r3d.train --freeze-backbone --lr 3e-4
    python -m model_training.r3d.train --rebuild-cache
    python -m model_training.r3d.train --no-cache --num-workers 0

Options:
    --model             r3d_18 | mc3_18 | r2plus1d_18 (default: r3d_18)
    --data-path         Path to dataset folder (default: dataset/)
    --epochs            Number of training epochs (default: 30)
    --batch-size        Batch size (default: 4)
    --lr                Learning rate (default: 1e-4)
    --resume            Resume from checkpoint path
    --freeze-backbone   Only train the FC head, freeze 3D-CNN layers
    --no-amp            Disable mixed precision (use FP32)
    --no-onnx           Skip ONNX export after training
    --no-cache          Disable ROI cache (slow — runs YOLO every epoch)
    --rebuild-cache     Force rebuild ROI cache even if one exists
    --num-workers       DataLoader workers (default: 4, 0 = single-process)
    --no-viz            Skip sample visualizations before training
    --viz               Create sample visualizations before training
"""

import os
import sys
import gc
import argparse
import numpy as np

# ---- Make imports work regardless of how the script is launched ----
# Walks up from this file to find the project root (parent of model_training/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
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
    build_filtered_label_maps,
    ActionRecognitionModel,
)
from model_training.shared.detection import PoseExtractor
from model_training.shared.visualization import create_sample_visualizations

# R3D-specific
from model_training.r3d.config import CONFIG
from model_training.r3d.model import (
    build_r3d_model,
    freeze_batchnorm,
    get_parameter_groups,
    export_onnx,
)


# =============================
# R3D-specific dataset wrapper
# =============================
class R3DVideoDataset(VideoDataset):
    """
    Extends VideoDataset to return clips in R3D format: (B, C, T, H, W).
    The base class returns (T, C, H, W); we permute here.
    """

    def __getitem__(self, idx):
        frames_tensor, label = super().__getitem__(idx)
        # (T, C, H, W) → (C, T, H, W)
        frames_tensor = frames_tensor.permute(1, 0, 2, 3)
        return frames_tensor, label


# =============================
# Validation
# =============================
def validate(model, val_loader, device, criterion, use_amp=False):
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    class_correct, class_total, class_preds = {}, {}, {}

    with torch.no_grad():
        for clips, labels in val_loader:
            clips, labels = clips.to(device), labels.to(device)

            if use_amp and device.type == "cuda":
                with autocast():
                    outputs = model(clips)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(clips)
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
# Training Loop
# =============================
def train_r3d(train_loader, val_loader, num_classes, label_to_idx, idx_to_label):
    device = torch.device(CONFIG["device"])
    use_amp = CONFIG.get("use_amp", True) and device.type == "cuda"

    # Build model
    model = build_r3d_model(num_classes, CONFIG).to(device)

    # Loss
    if CONFIG.get("use_class_weights", True):
        weights = compute_class_weights(train_loader.dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights,
                                         label_smoothing=CONFIG.get("label_smoothing", 0.1))
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG.get("label_smoothing", 0.1))

    # Optimizer
    is_resuming = CONFIG.get("checkpoint_path") and os.path.exists(CONFIG["checkpoint_path"])
    lr = CONFIG["finetune_learning_rate"] if is_resuming else CONFIG["base_learning_rate"]

    if CONFIG.get("differential_lr", True) and not CONFIG.get("freeze_backbone", False):
        param_groups = get_parameter_groups(model, CONFIG)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=CONFIG.get("weight_decay", 1e-4))
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=CONFIG.get("weight_decay", 1e-4),
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG.get("base_epochs", 30), eta_min=1e-6
    )

    scaler = GradScaler(enabled=use_amp)

    start_epoch, best_acc, best_loss = 0, 0.0, float("inf")
    best_state = None

    # Resume
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
        if is_resuming else CONFIG.get("base_epochs", 30)
    )
    patience_ctr = 0
    clip_norm = CONFIG.get("gradient_clip_norm", 1.0)

    print(f"\n🏋️ Training R3D ({CONFIG['model_variant']}) for {max_epochs - start_epoch} epochs")
    print(f"   AMP: {use_amp} | Device: {device}")

    for epoch in range(start_epoch, max_epochs):
        model.train()
        if CONFIG.get("freeze_bn", True):
            freeze_batchnorm(model)

        run_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}")
        for clips, labels in pbar:
            clips, labels = clips.to(device), labels.to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    outputs = model(clips)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(clips)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            run_loss += loss.item() * clips.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

        scheduler.step()
        t_loss = run_loss / total if total else float("inf")
        t_acc = correct / total if total else 0.0
        lrs = [f"{g['lr']:.6f}" for g in optimizer.param_groups]
        print(f"\n  Train Loss: {t_loss:.4f} | Acc: {t_acc:.4f} | LR: {', '.join(lrs)}")

        # Validation
        if len(val_loader) > 0:
            v_loss, v_acc, pc_acc, _ = validate(model, val_loader, device, criterion, use_amp)
            print(f"  Val   Loss: {v_loss:.4f} | Acc: {v_acc:.4f}")
            for li in sorted(pc_acc):
                s = "✓" if pc_acc[li] > 0 else "⚠️"
                print(f"    {s} {idx_to_label[li]}: {pc_acc[li]:.4f}")

            if v_loss < best_loss - CONFIG.get("min_delta", 0.001):
                best_loss, best_acc = v_loss, v_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
            ckpt_dir = CONFIG.get("checkpoint_dir", "checkpoints_r3d")
            save_checkpoint(
                model, optimizer, epoch, best_acc, label_to_idx, idx_to_label,
                feature_dim=0,  # not applicable for end-to-end
                checkpoint_path=os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
                best_val_loss=best_loss,
                extra={
                    "model_variant": CONFIG["model_variant"],
                    "sequence_length": CONFIG["sequence_length"],
                    "crop_size": list(CONFIG["crop_size"]),
                },
            )

        # Periodic GC
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Load best
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"\n✅ Best model loaded (loss={best_loss:.4f}, acc={best_acc:.4f})")

    # Save
    wrapped = ActionRecognitionModel(
        model=model, label_to_idx=label_to_idx, idx_to_label=idx_to_label,
        feature_dim=0, sequence_length=CONFIG["sequence_length"],
        model_type=f"R3D_{CONFIG['model_variant']}",
        extra_meta={
            "model_variant": CONFIG["model_variant"],
            "crop_size": list(CONFIG["crop_size"]),
            "mean": CONFIG["mean"],
            "std": CONFIG["std"],
        },
    )
    wrapped.save(CONFIG["model_save_path"])

    # ONNX export
    if CONFIG.get("export_onnx", True):
        onnx_path = CONFIG["model_save_path"].replace(".pth", ".onnx")
        export_onnx(model, num_classes, CONFIG, onnx_path)

    return wrapped, model


# =============================
# Main
# =============================
def main():
    parser = argparse.ArgumentParser(description="R3D 3D-CNN fine-tuning")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--model", type=str, default=None,
                        choices=["r3d_18", "mc3_18", "r2plus1d_18"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-onnx", action="store_true")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable ROI cache (slow — runs YOLO every epoch)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rebuild ROI cache even if one exists")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip sample visualizations before training")
    parser.add_argument("--viz", action="store_true",
                        help="Create sample visualizations before training")
    args = parser.parse_args()

    # Override config
    if args.data_path:
        CONFIG["data_path"] = args.data_path
    if args.model:
        CONFIG["model_variant"] = args.model
    if args.resume:
        CONFIG["checkpoint_path"] = args.resume
    if args.epochs:
        CONFIG["base_epochs"] = args.epochs
    if args.batch_size:
        CONFIG["batch_size"] = args.batch_size
    if args.lr:
        CONFIG["base_learning_rate"] = args.lr
    if args.freeze_backbone:
        CONFIG["freeze_backbone"] = True
    if args.no_amp:
        CONFIG["use_amp"] = False
    if args.no_onnx:
        CONFIG["export_onnx"] = False
    if args.num_workers is not None:
        CONFIG["num_workers"] = args.num_workers
    if args.no_viz:
        CONFIG["create_visualizations"] = False
    if args.viz:
        CONFIG["create_visualizations"] = True

    set_seed(42)
    print("=" * 60)
    print(f"🎮 R3D TRAINING — {CONFIG['model_variant'].upper()}")
    print("=" * 60)

    # Datasets (R3D-specific: returns (C, T, H, W))
    data_path = CONFIG["data_path"]
    train_ds = R3DVideoDataset(os.path.join(data_path, "train"), CONFIG)
    val_ds = R3DVideoDataset(os.path.join(data_path, "val"), CONFIG)

    if len(train_ds) == 0:
        print("❌ No training samples")
        sys.exit(1)

    print(f"\n📁 Train: {len(train_ds)} | Val: {len(val_ds)}")

    ok, valid_actions, new_train, new_val = validate_and_split_dataset(train_ds, val_ds, CONFIG)
    if not ok:
        sys.exit(1)
    apply_dataset_split(train_ds, val_ds, valid_actions, new_train, new_val)

    label_to_idx, idx_to_label = train_ds.get_label_mapping()
    print(f"\n📝 Labels: {label_to_idx}")

    # ==============================
    # PRE-COMPUTE ROI CACHE (one-time cost, massive speedup per epoch)
    # ==============================
    roi_cache = None
    pose_ext = None

    if CONFIG.get("use_adaptive_cropping") and CONFIG.get("use_pose_guided_crop"):
        pose_ext = PoseExtractor(
            CONFIG.get("pose_model", "yolo11n-pose.pt"),
            CONFIG.get("pose_conf_threshold", 0.3),
        )

    if not args.no_cache:        # Build a combined sample list for caching
        all_samples_ds = R3DVideoDataset.__new__(R3DVideoDataset)
        all_samples_ds.samples = train_ds.samples + val_ds.samples
        all_samples_ds.config = CONFIG

        # Force rebuild if requested
        if args.rebuild_cache:
            import hashlib as _h
            h = _h.md5()
            h.update(str(CONFIG.get("sequence_length", 16)).encode())
            h.update(str(CONFIG.get("default_stride", 4)).encode())
            h.update(str(CONFIG.get("crop_size", (112, 112))).encode())
            h.update(str(len(all_samples_ds.samples)).encode())
            cache_file = os.path.join(
                CONFIG.get("checkpoint_dir", "."),
                f"roi_cache_{h.hexdigest()[:8]}.pkl",
            )
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"🗑️  Deleted old cache: {cache_file}")

        roi_cache = precompute_roi_cache(all_samples_ds, CONFIG, pose_extractor=pose_ext)

        # Assign cache — __getitem__ now uses fast path
        train_ds.roi_cache = roi_cache
        val_ds.roi_cache = roi_cache
    else:
        print("\n⚠️  ROI cache DISABLED — training will be slow (YOLO runs every epoch)")

    # ==============================
    # DataLoaders
    # ==============================
    # With cache: safe to use num_workers > 0 (no YOLO model in workers)
    # Without cache: must use num_workers=0 (YOLO can't be pickled)
    nw = CONFIG.get("num_workers", 4) if roi_cache is not None else 0
    pin = CONFIG["device"] == "cuda"
    pf = CONFIG.get("prefetch_factor", 2) if nw > 0 else None
    pw = CONFIG.get("persistent_workers", True) and nw > 0

    print(f"\n📊 DataLoader: {nw} workers, pin_memory={pin}, "
          f"prefetch={pf}, persistent={pw}")

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=nw, pin_memory=pin,
        prefetch_factor=pf, persistent_workers=pw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=nw, pin_memory=pin,
        prefetch_factor=pf, persistent_workers=pw,
    )

    # ==============================
    # Sample Visualizations
    # ==============================
    if CONFIG.get("create_visualizations") and pose_ext:
        create_sample_visualizations(
            train_ds, pose_ext,
            num_samples=CONFIG.get("num_visualization_samples", 2),
            sample_rate=CONFIG.get("visualization_sample_rate", 5),
            visualize_skeletons=CONFIG.get("visualize_skeletons", False),
        )

    # Train
    print(f"\n🚀 Training...\n")
    wrapped, raw_model = train_r3d(
        train_loader, val_loader,
        num_classes=len(valid_actions),
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
    )

    # Production model
    device = torch.device(CONFIG["device"])
    if CONFIG.get("use_class_weights"):
        w = compute_class_weights(train_ds).to(device)
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=CONFIG.get("label_smoothing", 0.1))
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG.get("label_smoothing", 0.1))

    use_amp = CONFIG.get("use_amp", True) and device.type == "cuda"

    def _val_fn():
        raw_model.to(device).float()  # ensure FP32 on correct device
        _, _, pc, preds = validate(raw_model, val_loader, device, criterion, use_amp=False)
        return pc, preds

    keep, remove, per_class_acc = create_production_model(
        wrapped, _val_fn, min_val_accuracy=CONFIG.get("min_production_accuracy", 0.3)
    )

    # --- Base mapping: remove only 0% accuracy classes ---
    base_keep = [idx for idx in sorted(wrapped.idx_to_label.keys())
                 if per_class_acc.get(idx, 0.0) > 0.0]
    base_remove = [idx for idx in sorted(wrapped.idx_to_label.keys())
                   if per_class_acc.get(idx, 0.0) == 0.0]

    if base_remove:
        # Overwrite the default mapping with 0%-filtered version
        wrapped.save_filtered_mapping(
            CONFIG["model_save_path"], base_keep,
            per_class_acc=per_class_acc, suffix="",
        )
        print(f"\n  Base mapping: removed {len(base_remove)} classes with 0% accuracy:")
        for ri in base_remove:
            print(f"    ❌ {wrapped.idx_to_label.get(ri, f'class_{ri}')}")

    # --- Production mapping: remove classes below threshold ---
    if remove:
        wrapped.save_filtered_mapping(
            CONFIG["model_save_path"], keep,
            per_class_acc=per_class_acc, suffix="_production",
        )

    prod_path = CONFIG["model_save_path"].replace(".pth", "_production_mapping.json")

    print(f"\n✅ Done!")
    print(f"  Weights:             {CONFIG['model_save_path']} ({len(wrapped.label_to_idx)} classes total)")
    print(f"  Base mapping:        {CONFIG['model_save_path'].replace('.pth', '_mapping.json')} "
          f"({len(base_keep)} classes, 0% removed)")
    print(f"  Production mapping:  {prod_path} "
          f"({len(keep)} classes, <{CONFIG.get('min_production_accuracy', 0.3):.0%} removed)")

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()