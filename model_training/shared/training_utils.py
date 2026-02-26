"""
Shared Training Utilities
==========================

Seed management, class weight computation, checkpoint save/load,
production model filtering, and the ActionRecognitionModel wrapper.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from collections import Counter


# =============================
# Reproducibility
# =============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================
# Class Weights
# =============================
def compute_class_weights(dataset):
    """Inverse-frequency weights for balanced training."""
    label_counts = {}
    for _, label in dataset.samples:
        label_counts[label] = label_counts.get(label, 0) + 1

    total = len(dataset)
    num_classes = len(dataset.labels)

    weights = []
    for idx in range(num_classes):
        count = label_counts.get(idx, 1)
        weights.append(total / (num_classes * count))

    print(f"\n⚖️  Class weights (inverse frequency):")
    for idx, w in enumerate(weights):
        name = dataset.idx_to_label.get(idx, f"Class_{idx}")
        count = label_counts.get(idx, 0)
        print(f"   {name}: {count} samples, weight {w:.4f}")

    return torch.FloatTensor(weights)


# =============================
# Checkpoint Management
# =============================
def save_checkpoint(model, optimizer, epoch, best_val_acc, label_to_idx,
                    idx_to_label, feature_dim, checkpoint_path,
                    best_val_loss=None, extra=None):
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss if best_val_loss is not None else float("inf"),
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "feature_dim": feature_dim,
        "num_classes": len(label_to_idx),
    }
    if extra:
        ckpt.update(extra)

    torch.save(ckpt, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path} (epoch {epoch + 1})")


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Load a training checkpoint.

    Handles class-count mismatch gracefully (transfer-learning mode):
    copies all shared layers and reinitialises the final classifier.
    """
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None

    print(f"📂 Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    saved_classes = ckpt.get("num_classes", 0)

    # Detect final-layer name heuristically
    final_keys = _detect_classifier_keys(model)

    current_classes = _count_output_classes(model, final_keys)

    if saved_classes != current_classes and saved_classes > 0:
        print(f"⚠️  Class mismatch: checkpoint={saved_classes}, current={current_classes}")
        print("   Loading shared weights only (transfer learning)")

        state = ckpt["model_state_dict"]
        model_dict = model.state_dict()
        filtered = {
            k: v
            for k, v in state.items()
            if k in model_dict
            and v.shape == model_dict[k].shape
            and not any(fk in k for fk in final_keys)
        }
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        print(f"   ✅ Loaded {len(filtered)} shared layers")
        return {
            "epoch": -1,
            "best_val_acc": 0.0,
            "best_val_loss": float("inf"),
            "label_to_idx": ckpt.get("label_to_idx", {}),
            "idx_to_label": ckpt.get("idx_to_label", {}),
            "transfer_learning": True,
        }

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and ckpt.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"⚠️  Could not load optimizer state: {e}")

    print(f"✅ Checkpoint loaded (epoch {ckpt['epoch'] + 1}, "
          f"best_val_acc={ckpt.get('best_val_acc', 0):.4f})")
    return ckpt


def _detect_classifier_keys(model):
    """Return substrings that identify the final classification layer."""
    # Common patterns across architectures
    for name, _ in model.named_parameters():
        if "fc." in name or "classifier." in name or "head." in name:
            prefix = name.split(".")[0]
            return [prefix]
    return ["fc", "classifier", "head"]


def _count_output_classes(model, final_keys):
    """Try to detect the number of output classes from the model."""
    for name, param in model.named_parameters():
        if any(fk in name for fk in final_keys) and "weight" in name:
            return param.shape[0]
    return 0


# =============================
# ActionRecognitionModel wrapper
# =============================
class ActionRecognitionModel:
    """
    Wrapper that bundles a trained model with its label mappings
    and metadata. Works for both Intel LSTM and R3D models.
    """

    def __init__(self, model, label_to_idx, idx_to_label,
                 feature_dim, sequence_length, model_type="EncoderLSTM",
                 extra_meta=None):
        self.model = model
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.model_type = model_type
        self.extra_meta = extra_meta or {}

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        mapping_path = path.replace(".pth", "_mapping.json")
        data = {
            "label_to_idx": self.label_to_idx,
            "idx_to_label": {str(k): v for k, v in self.idx_to_label.items()},
            "feature_dim": self.feature_dim,
            "sequence_length": self.sequence_length,
            "model_type": self.model_type,
            "num_classes_total": len(self.label_to_idx),
        }
        data.update(self.extra_meta)
        with open(mapping_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Model saved: {path}")
        print(f"✅ Mapping saved: {mapping_path}")

    def save_filtered_mapping(self, path, classes_to_keep, per_class_acc=None, suffix=""):
        """
        Save a mapping JSON that only includes specified classes.
        Weights file is NOT re-saved — same .pth works with any mapping.
        
        Args:
            path: Base .pth path (used to derive mapping filename)
            classes_to_keep: List of class indices to include
            per_class_acc: Optional per-class accuracy dict
            suffix: Mapping filename suffix (e.g. '_production')
        """
        keep_set = set(classes_to_keep)
        filtered_l2i = {name: idx for name, idx in self.label_to_idx.items()
                        if idx in keep_set}
        filtered_i2l = {str(k): v for k, v in self.idx_to_label.items()
                        if k in keep_set}
        removed = [self.idx_to_label[i] for i in sorted(
            set(self.idx_to_label.keys()) - keep_set)]

        mapping_path = path.replace(".pth", f"{suffix}_mapping.json")
        data = {
            "label_to_idx": filtered_l2i,
            "idx_to_label": filtered_i2l,
            "feature_dim": self.feature_dim,
            "sequence_length": self.sequence_length,
            "model_type": self.model_type,
            "num_classes_total": len(self.label_to_idx),
            "num_classes_active": len(filtered_l2i),
            "removed_classes": removed,
        }
        if per_class_acc:
            data["per_class_accuracy"] = {
                self.idx_to_label.get(k, str(k)): round(v, 4)
                for k, v in per_class_acc.items()
            }
        data.update(self.extra_meta)

        with open(mapping_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Mapping saved: {mapping_path} "
              f"({len(filtered_l2i)} active / {len(self.label_to_idx)} total)")

    @staticmethod
    def load_mapping(path):
        mapping_path = path.replace(".pth", "_mapping.json")
        with open(mapping_path, "r") as f:
            return json.load(f)


# =============================
# Production Model Filtering
# =============================
def create_production_model(model_wrapper, val_fn, min_val_accuracy=0.3):
    """
    Remove classes that don't meet a minimum validation accuracy.

    Args:
        model_wrapper: ActionRecognitionModel
        val_fn: Callable that returns (per_class_acc: dict[int, float],
                                       class_predictions: dict[int, list])
                Called with the current model — pipeline-specific.
        min_val_accuracy: Threshold (0-1)

    Returns:
        (classes_to_keep: list[int], classes_to_remove: list[int],
         per_class_acc: dict)
    """
    print(f"\n🔍 PRODUCTION MODEL FILTERING")
    print(f"   Min validation accuracy: {min_val_accuracy:.1%}")
    print("=" * 70)

    per_class_acc, class_predictions = val_fn()

    keep, remove = [], []

    print(f"\n{'Action':<30} {'Acc':<10} {'Decision'}")
    print("-" * 70)

    for cls_idx in sorted(per_class_acc.keys()):
        name = model_wrapper.idx_to_label.get(cls_idx, f"class_{cls_idx}")
        acc = per_class_acc[cls_idx]

        if acc >= min_val_accuracy:
            keep.append(cls_idx)
            decision = "✅ KEEP"
        else:
            remove.append(cls_idx)
            confused = ""
            if cls_idx in class_predictions:
                preds = class_predictions[cls_idx]
                common = Counter(preds).most_common(2)
                confused_names = [
                    model_wrapper.idx_to_label.get(i, "?")
                    for i, _ in common if i != cls_idx
                ]
                if confused_names:
                    confused = f" (confused with: {', '.join(confused_names[:2])})"
            decision = f"❌ REMOVE{confused}"

        print(f"{name:<30} {acc:<10.4f} {decision}")

    print(f"\n   Keep: {len(keep)} | Remove: {len(remove)}")
    return keep, remove, per_class_acc


def build_filtered_label_maps(classes_to_keep, old_idx_to_label):
    """
    Build new contiguous label maps from a list of old indices to keep.

    Returns:
        (new_label_to_idx, new_idx_to_label, old_to_new_idx)
    """
    new_l2i, new_i2l, old2new = {}, {}, {}
    for new_idx, old_idx in enumerate(sorted(classes_to_keep)):
        name = old_idx_to_label[old_idx]
        new_l2i[name] = new_idx
        new_i2l[new_idx] = name
        old2new[old_idx] = new_idx
    return new_l2i, new_i2l, old2new
