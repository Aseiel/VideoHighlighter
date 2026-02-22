"""
R3D Model Components
=====================

- build_r3d_model: Load pretrained R3D variant + replace FC head
- freeze/unfreeze helpers
- ONNX export
"""

import torch
import torch.nn as nn
import torchvision.models.video as video_models


def build_r3d_model(num_classes, config):
    """
    Build an R3D model with the final FC layer replaced.

    Args:
        num_classes: Number of output classes
        config: Dict with keys: model_variant, pretrained, dropout, freeze_backbone, freeze_bn

    Returns:
        nn.Module on CPU (caller moves to device)
    """
    variant = config.get("model_variant", "r3d_18")
    pretrained = config.get("pretrained", True)

    weights = "DEFAULT" if pretrained else None
    print(f"🚀 Loading {variant} (pretrained={pretrained})...")

    if variant == "r3d_18":
        model = video_models.r3d_18(weights=weights)
    elif variant == "mc3_18":
        model = video_models.mc3_18(weights=weights)
    elif variant == "r2plus1d_18":
        model = video_models.r2plus1d_18(weights=weights)
    else:
        raise ValueError(f"Unknown variant: {variant}. Use r3d_18 / mc3_18 / r2plus1d_18")

    # Replace FC head
    in_features = model.fc.in_features
    dropout = config.get("dropout", 0.4)
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    print(f"   FC head: {in_features} → Dropout({dropout}) → {num_classes}")

    # Freeze backbone if requested
    if config.get("freeze_backbone", False):
        freeze_backbone(model)

    # Freeze BatchNorm
    if config.get("freeze_bn", True):
        freeze_batchnorm(model)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Params: {total:,} total, {trainable:,} trainable")

    return model


def freeze_backbone(model):
    """Freeze all layers except the FC head."""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
    print("   ❄️  Backbone frozen — only FC head is trainable")


def unfreeze_backbone(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
    print("   🔓 Backbone unfrozen — all layers trainable")


def freeze_batchnorm(model):
    """
    Set all BatchNorm layers to eval mode so running stats stay frozen.
    Call this AFTER model.train() in the training loop.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def get_parameter_groups(model, config):
    """
    Split parameters into backbone and head groups for differential LR.

    Returns:
        List of param-group dicts for the optimizer.
    """
    base_lr = config.get("base_learning_rate", 1e-4)
    backbone_factor = config.get("backbone_lr_factor", 0.1)

    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": base_lr * backbone_factor})
    groups.append({"params": head_params, "lr": base_lr})

    print(f"   Optimizer groups: backbone LR={base_lr * backbone_factor:.6f}, head LR={base_lr:.6f}")
    return groups


# =============================
# ONNX Export
# =============================
def export_onnx(model, num_classes, config, output_path="r3d_finetuned.onnx"):
    """Export model to ONNX format."""
    model.eval()
    model.cpu()
    model.float()

    seq_len = config.get("sequence_length", 16)
    h, w = config.get("crop_size", (112, 112))
    dummy = torch.randn(1, 3, seq_len, h, w)

    print(f"📦 Exporting ONNX: {output_path}")
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=14,
    )
    print(f"✅ ONNX saved: {output_path}")
    return output_path
