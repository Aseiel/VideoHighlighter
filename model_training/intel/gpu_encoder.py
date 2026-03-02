"""
GPU Feature Extractor
=====================

Drop-in replacement for IntelFeatureExtractor that runs entirely on
GPU via PyTorch instead of frame-by-frame on CPU via OpenVINO.

Same interface:
    .encode(frames_batch)   (B, T, C, H, W) → (B, T, feat_dim)

Place in:  model_training/intel/gpu_encoder.py
"""

import torch
import torch.nn as nn
import torchvision.models as models


# Backbones with their feature dimensions
BACKBONES = {
    "efficientnet_b0": 1280,
    "efficientnet_b2": 1408,
    "resnet50":        2048,
    "resnet34":        512,
    "resnet18":        512,
    "mobilenet_v3":    960,
}


def _build_backbone(name):
    """Build a backbone and remove its classification head."""
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        return nn.Sequential(m.features, m.avgpool, nn.Flatten())
    elif name == "efficientnet_b2":
        m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        return nn.Sequential(m.features, m.avgpool, nn.Flatten())
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        return nn.Sequential(*list(m.children())[:-1], nn.Flatten())
    elif name == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        return nn.Sequential(*list(m.children())[:-1], nn.Flatten())
    elif name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        return nn.Sequential(*list(m.children())[:-1], nn.Flatten())
    elif name == "mobilenet_v3":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        return nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    else:
        raise ValueError(f"Unknown backbone '{name}'. Available: {list(BACKBONES.keys())}")


class GPUFeatureExtractor:
    """
    GPU-native feature extractor.

    Same .encode() signature as IntelFeatureExtractor:
        encode( (B, T, C, H, W) ) → (B, T, feat_dim)   FloatTensor

    Everything stays on GPU — no CPU round-trips.
    """

    def __init__(self, backbone="efficientnet_b0", device=None):
        # ----- device -----
        if device is None:
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = 'xpu'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        self.device = torch.device(device)

        # Intel XPU → bfloat16, CUDA → float16, CPU → float32
        if device == 'xpu':
            self.dtype = torch.bfloat16
        elif device == 'cuda':
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # ----- backbone -----
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone '{backbone}'. "
                             f"Available: {list(BACKBONES.keys())}")

        self.feature_dim = BACKBONES[backbone]
        print(f"🔧 Loading {backbone} on {device} (dtype={self.dtype})")

        self.model = _build_backbone(backbone)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device, dtype=self.dtype)

        # ImageNet normalization constants (on device for speed)
        self.mean = torch.tensor([0.485, 0.456, 0.406],
                                 device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225],
                                device=self.device, dtype=self.dtype).view(1, 3, 1, 1)

        # Warmup pass
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=self.device, dtype=self.dtype)
            dummy = (dummy - self.mean) / self.std
            out = self.model(dummy)
            assert out.shape[-1] == self.feature_dim, \
                f"Expected {self.feature_dim}, got {out.shape[-1]}"

        print(f"  ✅ {backbone} ready — feature_dim={self.feature_dim}")

    @torch.no_grad()
    def encode(self, frames_batch):
        """
        Encode a batch of frame sequences.

        Args:
            frames_batch: (B, T, C, H, W) tensor — values in [0, 1]
                          Can be on any device; will be moved to self.device.

        Returns:
            torch.FloatTensor of shape (B, T, feat_dim)
        """
        if not isinstance(frames_batch, torch.Tensor):
            frames_batch = torch.from_numpy(frames_batch)

        B, T, C, H, W = frames_batch.shape

        # Flatten temporal dim into batch: (B*T, C, H, W)
        x = frames_batch.reshape(B * T, C, H, W)

        # Move to GPU + cast in one call
        x = x.to(self.device, dtype=self.dtype, non_blocking=True)

        # Normalize (on GPU — no CPU cost)
        x = (x - self.mean) / self.std

        # Forward — all B*T frames in one shot
        feats = self.model(x)  # (B*T, feat_dim)

        # Reshape back to (B, T, feat_dim) and return float32
        feats = feats.reshape(B, T, self.feature_dim)
        return feats.float()