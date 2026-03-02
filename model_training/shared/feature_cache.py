"""
Feature Cache
=============

Pre-encodes all video clips with the OpenVINO encoder ONCE and caches
the feature tensors to disk. Training then loads pure tensors — no
video decoding, no OpenVINO, no CPU bottleneck.

Flow:
    1. First run:  VideoDataset → OpenVINO encoder → save .pt cache (~12 min)
    2. Every epoch: Load cached tensors → LSTM on GPU (~2-3 min)

Place in:  model_training/shared/feature_cache.py
"""

import os
import hashlib
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def build_cache_path(dataset, config, cache_dir=None):
    """Deterministic cache filename based on dataset + config."""
    if cache_dir is None:
        cache_dir = config.get("checkpoint_dir", "checkpoints_intel")
    os.makedirs(cache_dir, exist_ok=True)

    h = hashlib.md5()
    h.update(str(len(dataset)).encode())
    h.update(str(config.get("sequence_length", 16)).encode())
    h.update(str(config.get("crop_size", (224, 224))).encode())
    h.update(str(config.get("default_stride", 4)).encode())
    # Include a few sample paths for uniqueness
    for sample in dataset.samples[:20]:
        h.update(str(sample).encode())

    return os.path.join(cache_dir, f"feature_cache_{h.hexdigest()[:10]}.pt")


def precompute_feature_cache(dataset, encoder, config,
                             cache_dir=None, force_rebuild=False):
    """
    Encode every clip with OpenVINO and save features + labels to disk.

    Args:
        dataset:        VideoDataset (returns frames, label)
        encoder:        IntelFeatureExtractor (.encode())
        config:         CONFIG dict
        cache_dir:      Where to store the cache file
        force_rebuild:  Delete existing cache and rebuild

    Returns:
        cache_path (str)
    """
    cache_path = build_cache_path(dataset, config, cache_dir)

    if os.path.exists(cache_path) and not force_rebuild:
        info = torch.load(cache_path, weights_only=False, map_location='cpu')
        print(f"✅ Feature cache loaded: {cache_path}")
        print(f"   {info['num_samples']} samples, feature dim = {info['feature_dim']}")
        return cache_path

    if force_rebuild and os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"🗑️  Deleted old feature cache")

    print(f"\n🔧 Pre-encoding {len(dataset)} clips with OpenVINO encoder...")
    print(f"   (One-time cost — all future epochs load from cache)")

    # Use batch_size=1, num_workers=0 for safety with OpenVINO
    temp_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for frames, labels in tqdm(temp_loader, desc="Encoding features"):
            # frames: (1, T, C, H, W) — encoder expects CPU tensors
            feats = encoder.encode(frames.cpu())  # → (1, T, feat_dim)
            all_features.append(feats.squeeze(0).cpu())  # → (T, feat_dim)
            all_labels.append(labels.squeeze(0).cpu())    # → scalar

    # Stack into single tensors
    features_tensor = torch.stack(all_features)  # (N, T, feat_dim)
    labels_tensor = torch.stack(all_labels)       # (N,)

    cache_data = {
        'features': features_tensor,
        'labels': labels_tensor,
        'num_samples': len(all_features),
        'feature_dim': features_tensor.shape[-1],
        'sequence_length': features_tensor.shape[1],
    }

    torch.save(cache_data, cache_path)
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"✅ Feature cache saved: {cache_path} ({size_mb:.1f} MB)")
    print(f"   {cache_data['num_samples']} samples, "
          f"feature dim = {cache_data['feature_dim']}, "
          f"seq len = {cache_data['sequence_length']}")

    return cache_path


class CachedFeatureDataset(Dataset):
    """
    Dataset that loads pre-encoded features from a cache file.
    Returns (features, label) where features is (T, feat_dim).

    Drop-in replacement for VideoDataset in the training loop —
    but the training loop skips encoder.encode() since features
    are already encoded.
    """

    def __init__(self, cache_path, sample_indices=None,
                 label_to_idx=None, idx_to_label=None):
        """
        Args:
            cache_path:     Path to the .pt cache file
            sample_indices: Optional list of indices to use (for train/val split)
            label_to_idx:   Optional label mapping from parent dataset
            idx_to_label:   Optional label mapping from parent dataset
        """
        cache = torch.load(cache_path, weights_only=False, map_location='cpu')
        self.all_features = cache['features']  # (N, T, feat_dim)
        self.all_labels = cache['labels']       # (N,)
        self.feature_dim = cache['feature_dim']
        self.num_total = cache['num_samples']

        if sample_indices is not None:
            self.indices = sample_indices
        else:
            self.indices = list(range(self.num_total))

        # For compatibility with compute_class_weights and other utils
        self.samples = [(i, int(self.all_labels[i].item())) for i in self.indices]

        unique_labels = sorted(set(int(self.all_labels[i].item()) for i in self.indices))

        if label_to_idx is not None and idx_to_label is not None:
            self.label_to_idx = label_to_idx
            self.idx_to_label = idx_to_label
            self.labels = sorted(label_to_idx.keys(), key=lambda k: label_to_idx[k])
        else:
            self.labels = [str(l) for l in unique_labels]
            self.idx_to_label = {l: str(l) for l in unique_labels}
            self.label_to_idx = {str(l): l for l in unique_labels}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.all_features[real_idx], self.all_labels[real_idx]