"""
Shared Dataset Utilities
=========================

Video dataset loading, frame sampling, ROI-based cropping,
validation/splitting, and augmentation helpers.

PERFORMANCE: ROI pre-computation cache
  - Before training, run precompute_roi_cache() ONCE
  - Runs YOLO + pose on all videos, saves ROIs to a pickle file
  - During training, load_video_cached() skips all detection — just reads
    frames, applies cached ROIs, crops, normalises
  - Typical speedup: 5-10x per epoch
"""

import os
import glob
import cv2
import random
import pickle
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .detection import (
    PersonTracker,
    SmartActionDetector,
    AdaptiveActionDetector,
    SmoothedROIDetector,
    PoseExtractor,
    merge_boxes,
    crop_roi,
    get_yolo_people_model,
)


# =============================
# Frame Index Computation
# =============================
def compute_frame_indices(total_frames, config):
    """Deterministic frame indices for a given video length."""
    seq_len = config["sequence_length"]
    strategy = config.get("sampling_strategy", "temporal_stride")

    if strategy == "temporal_stride":
        stride = config.get("default_stride", 4)
        stride = max(
            config.get("min_stride", 3),
            min(config.get("max_stride", 8), total_frames // (seq_len * 2)),
        )
        indices = np.arange(0, min(seq_len * stride, total_frames), stride)
        indices = indices[:seq_len]
    else:
        indices = np.linspace(0, total_frames - 1, seq_len).astype(int)

    return indices


# ==========================================================
# ROI Pre-Computation Cache
# ==========================================================
def precompute_roi_cache(dataset, config, cache_path=None, pose_extractor=None):
    """
    Run YOLO + pose detection on all videos ONCE and cache the ROIs.

    This is the single biggest training speedup — eliminates all neural-net
    detection from the data-loading hot path.

    Args:
        dataset: VideoDataset instance
        config: Training config dict
        cache_path: Where to save the pickle (default: auto-generated)
        pose_extractor: Optional PoseExtractor for adaptive cropping

    Returns:
        dict: {video_path: {int(frame_idx): (x1,y1,x2,y2) or None}}
    """
    if cache_path is None:
        h = hashlib.md5()
        h.update(str(config.get("sequence_length", 16)).encode())
        h.update(str(config.get("default_stride", 4)).encode())
        h.update(str(config.get("crop_size", (112, 112))).encode())
        h.update(str(len(dataset.samples)).encode())
        cache_path = os.path.join(
            config.get("checkpoint_dir", "."),
            f"roi_cache_{h.hexdigest()[:8]}.pkl",
        )

    # Check existing cache
    if os.path.exists(cache_path):
        print(f"📦 Loading ROI cache: {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        # Quick validation
        sample_paths = [vp for vp, _ in dataset.samples[:5]]
        hits = sum(1 for p in sample_paths if p in cache)
        if hits == len(sample_paths):
            print(f"   ✅ Cache valid ({len(cache)} videos)")
            return cache
        else:
            print(f"   ⚠️  Cache stale ({hits}/{len(sample_paths)} matched) — rebuilding")

    print(f"\n🔍 PRE-COMPUTING ROIs for {len(dataset.samples)} videos...")
    print(f"   This runs YOLO + pose ONCE so training epochs are fast.")
    print(f"   Cache: {cache_path}\n")

    yolo_model = get_yolo_people_model()
    cache = {}

    for video_path, _ in tqdm(dataset.samples, desc="Caching ROIs"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cache[video_path] = {}
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            cache[video_path] = {}
            continue

        indices = compute_frame_indices(total_frames, config)

        # Fresh detectors per video (no state leakage)
        action_det = SmartActionDetector(debug=False)
        adaptive_det = AdaptiveActionDetector(
            motion_threshold=config.get("motion_threshold", 5.0), debug=False
        )
        roi_smoother = (
            SmoothedROIDetector(
                window_size=5,
                base_alpha=config.get("smoothing_base_alpha", 0.5),
                adaptive=config.get("adaptive_smoothing", True),
            )
            if config.get("use_roi_smoothing", True)
            else None
        )
        tracker = PersonTracker(iou_threshold=0.3, max_lost_frames=5)

        video_rois = {}

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                video_rois[int(idx)] = None
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            tracked = action_det.detect_with_tracking(
                frame_rgb, yolo_model, tracker,
                max_people=config.get("max_action_people", 2),
            )
            people = (
                [box for _, box in tracked]
                if tracked and isinstance(tracked[0], tuple)
                else tracked
            )

            roi = None
            if (
                config.get("use_adaptive_cropping", False)
                and pose_extractor is not None
                and people
            ):
                roi, _ = adaptive_det.detect_action_region(
                    frame_rgb, people, pose_extractor, max_poses=2
                )
            elif people:
                roi = merge_boxes(people)

            if roi_smoother and roi is not None:
                roi = roi_smoother.update(roi)

            video_rois[int(idx)] = tuple(roi) if roi is not None else None

        cap.release()
        cache[video_path] = video_rois

    # Save
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(cache_path) / 1e6
    print(f"\n✅ ROI cache saved: {cache_path} ({size_mb:.1f} MB, {len(cache)} videos)")
    return cache


# ==========================================================
# FAST Video Loading (cached ROIs — no YOLO/pose)
# ==========================================================
def load_video_cached(path, config, roi_cache, is_training=True):
    """
    Load video using pre-computed ROIs. No YOLO, no pose, no tracker.
    This is the fast path used during training.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    seq_len = config["sequence_length"]
    crop_size = config["crop_size"]
    indices = compute_frame_indices(total_frames, config)

    video_rois = roi_cache.get(path, {}) if roi_cache else {}

    output_frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            if output_frames:
                output_frames.append(output_frames[-1].copy())
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = video_rois.get(int(idx))
        cropped = crop_roi(frame_rgb, roi, crop_size)

        if is_training and random.random() < config.get("augmentation_prob", 0.3):
            cropped = _apply_augmentation(cropped, config)

        cropped = cropped.astype(np.float32) / 255.0
        mean = np.array(config["mean"], dtype=np.float32)
        std = np.array(config["std"], dtype=np.float32)
        cropped = (cropped - mean) / std
        output_frames.append(cropped)

    cap.release()

    if not output_frames:
        return []

    while len(output_frames) < seq_len:
        remaining = seq_len - len(output_frames)
        to_add = min(remaining, len(output_frames))
        output_frames.extend(output_frames[:to_add])

    return np.stack(output_frames[:seq_len], axis=0)


# ==========================================================
# SLOW Video Loading (live detection — visualization / fallback)
# ==========================================================
def load_video_normalized(path, config, pose_extractor=None, is_training=True,
                          verbose=False, debug=False):
    """
    Load video with LIVE YOLO + pose detection (slow).
    Use load_video_cached() during training instead.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    seq_len = config["sequence_length"]
    crop_size = config["crop_size"]
    indices = compute_frame_indices(total_frames, config)

    yolo_people = get_yolo_people_model()
    action_detector = SmartActionDetector(debug=debug)
    adaptive_detector = AdaptiveActionDetector(
        motion_threshold=config.get("motion_threshold", 5.0), debug=debug
    )
    roi_smoother = (
        SmoothedROIDetector(
            window_size=5,
            base_alpha=config.get("smoothing_base_alpha", 0.5),
            adaptive=config.get("adaptive_smoothing", True),
            debug=config.get("debug_smoothing", False) or debug,
        )
        if config.get("use_roi_smoothing", True)
        else None
    )
    person_tracker = PersonTracker(iou_threshold=0.3, max_lost_frames=5)

    output_frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if not success:
            if output_frames:
                output_frames.append(output_frames[-1].copy())
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        tracked = action_detector.detect_with_tracking(
            frame_rgb, yolo_people, person_tracker,
            max_people=config.get("max_action_people", 2),
        )
        people = (
            [box for _, box in tracked]
            if tracked and isinstance(tracked[0], tuple)
            else tracked
        )

        roi = None
        if (
            config.get("use_adaptive_cropping", False)
            and pose_extractor is not None
            and people
        ):
            roi, _ = adaptive_detector.detect_action_region(
                frame_rgb, people, pose_extractor, max_poses=2
            )
        elif people:
            roi = merge_boxes(people)

        if roi_smoother and roi is not None:
            roi = roi_smoother.update(roi)

        cropped = crop_roi(frame_rgb, roi, crop_size)

        if is_training and random.random() < config.get("augmentation_prob", 0.3):
            cropped = _apply_augmentation(cropped, config)

        cropped = cropped.astype(np.float32) / 255.0
        mean = np.array(config["mean"], dtype=np.float32)
        std = np.array(config["std"], dtype=np.float32)
        cropped = (cropped - mean) / std
        output_frames.append(cropped)

    cap.release()

    if not output_frames:
        return []

    while len(output_frames) < seq_len:
        remaining = seq_len - len(output_frames)
        to_add = min(remaining, len(output_frames))
        output_frames.extend(output_frames[:to_add])

    return np.stack(output_frames[:seq_len], axis=0)


# ==========================================================
# Augmentation
# ==========================================================
def _apply_augmentation(frame, config):
    """Apply a single random augmentation to a uint8 RGB frame."""
    aug = random.random()
    if aug < 0.33:
        factor = random.uniform(0.85, 1.15)
        frame = np.clip(frame * factor, 0, 255).astype(np.uint8)
    elif aug < 0.66:
        factor = random.uniform(0.85, 1.15)
        mean = frame.mean(axis=(0, 1), keepdims=True)
        frame = np.clip((frame - mean) * factor + mean, 0, 255).astype(np.uint8)
    else:
        frame = np.fliplr(frame).copy()
    return frame


# ==========================================================
# VideoDataset
# ==========================================================
class VideoDataset(Dataset):
    """
    Generic video dataset. Two loading modes:

    1. CACHED (fast) — set roi_cache from precompute_roi_cache()
       No YOLO/pose in __getitem__. Safe for num_workers > 0.

    2. LIVE (slow) — roi_cache is None, runs detection per frame.
       Only for visualization or when caching isn't desired.
    """

    def __init__(self, root, config, pose_extractor=None, is_training=True,
                 roi_cache=None):
        self.config = config
        self.pose_extractor = pose_extractor
        self.is_training = is_training
        self.roi_cache = roi_cache
        self.samples = []

        if not os.path.exists(root):
            print(f"⚠️  Dataset path {root} does not exist")
            self.labels = []
            self.label_to_idx = {}
            self.idx_to_label = {}
            return

        class_folders = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        self.label_to_idx = {label: idx for idx, label in enumerate(class_folders)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.labels = class_folders

        print(f"📊 Detected {len(self.labels)} action classes:")
        for label, idx in self.label_to_idx.items():
            print(f"  {idx}: {label}")

        video_count = 0
        for label in self.labels:
            label_path = os.path.join(root, label)
            videos = (
                glob.glob(os.path.join(label_path, "*.mp4"))
                + glob.glob(os.path.join(label_path, "*.avi"))
                + glob.glob(os.path.join(label_path, "*.mov"))
            )
            for vp in videos:
                self.samples.append((vp, self.label_to_idx[label]))
                video_count += 1

        print(f"✅ Found {video_count} videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        if self.roi_cache is not None:
            # FAST PATH
            frames = load_video_cached(
                video_path,
                config=self.config,
                roi_cache=self.roi_cache,
                is_training=self.is_training,
            )
        else:
            # SLOW PATH
            use_pose = self.config.get("use_pose_guided_crop") or self.config.get(
                "use_adaptive_cropping"
            )
            frames = load_video_normalized(
                video_path,
                config=self.config,
                pose_extractor=self.pose_extractor if use_pose else None,
                is_training=self.is_training,
                verbose=False,
            )

        if len(frames) == 0:
            seq_len = self.config["sequence_length"]
            h, w = self.config["crop_size"]
            frames = np.zeros((seq_len, h, w, 3), dtype=np.float32)

        # (T, H, W, C) -> (T, C, H, W)
        frames = np.transpose(frames, (0, 3, 1, 2))
        return (
            torch.tensor(frames, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )

    def get_label_mapping(self):
        return self.label_to_idx, self.idx_to_label


# ==========================================================
# Dataset Validation & Auto-Split
# ==========================================================
def validate_and_split_dataset(train_dataset, val_dataset, config):
    """
    Check dataset sizes and auto-split if validation is insufficient.

    Returns:
        (is_valid, valid_actions, new_train_samples, new_val_samples)
    """
    from sklearn.model_selection import train_test_split

    min_train = config.get("min_train_per_action", 5)
    min_val = config.get("min_val_per_action", 2)

    print(f"\n📊 Validating dataset sizes...")
    print(f"  Min train videos/action: {min_train}")
    print(f"  Min val videos/action:   {min_val}")

    train_counts, val_counts = {}, {}
    for vp, lbl in train_dataset.samples:
        action = train_dataset.idx_to_label[lbl]
        train_counts.setdefault(action, []).append((vp, lbl))
    for vp, lbl in val_dataset.samples:
        action = val_dataset.idx_to_label[lbl]
        val_counts.setdefault(action, []).append((vp, lbl))

    valid_actions = []
    new_train, new_val = [], []

    for action in train_dataset.labels:
        t_vids = train_counts.get(action, [])
        v_vids = val_counts.get(action, [])
        t_cnt, v_cnt = len(t_vids), len(v_vids)
        total = t_cnt + v_cnt

        if t_cnt < min_train:
            print(f"  ❌ '{action}': {t_cnt} train (need {min_train}) — SKIPPED")
            continue

        # Enforce minimum val ratio (default 20%)
        min_val_ratio = config.get("min_val_ratio", 0.2)
        current_val_ratio = v_cnt / total if total > 0 else 0.0
        needs_resplit = v_cnt < min_val or current_val_ratio < min_val_ratio

        if needs_resplit:
            if total < min_train + min_val:
                print(f"  ⚠️  '{action}': {total} total (need {min_train + min_val}) — SKIPPED")
                continue
            reason = (
                f"{v_cnt} val" if v_cnt < min_val
                else f"{current_val_ratio:.0%} val ratio < {min_val_ratio:.0%}"
            )
            print(f"  🔄 '{action}': {t_cnt} train, {v_cnt} val → AUTO-SPLITTING ({reason})")
            all_vids = t_vids + v_vids
            val_ratio = max(min_val / total, min_val_ratio)
            val_ratio = min(val_ratio, 0.3)
            split_train, split_val = train_test_split(
                all_vids, test_size=val_ratio, random_state=42
            )
            new_train.extend(split_train)
            new_val.extend(split_val)
            valid_actions.append(action)
            print(f"     ✓ {len(split_train)} train, {len(split_val)} val")
        else:
            print(f"  ✅ '{action}': {t_cnt} train, {v_cnt} val — OK")
            new_train.extend(t_vids)
            new_val.extend(v_vids)
            valid_actions.append(action)

    if not valid_actions:
        print("\n❌ No actions meet minimum requirements.")
        return False, [], [], []

    print(f"\n✅ Validation complete: {len(valid_actions)} valid actions")
    print(f"  Training samples:   {len(new_train)}")
    print(f"  Validation samples: {len(new_val)}")
    return True, valid_actions, new_train, new_val


def apply_dataset_split(train_dataset, val_dataset, valid_actions,
                         new_train_samples, new_val_samples):
    """
    Apply the results of validate_and_split_dataset back to the datasets.
    Remaps labels so only valid_actions remain (contiguous 0..N-1).
    """
    new_label_to_idx = {a: i for i, a in enumerate(valid_actions)}
    new_idx_to_label = {i: a for a, i in new_label_to_idx.items()}

    def _remap(samples, old_idx_to_label):
        out = []
        for vp, old_lbl in samples:
            name = old_idx_to_label.get(old_lbl)
            if name in new_label_to_idx:
                out.append((vp, new_label_to_idx[name]))
        return out

    train_old = train_dataset.idx_to_label.copy()
    val_old = val_dataset.idx_to_label.copy()

    train_dataset.samples = _remap(new_train_samples, train_old)
    val_dataset.samples = _remap(new_val_samples, val_old)

    for ds in (train_dataset, val_dataset):
        ds.labels = valid_actions
        ds.label_to_idx = new_label_to_idx.copy()
        ds.idx_to_label = new_idx_to_label.copy()

    print(f"\n✅ Datasets updated:")
    print(f"  Train: {len(train_dataset.samples)} | Val: {len(val_dataset.samples)}")
    print(f"  Classes: {valid_actions}")