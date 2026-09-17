"""
YOLOX Object Detection Training (dataset prep + train_yolox handoff)

Builds a labeled dataset from labeler JSON exports, then delegates training to
the official YOLOX repo via ``training/train_yolox.py`` (Apache-2.0).

Pose/keypoint training is not supported here — YOLOX is detection only.
"""

import os
import json
import shutil
import cv2
import numpy as np
import subprocess
import sys
from pathlib import Path
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class YoloxDatasetBuilder:
    """
    Convert JSON labels to YOLO-format detection labels (standard layout for COCO export).
    """
    
    def __init__(self,
                 labels_dir,      # Directory with JSON label files
                 video_dir,       # Directory with video files
                 output_dir,      # Where to save YOLO format dataset
                 keypoint_names,  # List of class/keypoint names
                 img_size=640,
                 task="pose",     # "pose" = keypoints, "detect" = one box per labeled point
                 box_frac=0.12):  # detect-mode box size as a fraction of the frame

        self.labels_dir = Path(labels_dir)
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.keypoint_names = keypoint_names
        self.num_keypoints = len(keypoint_names)
        self.img_size = img_size
        self.task = task
        self.box_frac = float(box_frac)
        
        # Keypoint indices
        self.kp_to_idx = {name: i for i, name in enumerate(keypoint_names)}
        
        # Create output directories in the layout ultralytics expects:
        #   <output>/images/{train,val}/*.jpg  and  <output>/labels/{train,val}/*.txt
        # (the loader finds a label by swapping /images/ -> /labels/ in the path)
        self.img_train_dir = self.output_dir / 'images' / 'train'
        self.img_val_dir = self.output_dir / 'images' / 'val'
        self.lbl_train_dir = self.output_dir / 'labels' / 'train'
        self.lbl_val_dir = self.output_dir / 'labels' / 'val'

        for d in [self.img_train_dir, self.img_val_dir, self.lbl_train_dir, self.lbl_val_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _find_video(self, video_name):
        """Locate a video by name under video_dir (searched recursively, so videos
        organised in subfolders are still found)."""
        # direct hit
        cand = self.video_dir / video_name
        if cand.exists():
            return cand
        # recursive search by name / stem+ext
        stem = Path(video_name).stem
        matches = list(self.video_dir.rglob(video_name))
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            matches += list(self.video_dir.rglob(f"{stem}{ext}"))
        return matches[0] if matches else None
    
    def extract_frames_from_video(self, video_path, frame_indices, output_dir):
        """Extract specific frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"⚠️ Could not open: {video_path}")
            return []
        
        extracted = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for frame_idx in frame_indices:
            if frame_idx >= total_frames:
                continue
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Save frame as image
            img_name = f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
            img_path = output_dir / img_name
            cv2.imwrite(str(img_path), frame)
            extracted.append((img_path, frame_idx))
        
        cap.release()
        return extracted
    
    def convert_to_detect_format(self, frame_info, img_width, img_height):
        """Object-detection labels: one box per labeled point, each its own class.
        Each labeled point becomes a fixed-size box centered on it — reuses the
        existing point labels (and optical-flow/occlusion).
        Returns 'class_id xc yc w h' lines (normalised), or None if no points."""
        bw = bh = self.box_frac
        lines = []
        for name, raw in frame_info.get('points', {}).items():
            if name not in self.keypoint_names:
                continue
            idx = self.keypoint_names.index(name)
            # Support both old [x,y] and new [[x1,y1],[x2,y2]] formats
            instances = raw if (raw and isinstance(raw[0], list)) else [raw]
            for x, y in instances:
                xc = min(max(x / img_width, 0.0), 1.0)
                yc = min(max(y / img_height, 0.0), 1.0)
                lines.append(f"{idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        return "\n".join(lines) if lines else None

    def convert_label(self, frame_info, img_width, img_height):
        """Dispatch to the right label format for the configured task."""
        if self.task == "detect":
            return self.convert_to_detect_format(frame_info, img_width, img_height)
        return self.convert_to_yolo_format(frame_info, img_width, img_height)

    def convert_to_yolo_format(self, frame_info, img_width, img_height):
        """
        Convert keypoint coordinates to YOLO format:
        class_id x1 y1 x2 y2 kp1x kp1y kp1v kp2x kp2y kp2v ...
        
        Where:
        - class_id: 0 (single class for keypoint detection)
        - x1,y1,x2,y2: bounding box (normalized)
        - kpX,kpY: normalized keypoint coordinates (0-1)
        - kpV: visibility (0=not visible, 1=visible, 2=occluded)
        """
        # Get all visible keypoints
        visible_points = []
        for name in self.keypoint_names:
            if name in frame_info['points']:
                x, y = frame_info['points'][name]
                visible_points.append((x, y))
        
        if len(visible_points) < 2:
            return None  # Not enough points for a valid bbox
        
        # Calculate bounding box from keypoints with padding
        points = np.array(visible_points)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # Add padding: 20% of the keypoint spread, but at least an absolute minimum
        # (5% of the frame) so collinear/sparse keypoints (e.g. one point directly
        # above another) still yield a non-degenerate box instead of a zero-area line.
        pad_x = max((x_max - x_min) * 0.2, img_width * 0.05)
        pad_y = max((y_max - y_min) * 0.2, img_height * 0.05)
        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(img_width, x_max + pad_x)
        y2 = min(img_height, y_max + pad_y)
        
        # Normalize bounding box
        x1_norm = x1 / img_width
        y1_norm = y1 / img_height
        x2_norm = x2 / img_width
        y2_norm = y2 / img_height
        
        # Format: class_id x1 y1 x2 y2
        bbox_str = f"0 {x1_norm:.6f} {y1_norm:.6f} {x2_norm:.6f} {y2_norm:.6f}"
        
        # Add keypoints
        kp_str = []
        for name in self.keypoint_names:
            if name in frame_info['points']:
                x, y = frame_info['points'][name]
                # Normalize coordinates
                x_norm = x / img_width
                y_norm = y / img_height
                visibility = 2  # 2 = visible (YOLO uses 2 for visible)
                kp_str.append(f"{x_norm:.6f} {y_norm:.6f} {visibility}")
            else:
                # Keypoint not present
                kp_str.append("0.000000 0.000000 0")
        
        return f"{bbox_str} " + " ".join(kp_str)
    
    def build_dataset(self, train_ratio=0.8):
        """
        Build a YOLO-pose dataset from the labeler's JSON exports.

        Reads {video, keyframes:[{frame, points}]} files, extracts the labeled
        frames straight out of the videos, writes YOLO label .txt files, and
        splits everything into images/labels train+val folders.
        """
        print("📊 Building YOLO keypoint dataset...")

        label_files = list(self.labels_dir.rglob("*.json"))
        if not label_files:
            print(f"❌ No JSON files found in {self.labels_dir}")
            return None

        print(f"   Found {len(label_files)} label files")

        # Group labeled frames by their source video
        video_frames = {}
        for label_file in tqdm(label_files, desc="Reading labels"):
            with open(label_file, 'r') as f:
                data = json.load(f)

            video_name = data.get('video', '')
            if not video_name:
                continue

            video_path = self._find_video(video_name)
            if video_path is None:
                print(f"⚠️ Video not found: {video_name}")
                continue

            # Sanity check: labeler should export native video-pixel coords
            if data.get('coordinate_space') and data['coordinate_space'] != 'video_pixels':
                print(f"⚠️ {label_file.name}: coordinate_space="
                      f"{data['coordinate_space']!r} (expected 'video_pixels'). "
                      f"Re-export from the updated labeler or labels will be misaligned.")

            for frame_info in data.get('keyframes', []):
                frame_idx = frame_info.get('frame')
                points = frame_info.get('points', {})
                if frame_idx is None or not points:
                    continue
                video_frames.setdefault(video_path, []).append({
                    'frame': frame_idx,
                    'points': points,
                    'phase': frame_info.get('phase', 'unknown')
                })

        if not video_frames:
            print("❌ No usable labeled frames found.")
            return None

        print(f"   Found keypoints in {len(video_frames)} videos")

        # Stage extracted frames + labels, then split into train/val
        stage_img = self.output_dir / 'images' / '_staging'
        stage_lbl = self.output_dir / 'labels' / '_staging'
        stage_img.mkdir(parents=True, exist_ok=True)
        stage_lbl.mkdir(parents=True, exist_ok=True)

        stems = []
        seen_stems = set()
        for video_path, frames in tqdm(video_frames.items(), desc="Extracting frames"):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"⚠️ Could not open: {video_path}")
                continue
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for fi in frames:
                idx = fi['frame']
                if idx < 0 or idx >= total:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                h, w = frame.shape[:2]
                yolo_label = self.convert_label(fi, w, h)
                if yolo_label is None:
                    continue

                stem = f"{video_path.stem}_frame_{idx:06d}"
                cv2.imwrite(str(stage_img / f"{stem}.jpg"), frame)
                with open(stage_lbl / f"{stem}.txt", 'w') as f:
                    f.write(yolo_label + '\n')
                if stem not in seen_stems:   # same video+frame may appear in >1 label file
                    seen_stems.add(stem)
                    stems.append(stem)

            cap.release()

        if len(stems) < 2:
            print(f"❌ Only {len(stems)} valid sample(s) — need at least 2 (and "
                  f"realistically a few hundred). Label more frames first.")
            shutil.rmtree(stage_img, ignore_errors=True)
            shutil.rmtree(stage_lbl, ignore_errors=True)
            return None

        print(f"   Created {len(stems)} labeled samples")

        train_stems, val_stems = train_test_split(
            stems, test_size=1 - train_ratio, random_state=42
        )

        def _place(stem_list, img_dst, lbl_dst):
            for stem in stem_list:
                shutil.move(str(stage_img / f"{stem}.jpg"), str(img_dst / f"{stem}.jpg"))
                shutil.move(str(stage_lbl / f"{stem}.txt"), str(lbl_dst / f"{stem}.txt"))

        _place(train_stems, self.img_train_dir, self.lbl_train_dir)
        _place(val_stems, self.img_val_dir, self.lbl_val_dir)

        # Clean up staging
        shutil.rmtree(stage_img, ignore_errors=True)
        shutil.rmtree(stage_lbl, ignore_errors=True)

        print(f"   Train: {len(train_stems)} samples")
        print(f"   Val:   {len(val_stems)} samples")

        self.create_dataset_yaml()

        return {
            'train': len(train_stems),
            'val': len(val_stems),
            'total': len(stems),
            'keypoints': self.keypoint_names
        }

    def create_dataset_yaml(self):
        """Create dataset.yaml for YOLO training (detect: one class per name;
        pose: single 'object' class with N keypoints)."""
        if self.task == "detect":
            yaml_content = {
                'path': str(self.output_dir.absolute()),
                'train': 'images/train',
                'val': 'images/val',
                'nc': self.num_keypoints,
                'names': list(self.keypoint_names),
            }
        else:
            yaml_content = {
                'path': str(self.output_dir.absolute()),
                'train': 'images/train',
                'val': 'images/val',
                'nc': 1,  # single object class
                'names': ['object'],
                'kpt_shape': [self.num_keypoints, 3],  # [num_keypoints, (x, y, visibility)]
                'flip_idx': list(range(self.num_keypoints))  # identity = no L/R symmetry
            }

        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        print(f"✅ Dataset YAML saved: {yaml_path}")
        return yaml_path


# =============================
# MAIN USAGE
# =============================

if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="Prepare dataset and launch YOLOX training")
    _ap.add_argument("--yolox-dir", type=Path, default=None,
                     help="Path to cloned Megvii-BaseDetection/YOLOX repo (required to train)")
    _ap.add_argument("--size", default="s", choices=["nano", "tiny", "s", "m", "l", "x"],
                     help="YOLOX model size")
    _args = _ap.parse_args()

    ROOT = Path(__file__).resolve().parent.parent
    KEYPOINT_NAMES = []
    LABELS_DIR = str(ROOT / "dataset" / "train" / "labels")
    VIDEO_DIR = str(ROOT / "dataset")
    OUTPUT_DIR = str(ROOT / "yolox_dataset")
    MODEL_SIZE = _args.size
    TASK = "detect"  # object detection only via YOLOX
    BOX_FRAC = 0.12

    if TASK == "pose":
        print("❌ Pose/keypoint training is not available (YOLOX is detection-only).")
        print("   Use TASK='detect' for object classes, or add RTMPose later for keypoints.")
        raise SystemExit(1)

    if not Path(LABELS_DIR).exists():
        print(f"❌ Labels folder '{LABELS_DIR}' not found.\n"
              f"   Export labels from labeler.py, then re-run.")
        raise SystemExit(1)

    _label_files = sorted(Path(LABELS_DIR).rglob("*.json"))
    if _label_files:
        try:
            with open(_label_files[0], 'r', encoding='utf-8') as _f:
                _names = json.load(_f).get('keypoint_names')
            if _names:
                KEYPOINT_NAMES = _names
                print(f"   Using class names from labels: {KEYPOINT_NAMES}")
        except Exception as _e:
            print(f"⚠️ Could not read keypoint_names from labels: {_e}")

    if not KEYPOINT_NAMES:
        print("❌ No class names found in label JSON ('keypoint_names' field).")
        raise SystemExit(1)

    builder = YoloxDatasetBuilder(
        labels_dir=LABELS_DIR,
        video_dir=VIDEO_DIR,
        output_dir=OUTPUT_DIR,
        keypoint_names=KEYPOINT_NAMES,
        img_size=640,
        task=TASK,
        box_frac=BOX_FRAC,
    )
    print(f"🧭 Task: {TASK} (YOLOX object detection)")

    stats = builder.build_dataset(train_ratio=0.8)
    if not stats:
        print("❌ Dataset build produced no samples — aborting.")
        raise SystemExit(1)

    print(f"\n📊 Dataset stats:")
    print(f"   Train: {stats['train']} samples")
    print(f"   Val:   {stats['val']} samples")
    print(f"   Total: {stats['total']} samples")

    # Write classes.txt for COCO conversion
    classes_path = Path(OUTPUT_DIR) / "classes.txt"
    classes_path.write_text("\n".join(KEYPOINT_NAMES) + "\n", encoding="utf-8")

    convert_tool = ROOT / "tools" / "convert_yolo_to_coco.py"
    subprocess.check_call([sys.executable, str(convert_tool), OUTPUT_DIR])

    if _args.yolox_dir is None:
        print("\n✅ Dataset ready under:", OUTPUT_DIR)
        print("Next: clone YOLOX and run training/train_yolox.py --yolox-dir <path>")
        raise SystemExit(0)

    train_yolox = ROOT / "training" / "train_yolox.py"
    raise SystemExit(subprocess.call([
        sys.executable, str(train_yolox),
        "--dataset", OUTPUT_DIR,
        "--yolox-dir", str(_args.yolox_dir),
        "--size", MODEL_SIZE,
    ]))
