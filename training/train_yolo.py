"""
YOLO Keypoint Detection Training
Trains YOLOv8-pose on your labeled keypoint dataset
"""

import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False

class YOLOKeypointDatasetBuilder:
    """
    Convert JSON labels to YOLO keypoint format
    """
    
    def __init__(self, 
                 labels_dir,      # Directory with JSON label files
                 video_dir,       # Directory with video files
                 output_dir,      # Where to save YOLO format dataset
                 keypoint_names,  # List of keypoint names (e.g., ['kp1', 'kp2', 'kp3', 'kp4'])
                 img_size=640):
        
        self.labels_dir = Path(labels_dir)
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.keypoint_names = keypoint_names
        self.num_keypoints = len(keypoint_names)
        self.img_size = img_size
        
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
        """Locate a video file in video_dir by name (with or without extension)."""
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            for cand in (self.video_dir / video_name,
                         self.video_dir / f"{Path(video_name).stem}{ext}"):
                if cand.exists():
                    return cand
        return None
    
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
        
        # Add padding (20%)
        pad_x = (x_max - x_min) * 0.2
        pad_y = (y_max - y_min) * 0.2
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

        label_files = list(self.labels_dir.glob("*.json"))
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
                yolo_label = self.convert_to_yolo_format(fi, w, h)
                if yolo_label is None:
                    continue

                stem = f"{video_path.stem}_frame_{idx:06d}"
                cv2.imwrite(str(stage_img / f"{stem}.jpg"), frame)
                with open(stage_lbl / f"{stem}.txt", 'w') as f:
                    f.write(yolo_label + '\n')
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
        """Create dataset.yaml for YOLO training"""
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


class YOLOKeypointTrainer:
    """
    Train YOLOv8-pose on keypoint dataset
    """
    
    def __init__(self, dataset_yaml, model_size='n', device='cpu'):
        """
        Args:
            dataset_yaml: Path to dataset.yaml
            model_size: 'n', 's', 'm', 'l', 'x' (YOLO model sizes)
            device: 'cpu' or 'cuda'
        """
        self.dataset_yaml = dataset_yaml
        self.model_size = model_size
        self.device = device
        
        # Load pretrained model
        model_name = f"yolo11{model_size}-pose.pt"
        print(f"📂 Loading pretrained: {model_name}")
        self.model = YOLO(model_name)
    
    def train(self, 
              epochs=100,
              imgsz=640,
              batch_size=16,
              lr0=0.001,
              augment=True,
              patience=20,
              project='yolo_keypoint_training',
              name='keypoint_detector'):
        """
        Train YOLO keypoint detector
        
        Args:
            epochs: Number of training epochs
            imgsz: Image size
            batch_size: Batch size
            lr0: Initial learning rate
            augment: Use data augmentation
            patience: Early stopping patience
            project: Project name for outputs
            name: Run name
        """
        print(f"🚀 Starting YOLO keypoint training...")
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {imgsz}")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {self.device}")
        
        # Train the model
        results = self.model.train(
            data=str(self.dataset_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            lr0=lr0,
            augment=augment,
            patience=patience,
            device=self.device,
            project=project,
            name=name,
            exist_ok=True,
            workers=4,
            verbose=True,
            # Loss gains (kpt_shape / flip_idx come from dataset.yaml, not here)
            box=7.5,   # Box loss gain
            cls=0.5,   # Class loss gain
            dfl=1.5,   # DFL loss gain
            pose=12.0, # Pose loss gain
            kobj=1.0,  # Keypoint objectness loss gain
        )
        
        print(f"✅ Training complete!")
        return results
    
    def export(self, format='onnx', imgsz=640):
        """
        Export trained model to different formats
        
        Args:
            format: 'onnx', 'torchscript', 'tflite', etc.
            imgsz: Image size for export
        """
        print(f"📦 Exporting model to {format}...")
        
        # Find best model
        best_model = Path(f"yolo_keypoint_training/keypoint_detector/weights/best.pt")
        if not best_model.exists():
            print(f"⚠️ Best model not found at: {best_model}")
            return None
        
        model = YOLO(str(best_model))
        exported = model.export(format=format, imgsz=imgsz)
        
        print(f"✅ Model exported to: {exported}")
        return exported


def test_trained_model(model_path, test_video_path, keypoint_names=None):
    """
    Test trained model on a video
    """
    if keypoint_names is None:
        keypoint_names = ['kp1', 'kp2', 'kp3', 'kp4']
    
    print(f"🧪 Testing model on: {test_video_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Process video
    cap = cv2.VideoCapture(test_video_path)
    if not cap.isOpened():
        print(f"❌ Could not open: {test_video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer for output
    out_path = Path(test_video_path).stem + "_keypoints.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    print(f"📹 Processing video, writing to: {out_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame, conf=0.25, verbose=False)
        
        # Draw results
        for r in results:
            if r.keypoints is not None:
                keypoints = r.keypoints.data.cpu().numpy()
                
                for kp_set in keypoints:
                    for i, (x, y, conf) in enumerate(kp_set):
                        if conf > 0.3:
                            # Draw keypoint
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                            cv2.putText(frame, f"{keypoint_names[i]}", 
                                      (int(x)+10, int(y)-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"✅ Output saved: {out_path}")


# =============================
# MAIN USAGE
# =============================

if __name__ == "__main__":
    # ============================================
    # CONFIG — edit these for your setup
    # ============================================
    KEYPOINT_NAMES = []  # filled from the labeler's exported JSON (see below)
    LABELS_DIR = "labels"       # folder with the labeler's *_labels.json exports
    VIDEO_DIR  = "."            # folder containing the source videos
    OUTPUT_DIR = "yolo_dataset" # where the YOLO dataset gets written
    MODEL_SIZE = 'n'            # 'n' fast, 's'/'m' more accurate
    EPOCHS     = 100
    DEVICE     = 0 if _HAS_CUDA else 'cpu'

    # ============================================
    # STEP 1: Build dataset from JSON labels
    # ============================================
    if not Path(LABELS_DIR).exists():
        print(f"❌ Labels folder '{LABELS_DIR}' not found.\n"
              f"   Export labels from labeler.py into that folder, then re-run.")
        raise SystemExit(1)

    # Use the keypoint names recorded by the labeler (single source of truth)
    _label_files = sorted(Path(LABELS_DIR).glob("*.json"))
    if _label_files:
        try:
            with open(_label_files[0], 'r', encoding='utf-8') as _f:
                _names = json.load(_f).get('keypoint_names')
            if _names:
                KEYPOINT_NAMES = _names
                print(f"   Using keypoint names from labels: {KEYPOINT_NAMES}")
        except Exception as _e:
            print(f"⚠️ Could not read keypoint_names from labels: {_e}")

    if not KEYPOINT_NAMES:
        print("❌ No keypoint names found. Export labels from labeler.py "
              "(the JSON carries 'keypoint_names'), then re-run.")
        raise SystemExit(1)

    builder = YOLOKeypointDatasetBuilder(
        labels_dir=LABELS_DIR,
        video_dir=VIDEO_DIR,
        output_dir=OUTPUT_DIR,
        keypoint_names=KEYPOINT_NAMES,
        img_size=640
    )

    stats = builder.build_dataset(train_ratio=0.8)
    if not stats:
        print("❌ Dataset build produced no samples — aborting.")
        raise SystemExit(1)

    print(f"\n📊 Dataset stats:")
    print(f"   Train: {stats['train']} samples")
    print(f"   Val:   {stats['val']} samples")
    print(f"   Total: {stats['total']} samples")

    # ============================================
    # STEP 2: Train YOLO model
    # ============================================
    trainer = YOLOKeypointTrainer(
        dataset_yaml=str(Path(OUTPUT_DIR) / "dataset.yaml"),
        model_size=MODEL_SIZE,
        device=DEVICE
    )

    results = trainer.train(
        epochs=EPOCHS,
        imgsz=640,
        batch_size=16,
        lr0=0.001,
        augment=True,
        patience=20
    )

    # ============================================
    # STEP 3: Export for deployment
    # ============================================
    trainer.export(format='onnx')

    # ============================================
    # STEP 4: Test on a video (optional)
    # ============================================
    best = Path("yolo_keypoint_training/keypoint_detector/weights/best.pt")
    test_video = next(iter(Path(VIDEO_DIR).glob("*.mp4")), None)
    if best.exists() and test_video is not None:
        test_trained_model(
            model_path=str(best),
            test_video_path=str(test_video),
            keypoint_names=KEYPOINT_NAMES
        )

    print("\n✅ YOLO keypoint training complete!")