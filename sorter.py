# auto_sort_actions_custom_only.py
"""
Takes cropped clips and sorts them into 37 action folders using custom fine-tuned model only
Includes per-clip confidence debugging + CSV logging
Only creates destination folder when actually sorting a clip there
Lowered confidence threshold to 0.1
"""

import os
import cv2
import numpy as np
import json
import shutil
import csv
from pathlib import Path
from openvino.runtime import Core

# =============================
# Paths - adjust to your setup
# =============================
BASE_DIR = Path(__file__).parent.resolve()

ENCODER_XML = BASE_DIR / "models/intel_action/encoder/FP32/action-recognition-0001-encoder.xml"
ENCODER_BIN = BASE_DIR / "models/intel_action/encoder/FP32/action-recognition-0001-encoder.bin"
CUSTOM_DECODER_XML = BASE_DIR / "action_classifier_3d.xml"
CUSTOM_DECODER_BIN = BASE_DIR / "action_classifier_3d.bin"
CUSTOM_MAPPING_PATH = BASE_DIR / "intel_finetuned_classifier_3d_mapping.json"

SEQUENCE_LENGTH = 16


# =============================
# Utilities
# =============================
def load_custom_labels():
    if not CUSTOM_MAPPING_PATH.exists():
        raise FileNotFoundError(f"Label mapping not found: {CUSTOM_MAPPING_PATH}")

    with open(CUSTOM_MAPPING_PATH, "r") as f:
        data = json.load(f)

    idx_to_label = {int(k): v for k, v in data["idx_to_label"].items()}
    label_to_idx = {v: k for k, v in idx_to_label.items()}

    print(f"✓ Loaded {len(idx_to_label)} action labels")
    return idx_to_label, label_to_idx


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def preprocess_frame(frame, input_shape):
    N, C, H, W = input_shape
    h, w = frame.shape[:2]

    scale = min(W / w, H / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    pad_top = (H - new_h) // 2
    pad_bottom = H - new_h - pad_top
    pad_left = (W - new_w) // 2
    pad_right = W - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    padded = padded.transpose(2, 0, 1)
    return np.expand_dims(padded, axis=0).astype(np.float32)


def load_video_clip(video_path):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < SEQUENCE_LENGTH:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, SEQUENCE_LENGTH, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = frames[-1] if frames else np.zeros((224, 224, 3), np.uint8)
        frames.append(frame)

    cap.release()
    return frames


def load_models():
    print("🔧 Loading models...")
    ie = Core()

    encoder_model = ie.read_model(ENCODER_XML, ENCODER_BIN)
    decoder_model = ie.read_model(CUSTOM_DECODER_XML, CUSTOM_DECODER_BIN)

    enc = ie.compile_model(encoder_model, "CPU")
    dec = ie.compile_model(decoder_model, "CPU")

    return (
        enc,
        enc.input(0),
        enc.output(0),
        dec,
        dec.input(0),
        dec.output(0),
    )


def classify_clip(
    video_path,
    enc,
    enc_in,
    enc_out,
    dec,
    dec_in,
    dec_out,
    idx_to_label,
):
    frames = load_video_clip(video_path)
    if len(frames) != SEQUENCE_LENGTH:
        return None, 0.0, None, None

    features = []
    for f in frames:
        inp = preprocess_frame(f, enc_in.shape)
        out = enc([inp])[enc_out]
        features.append(out[0].flatten())

    sequence = np.expand_dims(np.stack(features), axis=0)
    logits = dec([sequence])[dec_out].flatten()
    probs = softmax(logits)

    idx = int(np.argmax(probs))
    return idx_to_label[idx], float(probs[idx]), idx, probs


# =============================
# Main sorter
# =============================
def auto_sort_clips(
    input_folder,
    output_folder,
    confidence_threshold=0.1,  # Lowered to 0.1
    review_threshold=0.05,      # Lowered accordingly
    debug=False,
):
    idx_to_label, _ = load_custom_labels()
    enc, enc_in, enc_out, dec, dec_in, dec_out = load_models()

    sorted_dir = Path(output_folder) / "sorted"
    uncertain_dir = Path(output_folder) / "uncertain"
    review_dir = Path(output_folder) / "review"

    # Only create the base output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Create uncertain and review dirs (always needed)
    uncertain_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    
    # Track which sorted folders we've created
    created_sorted_folders = set()

    csv_path = Path(output_folder) / "classification_debug.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["filename", "action", "confidence", "decision", "top5"])

    videos = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        videos.extend(Path(input_folder).glob(ext))

    print(f"\n🔍 Sorting {len(videos)} clips\n")

    for i, v in enumerate(videos, 1):
        action, conf, _, probs = classify_clip(
            v, enc, enc_in, enc_out, dec, dec_in, dec_out, idx_to_label
        )

        if action is None:
            continue

        print(
            f"\r[{i}/{len(videos)}] {v.name} → {action} ({conf:.3f})",
            end="",
        )

        if conf >= confidence_threshold:
            # Create sorted action folder only when needed
            action_dir = sorted_dir / action
            if action not in created_sorted_folders:
                action_dir.mkdir(parents=True, exist_ok=True)
                created_sorted_folders.add(action)
            dest = action_dir / v.name
            decision = "sorted"
        elif conf >= review_threshold:
            dest = uncertain_dir / f"{action}_{conf:.2f}_{v.name}"
            decision = "uncertain"
        else:
            dest = review_dir / f"{action}_{conf:.2f}_{v.name}"
            decision = "review"

        top5 = np.argsort(probs)[-5:][::-1]
        top5_str = "; ".join(
            f"{idx_to_label[i]}:{probs[i]:.3f}" for i in top5
        )

        csv_writer.writerow(
            [v.name, action, f"{conf:.6f}", decision, top5_str]
        )

        if debug:
            print(f"\n   🔎 Top-5 → {top5_str}")

        shutil.copy2(v, dest)

    csv_file.close()
    print("\n\n✅ Done")
    print(f"📝 Confidence log: {csv_path}")


# =============================
# CLI
# =============================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="sorted_dataset")
    p.add_argument("--threshold", type=float, default=0.1)  # Lowered default
    p.add_argument("--review-threshold", type=float, default=0.05)  # Lowered default
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    auto_sort_clips(
        args.input,
        args.output,
        args.threshold,
        args.review_threshold,
        args.debug,
    )