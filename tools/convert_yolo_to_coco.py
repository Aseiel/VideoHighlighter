"""Convert a YOLO-format detection dataset to COCO JSON for YOLOX training.

Expected layout (ultralytics-style):
    dataset/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt     # class cx cy w h  (normalized)
      labels/val/*.txt
      classes.txt              # optional, one class name per line

Usage:
    python tools/convert_yolo_to_coco.py yolo_dataset/ --out yolo_dataset/coco_annotations.json
    python tools/convert_yolo_to_coco.py yolo_dataset/ --split train --out yolo_dataset/coco_train.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_class_names(dataset_dir: Path) -> list[str]:
    classes_file = dataset_dir / "classes.txt"
    if classes_file.is_file():
        names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if names:
            return names
    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.is_file():
        try:
            import yaml
            data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
            names = data.get("names")
            if isinstance(names, dict):
                return [names[k] for k in sorted(names, key=lambda x: int(x) if str(x).isdigit() else x)]
            if isinstance(names, list):
                return [str(n) for n in names]
        except Exception:
            pass
    return ["object"]


def _yolo_line_to_bbox(parts: list[str], img_w: int, img_h: int) -> tuple[int, float, float, float, float] | None:
    if len(parts) < 5:
        return None
    cls_id = int(float(parts[0]))
    cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    x1 = max(0.0, (cx - bw / 2) * img_w)
    y1 = max(0.0, (cy - bh / 2) * img_h)
    w = min(img_w - x1, bw * img_w)
    h = min(img_h - y1, bh * img_h)
    return cls_id, x1, y1, w, h


def convert_split(dataset_dir: Path, split: str, class_names: list[str]) -> dict:
    import cv2

    img_dir = dataset_dir / "images" / split
    lbl_dir = dataset_dir / "labels" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing image dir: {img_dir}")

    images = []
    annotations = []
    ann_id = 1
    for img_id, img_path in enumerate(sorted(img_dir.glob("*")), start=1):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        images.append({
            "id": img_id,
            "file_name": str(Path("images") / split / img_path.name).replace("\\", "/"),
            "width": w,
            "height": h,
        })
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.is_file():
            continue
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            parsed = _yolo_line_to_bbox(parts, w, h)
            if parsed is None:
                continue
            cls_id, x1, y1, bw, bh = parsed
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id + 1,
                "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    return {"images": images, "annotations": annotations, "categories": categories}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert YOLO dataset to COCO JSON")
    parser.add_argument("dataset_dir", type=Path, help="Root of YOLO dataset")
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path (split=both -> <dir>/coco_<split>.json)")
    args = parser.parse_args(argv)

    dataset_dir = args.dataset_dir.resolve()
    class_names = _load_class_names(dataset_dir)
    splits = ["train", "val"] if args.split == "both" else [args.split]

    for split in splits:
        coco = convert_split(dataset_dir, split, class_names)
        if args.out and len(splits) == 1:
            out_path = args.out
        else:
            out_path = dataset_dir / f"coco_{split}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
        print(f"Wrote {out_path} — {len(coco['images'])} images, "
              f"{len(coco['annotations'])} boxes, {len(class_names)} classes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
