"""
Generate YOLO-format labels automatically using a pretrained YOLO model.

Typical usage:
    python tools/auto_label_yolo.py \
        --model yolov8m.pt \
        --roots carla_datasetv2/real_A carla_datasetv2/real_B carla_datasetv2/fake_B \
        --device mps --conf 0.35 --imgsz 640 \
        --class-map 2=0 5=0 7=0 0=1

By default, label files are written to a sibling `labels/` directory that mirrors
each images folder. Existing files are only overwritten when `--overwrite` is set.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label images with a YOLO model.")
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        type=Path,
        help="Directories that contain images (processed recursively).",
    )
    parser.add_argument("--model", type=str, required=True, help="Path or name of YOLO weights.")
    parser.add_argument("--device", type=str, default="cpu", help="Device string (cpu, mps, 0, etc.).")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for inference.")
    parser.add_argument(
        "--class-map",
        nargs="*",
        default=None,
        help="Optional mapping from source class id to new id, e.g., 2=0 5=0 (others dropped).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing label files instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference but do not write label files (useful for testing).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit on number of images to process (for debugging).",
    )
    return parser.parse_args()


def discover_images(roots: Sequence[Path]) -> List[Path]:
    images: List[Path] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Root directory not found: {root}")
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                images.append(path)
    images.sort()
    return images


def chunk(seq: Sequence[Path], size: int) -> Iterator[List[Path]]:
    for start in range(0, len(seq), size):
        yield list(seq[start : start + size])


def infer_label_path(image_path: Path) -> Path:
    parent = image_path.parent
    if parent.name.lower() == "images":
        base = parent.parent
    else:
        base = parent
    label_dir = base / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    return label_dir / f"{image_path.stem}.txt"


def parse_class_map(pairs: Iterable[str] | None) -> Dict[int, int] | None:
    if not pairs:
        return None
    mapping: Dict[int, int] = {}
    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"Invalid class mapping '{raw}'. Expected format 'src=dst'.")
        left, right = raw.split("=", 1)
        src = int(left.strip())
        dst = int(right.strip())
        mapping[src] = dst
    return mapping


def main() -> None:
    args = parse_args()
    images = discover_images(args.roots)
    if not images:
        raise RuntimeError("No images found under provided roots.")
    if args.max_images:
        images = images[: args.max_images]

    print(f"Found {len(images)} images. Running inference with {args.model}...")
    model = YOLO(args.model)
    class_map = parse_class_map(args.class_map)

    total_written = 0
    total_skipped = 0

    for batch_paths in chunk(images, args.batch):
        results = model.predict(
            source=batch_paths,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )

        for result in results:
            img_path = Path(result.path)
            label_path = infer_label_path(img_path)
            if label_path.exists() and not args.overwrite:
                total_skipped += 1
                continue

            boxes = result.boxes
            if boxes is None or boxes.shape[0] == 0:
                label_path.write_text("")
                total_written += 1
                continue

            lines: List[str] = []
            for idx in range(boxes.shape[0]):
                cls_id = int(boxes.cls[idx].item())
                if class_map is not None:
                    if cls_id not in class_map:
                        continue
                    target_cls = class_map[cls_id]
                else:
                    target_cls = cls_id

                xywhn = boxes.xywhn[idx].tolist()
                conf = boxes.conf[idx].item() if boxes.conf is not None else 1.0
                line = f"{target_cls} " + " ".join(f"{v:.6f}" for v in xywhn) + f" {conf:.4f}"
                lines.append(line)

            if args.dry_run:
                continue

            label_path.write_text("\n".join(lines))
            total_written += 1

    print(f"✅ Auto-label complete. Written: {total_written}, skipped: {total_skipped}")


if __name__ == "__main__":
    main()




