"""
Utility to split FastCut dataset into train/val/test subsets for YOLO.

The script reads images from `--images` (default: carla_datasetv2/fake_B/images)
and their matching YOLO labels from `--labels` (default: carla_datasetv2/fake_B/labels),
then creates symlinked (or copied) subsets under `--output`.

Example:
    python tools/split_fastcut_dataset.py \
        --images carla_datasetv2/fake_B/images \
        --labels carla_datasetv2/fake_B/labels \
        --output carla_datasetv2/fastcut_split \
        --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 \
        --mode symlink --force
"""
from __future__ import annotations

import argparse
import math
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split FastCut dataset into train/val/test subsets.")
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("carla_datasetv2/fake_B/images"),
        help="Path to source images directory.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("carla_datasetv2/fake_B/labels"),
        help="Path to source labels directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("carla_datasetv2/fastcut_split"),
        help="Output base directory for split subsets.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Portion of data for training.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Portion of data for validation.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Portion of data for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Whether to create symbolic links or copy files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directories if they already exist.",
    )
    return parser.parse_args()


def _validate_ratios(train: float, val: float, test: float) -> Tuple[float, float, float]:
    total = train + val + test
    if total <= 0:
        raise ValueError("Sum of ratios must be positive.")
    return train / total, val / total, test / total


def _gather_images(images_dir: Path, labels_dir: Path, skip_empty: bool = True) -> List[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    files = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise RuntimeError(f"No image files found in {images_dir}")
    
    # 빈 레이블 파일 제외
    if skip_empty and labels_dir.exists():
        valid_files = []
        for img_path in files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            # 레이블 파일이 있고 비어있지 않은 경우만 포함
            if label_path.exists() and label_path.stat().st_size > 0:
                valid_files.append(img_path)
        files = valid_files
        print(f"  (빈 레이블 제외: {len([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]) - len(files)}개)")
    
    files.sort()
    return files


def _prepare_subset_dir(base: Path, subset: str, force: bool) -> Tuple[Path, Path]:
    subset_dir = base / subset
    img_dir = subset_dir / "images"
    lbl_dir = subset_dir / "labels"
    if subset_dir.exists():
        if force:
            shutil.rmtree(subset_dir)
        else:
            raise FileExistsError(f"{subset_dir} already exists. Use --force to overwrite.")
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if mode == "symlink":
        # Remove if exists
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        rel_src = os.path.relpath(src, dst.parent)
        os.symlink(rel_src, dst)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()

    train_ratio, val_ratio, test_ratio = _validate_ratios(
        args.train_ratio, args.val_ratio, args.test_ratio
    )
    images = _gather_images(args.images, args.labels, skip_empty=True)

    random.Random(args.seed).shuffle(images)
    total = len(images)
    n_train = math.floor(total * train_ratio)
    n_val = math.floor(total * val_ratio)
    n_test = total - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    print(f"Total images: {total}")
    for name, files in splits.items():
        print(f"  {name}: {len(files)}")

    for subset, file_list in splits.items():
        img_out, lbl_out = _prepare_subset_dir(args.output, subset, args.force)
        for img_path in file_list:
            label_name = img_path.with_suffix(".txt").name
            label_src = args.labels / label_name
            if not label_src.exists():
                raise FileNotFoundError(f"Missing label for {img_path.name}: {label_src}")
            _link_or_copy(img_path, img_out / img_path.name, args.mode)
            _link_or_copy(label_src, lbl_out / label_name, args.mode)

    print(f"✅ Split complete. Output written to {args.output}")


if __name__ == "__main__":
    main()




