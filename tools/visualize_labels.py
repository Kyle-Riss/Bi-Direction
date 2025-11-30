import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO label annotations.")
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Dataset roots that contain images/ and labels/ subdirectories (or flat image/label pairs).",
    )
    parser.add_argument("--num-samples", type=int, default=5, help="Number of images per dataset to render.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/label_previews"), help="Directory to save rendered images.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducible sampling.")
    return parser.parse_args()


def find_pairs(root: Path):
    img_dir = root / "images" if (root / "images").exists() else root
    label_dir = root / "labels" if (root / "labels").exists() else root

    pairs = []
    for img_path in sorted(img_dir.glob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = label_dir / (img_path.stem + ".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
    return pairs


def load_labels(label_path: Path):
    entries = []
    text = label_path.read_text().strip()
    if not text:
        return entries
    for line in text.splitlines():
                    parts = line.strip().split()
        if len(parts) != 5:
                continue
        cls, cx, cy, w, h = map(float, parts)
        entries.append((int(cls), cx, cy, w, h))
    return entries


def render_sample(image_path: Path, label_path: Path, out_path: Path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    labels = load_labels(label_path)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    for cls, cx, cy, bw, bh in labels:
        box_w = bw * w
        box_h = bh * h
        x0 = (cx - bw / 2) * w
        y0 = (cy - bh / 2) * h
        rect = plt.Rectangle((x0, y0), box_w, box_h, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(x0, y0, f"{cls}", color="yellow", fontsize=10, bbox=dict(facecolor="black", alpha=0.4, pad=1))
    ax.axis("off")
    ax.set_title(f"{image_path.parent.parent.name}/{image_path.stem}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    args = parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for root_str in args.roots:
        root = Path(root_str)
        pairs = find_pairs(root)
        if not pairs:
            print(f"[WARN] No image/label pairs found in {root}")
            continue
    
        sample_count = min(args.num_samples, len(pairs))
        chosen = random.sample(pairs, sample_count)
        dataset_name = root.name

        for img_path, label_path in chosen:
            safe_name = f"{dataset_name}_{img_path.stem}.png"
            out_path = args.output_dir / safe_name.replace(" ", "_")
            print(f"[INFO] Rendering {img_path} -> {out_path}")
            render_sample(img_path, label_path, out_path)


if __name__ == "__main__":
    main()





