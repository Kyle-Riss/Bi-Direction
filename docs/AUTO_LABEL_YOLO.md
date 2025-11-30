# Auto Labeling with YOLO

Use the helper script `tools/auto_label_yolo.py` to generate YOLO-format labels from pretrained detectors.

## Prerequisites
- Ultralytics package installed (`pip install ultralytics`)
- Pretrained YOLO weights (e.g., `yolov8m.pt`) accessible locally

## Basic Command
```bash
python tools/auto_label_yolo.py \
    --model yolov8m.pt \
    --roots carla_datasetv2/real_A carla_datasetv2/real_B carla_datasetv2/fake_B \
    --device mps \
    --conf 0.35 \
    --imgsz 640 \
    --class-map 2=0 5=0 7=0 0=1 \
    --overwrite
```

### Important Flags
- `--roots`: one or more directories containing images; scanning is recursive.
- `--class-map`: optional `src=dst` mapping to collapse pretrained classes into your custom schema. Any class not in the mapping is discarded.
- `--overwrite`: replace existing `.txt` labels. Omit to skip files that already have labels.
- `--max-images`: limit processed images for spot checks.
- `--dry-run`: run inference and report stats without writing files.

### Output Layout
For each image, a normalized YOLO label is written next to the image hierarchy:
- If an image lives under `foo/images/`, labels go to `foo/labels/`.
- Otherwise, labels are placed in `<image_dir>/labels/`.

Each line follows the pattern:
```
<class_id> <cx> <cy> <w> <h> <confidence>
```

## Review & Editing
Auto labels are best treated as a starting point:
- Inspect them in CVAT/labelImg to correct false positives or misses.
- Re-run the script with `--overwrite` after manual fixes only when you really mean to redo them.

Once labels exist, move on to dataset splitting and YAML configuration for YOLO fine-tuning.




