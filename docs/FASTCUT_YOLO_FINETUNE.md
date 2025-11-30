# FastCut → YOLOv8 파인튜닝 가이드

FastCut으로 실제 도로 스타일로 변환된 `carla_datasetv2/fake_B` 이미지를 그대로
Ultralytics YOLOv8에 파인튜닝하여 간단하게 도메인 적응을 수행하는 방법입니다.

## 1. 데이터 준비 확인

```
carla_datasetv2/
├── fake_B/
│   ├── images/   # FastCut 변환 이미지 (1,790장)
│   └── labels/   # real_A와 동일한 YOLO 라벨 (심볼릭 링크 가능)
└── dataset.yaml
```

아래 명령으로 이미지/라벨 개수가 동일한지 확인합니다.

```bash
ls carla_datasetv2/fake_B/images | wc -l
ls carla_datasetv2/fake_B/labels | wc -l
```

둘 다 1,790이 나오면 정상입니다. (`labels`는 `../labels`를 향하는 심볼릭 링크여도 됩니다.)

## 2. 데이터셋 YAML (carla_fastcut.yaml)

레포지토리 루트에 `carla_fastcut.yaml`이 있습니다.
기본적으로 train/val/test를 모두 `fake_B/images` 디렉토리로 지정했습니다.
별도 검증 세트를 쓰고 싶다면 `val`, `test` 항목을 원하는 폴더로 수정하세요.

```yaml
path: carla_datasetv2
train: fake_B
val: fake_B
test: fake_B
names:
  0: vehicle
  1: pedestrian
```

## 3. 파인튜닝 실행

### Python 스크립트 사용

```bash
python tools/train_yolov8_fastcut.py \
    --data carla_fastcut.yaml \
    --weights yolov8n.pt \
    --epochs 30 \
    --batch 16 \
    --imgsz 320 \
    --device mps
```

- `--device`는 `mps`, `cuda`, `cpu` 중 하나입니다. MPS나 CUDA를 사용할 수 없는 경우 자동으로 CPU로 폴백합니다.
- 결과는 `runs/fastcut/yolov8_fastcut/weights/best.pt` 등에 저장됩니다.

### Ultralytics CLI 직접 사용

동일한 설정을 CLI로도 실행할 수 있습니다.

```bash
yolo detect train \
    data=carla_fastcut.yaml \
    model=yolov8n.pt \
    epochs=30 \
    imgsz=320 \
    batch=16 \
    device=mps \
    project=runs/fastcut \
    name=yolov8_cli
```

## 4. 결과 활용

- 최종 가중치: `runs/fastcut/<run_name>/weights/best.pt`
- 평가: `yolo detect val data=carla_fastcut.yaml model=runs/fastcut/<run_name>/weights/best.pt`
- 추론: `yolo detect predict source=path/to/images model=...`

필요하면 Stage 1/2 GRU 파이프라인 대신 이 가중치를 사용하여
실제 도로 스타일에 맞춘 기본 YOLO 모델을 바로 얻을 수 있습니다.

