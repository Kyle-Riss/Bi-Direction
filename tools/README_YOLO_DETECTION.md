# YOLO로 바운딩 박스 자동 생성 가이드

test 폴더의 이미지들에 대해 YOLO 모델을 사용하여 자동으로 바운딩 박스를 검출하고 YOLO 형식 라벨 파일을 생성하는 방법입니다.

## 사용 방법

### 1. 필수 패키지 설치

```bash
pip install ultralytics
```

### 2. YOLO로 객체 검출 및 라벨 생성

```bash
python tools/detect_with_yolo.py \
    --model yolov8n.pt \
    --image_dir test \
    --output_dir test/labels \
    --conf 0.25 \
    --iou 0.45
```

**파라미터 설명:**
- `--model`: YOLO 모델 파일 경로 (기본: yolov8n.pt)
- `--image_dir`: 이미지가 있는 디렉토리 (기본: test)
- `--output_dir`: 라벨 파일을 저장할 디렉토리 (기본: test/labels)
- `--conf`: 신뢰도 임계값 (0.0-1.0, 기본: 0.25)
- `--iou`: IoU 임계값 (0.0-1.0, 기본: 0.45)

### 3. 검증

생성된 라벨 파일 검증:

```bash
python tools/validate_labels.py \
    --img_dir test \
    --label_dir test/labels \
    --num_classes 5
```

## 클래스 매핑

YOLOv8 (COCO 데이터셋) 클래스를 우리의 커스텀 클래스로 매핑:

| COCO 클래스 | COCO ID | 커스텀 클래스 | 커스텀 ID |
|------------|---------|--------------|-----------|
| person | 0 | pedestrian | 0 |
| bicycle | 1 | bicycle_motorcycle | 3 |
| car | 2 | car | 1 |
| motorcycle | 3 | bicycle_motorcycle | 3 |
| bus | 5 | truck_bus | 2 |
| truck | 7 | truck_bus | 2 |

**참고:** traffic_sign (class 4)은 YOLOv8 기본 모델에서 검출되지 않을 수 있습니다.

## 문제 해결

### MPS 디바이스 오류

맥북에서 MPS(Metal Performance Shaders) 관련 오류가 발생하면:

1. `tools/detect_with_yolo.py` 파일에서 `device='cpu'`로 설정되어 있는지 확인
2. 또는 환경 변수 설정:
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```

### 메모리 부족

이미지가 많으면 배치 처리로 변경하거나 더 작은 모델 사용:
- `yolov8n.pt` (nano - 가장 작음)
- `yolov8s.pt` (small)
- `yolov8m.pt` (medium)

### 신뢰도 조정

검출 결과가 너무 많거나 적으면 `--conf` 값을 조정:
- 값이 높을수록: 더 확실한 검출만 (객체 수 적음)
- 값이 낮을수록: 더 많은 검출 (객체 수 많음, 오검출 가능)

## 예제

### 기본 실행
```bash
python tools/detect_with_yolo.py --image_dir test --output_dir test/labels
```

### 높은 신뢰도로 실행 (오검출 줄이기)
```bash
python tools/detect_with_yolo.py --image_dir test --output_dir test/labels --conf 0.5
```

### 낮은 신뢰도로 실행 (더 많은 검출)
```bash
python tools/detect_with_yolo.py --image_dir test --output_dir test/labels --conf 0.1
```

## 생성된 파일 형식

각 이미지마다 `.txt` 파일이 생성됩니다:

**예시: `test.jpg` → `test.txt`**

```
0 0.456789 0.234567 0.123456 0.345678
1 0.678901 0.456789 0.234567 0.156789
2 0.345678 0.567890 0.189012 0.234567
```

**형식:** `class_id x_center y_center width height` (모두 정규화된 값, 0-1)








