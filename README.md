# Bi-Direction: 영상 데이터 객체 감지 프로젝트

## 🎯 최종 목표
**영상 데이터에서 객체를 정확하게 감지하는 모델 만들기**

## 📋 프로젝트 개요

### 목적
- 영상 데이터에서 객체 검출 성능 향상
- CARLA 시뮬레이션 데이터와 실제 도로 데이터를 혼합한 Mixed 데이터셋 사용
- YOLOv8 기반 객체 탐지 모델 학습

### 주요 특징
- **Mixed 데이터셋**: realB (실제 도로) + fakeB (CARLA 시뮬레이션) 혼합
- **4개 클래스 탐지**: vehicle, pedestrian, traffic_sign, traffic_light
- **자동 라벨링**: 표지판/신호등 자동 라벨링 및 필터링
- **비디오 추론 최적화**: 종횡비 크롭, confidence/NMS 튜닝

## 📁 프로젝트 구조

```
Bi-Direction/
├── config.py                 # 중앙 설정 관리
├── main.py                   # 메인 학습 스크립트
├── requirements.txt          # Python 패키지 의존성
├── carla_mixed.yaml          # Mixed 데이터셋 설정 (4개 클래스)
├── models/                   # 모델 코드
├── datasets/                 # 데이터셋 코드
├── tools/                    # 도구 스크립트
│   ├── train_mixed_with_traffic.py  # 표지판/신호등 포함 학습
│   ├── auto_label_traffic_signs_lights.py  # 자동 라벨링
│   ├── filter_traffic_labels.py     # 라벨 필터링
│   └── infer_video_with_crop.py     # 비디오 추론 (크롭)
├── carla_datasetv2/          # CARLA 데이터셋
│   ├── realB_split/         # 실제 도로 데이터 (train/val/test)
│   └── fakeB_split/         # CARLA 시뮬레이션 데이터 (train/val/test)
└── runs/                     # 학습 결과 저장
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Python 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터셋 준비

#### Mixed 데이터셋 (realB + fakeB)

현재 프로젝트는 **realB (실제 도로 데이터)**와 **fakeB (CARLA 시뮬레이션 데이터)**를 혼합한 데이터셋을 사용합니다.

**데이터셋 구조:**
```
carla_datasetv2/
├── realB_split/
│   ├── train/
│   │   ├── images/          # 1,223개 이미지
│   │   └── labels/          # 필터링된 라벨 (4개 클래스)
│   ├── val/
│   └── test/
└── fakeB_split/
    ├── train/
    │   ├── images/          # 894개 이미지
    │   └── labels/          # 필터링된 라벨 (4개 클래스)
    ├── val/
    └── test/
```

**클래스 정의:**
- `0: vehicle` - 차량
- `1: pedestrian` - 보행자
- `2: traffic_sign` - 표지판
- `3: traffic_light` - 신호등

#### 압축된 데이터셋 사용

```bash
# 학습 데이터셋 압축 해제
tar -xzf training_dataset_final.tar.gz
```

### 3. 모델 학습

#### 기본 Mixed 모델 학습 (2개 클래스: vehicle, pedestrian)

```bash
python tools/train_yolov8_fastcut.py \
    --data carla_mixed.yaml \
    --weights yolov8n.pt \
    --epochs 15 \
    --batch 4 \
    --imgsz 192 \
    --device mps \
    --name mixed_full_e15
```

#### 표지판/신호등 포함 학습 (4개 클래스)

**⚠️ 중요: 라벨링된 데이터셋으로 재학습이 필요합니다!**

현재 데이터셋에는 표지판/신호등 라벨이 자동으로 생성되어 있습니다. 이 라벨들은 휴리스틱 방법으로 생성되었으므로, 재학습을 통해 모델 성능을 향상시킬 수 있습니다.

```bash
python tools/train_mixed_with_traffic.py \
    --data carla_mixed.yaml \
    --weights runs/fastcut/mixed_full_e15/weights/best.pt \
    --epochs 20 \
    --batch 4 \
    --imgsz 192 \
    --lr 5e-5 \
    --device mps \
    --name mixed_with_traffic_e20
```

**파라미터 설명:**
- `--data`: 데이터셋 YAML 파일 (carla_mixed.yaml)
- `--weights`: 사전 학습된 모델 (2개 클래스 모델)
- `--epochs`: 학습 epoch 수
- `--batch`: 배치 크기
- `--imgsz`: 이미지 크기 (192x192)
- `--lr`: 학습률 (추가 학습이므로 낮게 설정: 5e-5)
- `--device`: 학습 디바이스 (mps/cuda/cpu)
- `--name`: 실험 이름

### 4. 비디오 추론

#### 종횡비 크롭 적용 추론 (권장)

```bash
python tools/infer_video_with_crop.py \
    --model runs/fastcut/mixed_with_traffic_e20/weights/best.pt \
    --video archive/bdd100k/videos/train/00c12bd0-bb46e479.mov \
    --output runs/video_comparison/mixed_cropped.mp4 \
    --conf 0.20 \
    --crop_mode bottom
```

**크롭 모드:**
- `bottom`: 하단 중심 크롭 (도로 영역 포함, 권장)
- `center`: 중앙 크롭

## 📊 데이터 현황

### 학습 데이터
- **realB_split/train**: 1,223개 이미지
- **fakeB_split/train**: 894개 이미지
- **총 학습 이미지**: 2,117개

### 라벨 통계 (필터링 후)
- **vehicle**: 3,372개
- **pedestrian**: 231개
- **traffic_sign**: 1,535개
- **traffic_light**: 6,457개
- **총 라벨**: 11,595개

## 🔧 주요 도구

### 1. 자동 라벨링

표지판/신호등 자동 라벨링 (휴리스틱 방법):

```bash
python tools/auto_label_traffic_signs_lights.py \
    --image_dir carla_datasetv2/realB_split/train/images \
    --label_dir carla_datasetv2/realB_split/train/labels \
    --output_dir carla_datasetv2/realB_split/train/labels_with_traffic \
    --method heuristic \
    --conf 0.3
```

### 2. 라벨 필터링

자동 라벨링 결과 필터링 (false positive 제거):

```bash
python tools/filter_traffic_labels.py \
    --input_dir carla_datasetv2/realB_split/train/labels_with_traffic \
    --output_dir carla_datasetv2/realB_split/train/labels_filtered \
    --min_conf 0.4 \
    --min_size 0.005 \
    --max_size 0.25 \
    --max_signs 8 \
    --max_lights 12 \
    --iou_threshold 0.6
```

### 3. 모델 비교

두 모델을 비디오에서 비교:

```bash
python tools/compare_models_on_video.py \
    --video archive/bdd100k/videos/train/00c12bd0-bb46e479.mov \
    --model1 runs/fastcut/baseline_realA_full_e15/weights/best.pt \
    --model2 runs/fastcut/mixed_full_e15/weights/best.pt \
    --output_dir runs/video_comparison \
    --conf 0.25
```

## 📝 학습 전략

### 1단계: 기본 모델 학습 (2개 클래스)
- realB + fakeB 데이터로 vehicle, pedestrian 학습
- 이미지 크기: 192x192
- Epochs: 15

### 2단계: 표지판/신호등 추가 학습 (4개 클래스)
- **⚠️ 라벨링된 데이터셋으로 재학습 필요**
- 기존 모델에 traffic_sign, traffic_light 클래스 추가
- 낮은 학습률로 추가 학습 (5e-5)
- Epochs: 20

### 3단계: 비디오 추론 최적화
- 종횡비 크롭 적용 (720x1280 → 720x720)
- Confidence/NMS 튜닝
- Temporal tracking (향후 구현)

## ⚠️ 중요 사항

### 라벨링된 데이터셋 재학습 필요

현재 데이터셋의 표지판/신호등 라벨은 **자동 라벨링(휴리스틱 방법)**으로 생성되었습니다. 이는 초기 라벨링이며, 다음과 같은 이유로 재학습이 필요합니다:

1. **False Positive**: 휴리스틱 방법은 색상 기반이므로 노이즈가 많을 수 있음
2. **정확도 향상**: 재학습을 통해 모델이 더 정확한 패턴을 학습
3. **성능 개선**: 실제 비디오에서의 탐지 성능 향상

**재학습 방법:**
```bash
# 1. 필터링된 라벨로 학습
python tools/train_mixed_with_traffic.py \
    --data carla_mixed.yaml \
    --weights runs/fastcut/mixed_full_e15/weights/best.pt \
    --epochs 20 \
    --name mixed_with_traffic_e20

# 2. 학습 결과 확인
# runs/fastcut/mixed_with_traffic_e20/results.csv 확인

# 3. 비디오에서 테스트
python tools/infer_video_with_crop.py \
    --model runs/fastcut/mixed_with_traffic_e20/weights/best.pt \
    --video <비디오_경로> \
    --output <출력_경로>
```

## 🎬 비디오 추론 최적화

### 종횡비 문제 해결

비디오 프레임(720x1280)과 학습 이미지(1280x1280)의 종횡비 차이로 인한 성능 저하를 해결하기 위해 크롭 방식을 사용합니다:

```
비디오 프레임 (720×1280)
    ↓
하단 중심 720×720 크롭 (도로 영역 포함)
    ↓
192×192 리사이즈
    ↓
추론
```

이 방식으로 **평균 탐지 수가 0.36 → 0.76으로 약 2배 향상**되었습니다.

## 📦 데이터셋 압축

학습에 필요한 데이터셋은 `training_dataset_final.tar.gz`에 포함되어 있습니다:

```bash
# 압축 해제
tar -xzf training_dataset_final.tar.gz

# 포함 내용:
# - realB_split/train (이미지 + 필터링된 라벨)
# - fakeB_split/train (이미지 + 필터링된 라벨)
# - realB_split/val
# - fakeB_split/val
# - carla_mixed.yaml (4개 클래스 설정)
```

## 🔍 문제 해결

### MPS 디바이스 오류 (macOS)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### 메모리 부족

배치 크기를 줄이거나 더 작은 모델 사용:
- `--batch 2` 또는 `--batch 1`
- `yolov8n.pt` (nano - 가장 작음)

## 📚 참고 문서

- `docs/VIDEO_INFERENCE_ISSUES.md`: 비디오 추론 성능 문제 분석
- `docs/MIXED_MODEL_IMPROVEMENT_PLAN.md`: 모델 성능 개선 계획

## 📄 라이선스

[라이선스 정보]
