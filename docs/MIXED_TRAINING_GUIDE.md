# 혼합 학습 가이드 (모델 담당)

CARLA 시뮬레이션 데이터와 리얼 데이터를 혼합하여 학습하는 방법

## 📋 목차

1. [데이터 준비](#1-데이터-준비)
2. [학습 전략](#2-학습-전략)
3. [사용 방법](#3-사용-방법)
4. [주의사항](#4-주의사항)

---

## 1. 데이터 준비

### 1.1 리얼 데이터 준비

```bash
# test 폴더의 리얼 데이터를 datasets/real_data로 정리
python tools/prepare_real_data.py \
    --test_dir test \
    --output_dir datasets/real_data \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

이 스크립트는:
- metadata.tsv를 파싱하여 라벨 정보 추출
- 이미지를 train/val/test로 분할
- 라벨 정보를 label_info.txt에 저장

**⚠️ 중요**: 현재 `vectors.tsv`는 임베딩 벡터로 보이며 바운딩 박스 정보가 없습니다.
- 바운딩 박스 정보는 데이터 담당자에게 요청 필요
- 바운딩 박스 정보를 받으면 YOLO 형식 라벨 파일(.txt) 생성 필요

### 1.2 CARLA 데이터 준비 (시뮬레이션 담당)

시뮬레이션 담당자가 제공할 데이터:
- CARLA 이미지 디렉토리 (예: `datasets/carla_data/train/images/`)
- CARLA 라벨 디렉토리 (예: `datasets/carla_data/train/labels/`)

CARLA 라벨은 이미 YOLO 형식으로 제공될 예정.

### 1.3 라벨 검증

데이터 준비 후 라벨 파일 검증:

```bash
# 리얼 데이터 검증 (라벨 파일이 있을 경우)
python tools/validate_labels.py \
    --img_dir datasets/real_data/train/images \
    --label_dir datasets/real_data/train/labels \
    --num_classes 5

# CARLA 데이터 검증
python tools/validate_labels.py \
    --img_dir datasets/carla_data/train/images \
    --label_dir datasets/carla_data/train/labels \
    --num_classes 5
```

---

## 2. 학습 전략

### 2.1 2단계 학습 전략

1. **Pre-training (CARLA만)**
   - CARLA 시뮬레이션 데이터로 모델 초기 학습
   - 목적: 기본 객체 검출 능력 습득
   - 데이터: CARLA train만 사용

2. **Fine-tuning (혼합 학습)**
   - CARLA + 리얼 데이터 혼합 학습
   - 목적: 실제 환경에 적응
   - 데이터: CARLA train + Real train 혼합

### 2.2 혼합 비율 전략

초기에는 CARLA 데이터가 많을 수 있으므로:

- **Stage 1 (초기)**: CARLA 70%, Real 30%
- **Stage 2 (중기)**: CARLA 50%, Real 50%
- **Stage 3 (후기)**: CARLA 30%, Real 70%

---

## 3. 사용 방법

### 3.1 CARLA 단독 학습 (Pre-training)

```python
from datasets.dataset import create_dataloader

# CARLA 데이터만 사용
train_loader = create_dataloader(
    img_dir='datasets/carla_data/train/images',
    label_dir='datasets/carla_data/train/labels',
    num_frames=3,
    img_size=128,  # coarse stage
    batch_size=8,
    shuffle=True,
    num_workers=2,
    normalize=True
)

val_loader = create_dataloader(
    img_dir='datasets/carla_data/val/images',
    label_dir='datasets/carla_data/val/labels',
    num_frames=3,
    img_size=128,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    normalize=True
)

# 학습 진행...
```

### 3.2 혼합 학습 (Fine-tuning)

```python
from datasets.mixed_dataset import create_mixed_training_dataloaders

# CARLA + 리얼 데이터 혼합
train_loader, val_loader = create_mixed_training_dataloaders(
    carla_train_dir='datasets/carla_data/train',
    carla_val_dir='datasets/carla_data/val',
    real_train_dir='datasets/real_data/train',
    real_val_dir='datasets/real_data/val',
    num_frames=3,
    img_size=320,  # fine stage
    batch_size=2,
    carla_weight=0.5,  # CARLA 50%
    real_weight=0.5,   # Real 50%
    mode='concat',     # 또는 'weighted'
    num_workers=2,
    normalize=True
)

# 학습 진행...
```

### 3.3 가중치 샘플링 모드

```python
# 가중치 샘플링 모드 사용 (작은 데이터셋 확장)
train_loader, val_loader = create_mixed_training_dataloaders(
    carla_train_dir='datasets/carla_data/train',
    real_train_dir='datasets/real_data/train',
    carla_weight=0.7,  # CARLA 70%
    real_weight=0.3,   # Real 30%
    mode='weighted',   # 가중치 샘플링
    ...
)
```

---

## 4. 주의사항

### 4.1 라벨 포맷 통일

- **클래스 ID**: CARLA와 리얼 데이터 모두 동일한 클래스 정의 사용
  - 0: pedestrian
  - 1: car
  - 2: truck_bus
  - 3: bicycle_motorcycle
  - 4: traffic_sign

- **좌표 형식**: 모두 YOLO 형식 (정규화된 중심 좌표)
  ```
  class_id x_center y_center width height
  ```

### 4.2 데이터 불균형

- CARLA 데이터가 많을 수 있음
- 리얼 데이터가 적으면 가중치 조정 또는 증강 필요

### 4.3 도메인 적응

- 시뮬레이션과 실제 환경의 차이 존재
- 필요시 Adversarial Training 또는 Domain Adaptation 기법 적용 고려

### 4.4 평가

- 검증 세트는 CARLA와 리얼을 분리하여 평가 권장
- 각 도메인별 성능 추적

---

## 5. 체크리스트

### 데이터 준비
- [ ] 리얼 데이터를 datasets/real_data로 정리 완료
- [ ] CARLA 데이터 수신 (시뮬레이션 담당)
- [ ] 라벨 파일 검증 완료
- [ ] 클래스 ID 매핑 확인 완료

### 학습 준비
- [ ] CARLA 단독 학습 스크립트 준비
- [ ] 혼합 학습 스크립트 준비
- [ ] 학습 파이프라인 테스트 완료

### 학습 진행
- [ ] Stage 1: CARLA 단독 학습 완료
- [ ] Stage 2: 혼합 학습 진행
- [ ] 검증 세트 평가 완료

---

## 6. 문의 사항

- 데이터 준비 관련: 데이터 담당자와 협의
- CARLA 데이터 관련: 시뮬레이션 담당자와 협의
- 모델 학습 관련: 모델 담당자 (현재)









