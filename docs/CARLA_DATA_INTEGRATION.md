# CARLA 데이터 통합 가이드

FastCut으로 처리된 CARLA 시뮬레이션 데이터를 받아서 학습에 통합하는 방법

## 📋 개요

시뮬레이션 담당자가 FastCut을 사용하여 CARLA 데이터를 처리 중입니다.
처리된 데이터를 받으면 다음과 같은 순서로 통합합니다:

1. **데이터 구조 검증**
2. **데이터 분할 (train/val/test)**
3. **라벨 형식 검증**
4. **리얼 데이터와 통합**

## 📁 예상 데이터 구조

FastCut으로 처리된 CARLA 데이터는 다음과 같은 구조일 것으로 예상됩니다:

```
carla_data/
├── images/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
└── labels/
    ├── frame_0001.txt  (YOLO 형식)
    ├── frame_0002.txt
    └── ...
```

각 라벨 파일은 YOLO 형식:
```
class_id x_center y_center width height
```

## 🚀 사용 방법

### 1. CARLA 데이터 준비

```bash
python tools/prepare_carla_data.py \
    --carla_dir /path/to/carla_data \
    --output_dir datasets/carla_data \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

이 명령은 다음 작업을 수행합니다:
- 데이터 구조 검증
- 라벨 형식 검증
- train/val/test 분할
- `datasets/carla_data/` 구조로 정리

### 2. 통합 상태 확인

```bash
python tools/integrate_carla_real.py \
    --carla_dir datasets/carla_data \
    --real_dir datasets/real_data
```

이 명령은:
- CARLA 데이터 가용성 확인
- 리얼 데이터 가용성 확인
- 사용 가능한 학습 방법 안내

### 3. 도메인 적응 학습 시작

CARLA 데이터와 리얼 데이터가 모두 준비되면:

```bash
python tools/train_domain_adaptation.py \
    --carla_train datasets/carla_data/train \
    --real_train datasets/real_data/train \
    --epochs 20 \
    --batch_size 8 \
    --lambda_adv 0.1 \
    --lambda_align 0.1 \
    --device mps
```

## 📊 데이터 통계

### 리얼 데이터 (현재 상태)
- ✅ 1156개 이미지
- ✅ 라벨 파일 생성 완료
- ✅ 평균 검출: 4.93개 객체/이미지

### CARLA 데이터 (예상)
- 시뮬레이션 담당자로부터 수신 대기
- FastCut으로 처리된 합성 이미지
- 다양한 환경 변수 적용 (광원, 날씨, 배경, 노이즈)

## 🔄 통합 파이프라인

```
CARLA 데이터 (FastCut 처리)
    ↓
prepare_carla_data.py
    ↓
datasets/carla_data/
    ├── train/
    ├── val/
    └── test/
    ↓
통합 (리얼 데이터와)
    ↓
도메인 적응 학습
    ↓
평가 및 검증
```

## 💡 학습 전략

### 전략 1: 단계별 학습 (권장)

1. **CARLA 단독 학습** (Pre-training)
   - 시뮬레이션 데이터로 기본 학습
   - 빠른 수렴 및 초기 성능 확보

2. **도메인 적응 학습** (Adversarial + Alignment)
   - CARLA + 리얼 데이터 동시 사용
   - 도메인 갭 감소

3. **혼합 Fine-tuning**
   - 최적 비율로 혼합 학습
   - 최종 성능 향상

### 전략 2: 직접 혼합 학습

```python
from datasets.mixed_dataset import create_mixed_training_dataloaders

# 비율 조절 가능
train_loader, val_loader = create_mixed_training_dataloaders(
    carla_train_dir='datasets/carla_data/train',
    real_train_dir='datasets/real_data/train',
    carla_weight=0.5,  # 50% : 50%
    real_weight=0.5,
    mode='concat'
)
```

### 전략 3: 비율 조절 실험

```python
# 시뮬레이션 중심 (초기)
carla_weight=0.7, real_weight=0.3

# 균형 (중간)
carla_weight=0.5, real_weight=0.5

# 리얼 중심 (후기)
carla_weight=0.3, real_weight=0.7
```

## 🔍 데이터 검증

### 자동 검증

`prepare_carla_data.py`가 자동으로 수행:
- 이미지-라벨 매칭 확인
- 라벨 형식 검증 (YOLO 형식)
- 클래스 ID 범위 확인
- 좌표 범위 확인 [0, 1]

### 수동 확인

```python
from tools.prepare_carla_data import validate_carla_structure, check_label_format

# 구조 검증
is_valid = validate_carla_structure('path/to/carla_data')

# 라벨 검증
label_dir = 'path/to/carla_data/labels'
is_valid_labels = check_label_format(label_dir, num_classes=5)
```

## 📝 체크리스트

CARLA 데이터 수신 시:

- [ ] 데이터 구조 확인 (`images/`, `labels/` 디렉토리)
- [ ] `prepare_carla_data.py` 실행
- [ ] 라벨 형식 검증 통과 확인
- [ ] 통합 상태 확인 (`integrate_carla_real.py`)
- [ ] 도메인 적응 학습 시작
- [ ] 성능 평가 (mAP, F1)

## ⚠️ 주의사항

1. **라벨 형식**: YOLO 형식이어야 함
   ```
   class_id x_center y_center width height
   ```
   모든 값이 정규화되어야 함 ([0, 1] 범위)

2. **클래스 매핑**: CARLA 클래스와 리얼 클래스가 일치해야 함
   - 0: pedestrian
   - 1: car
   - 2: truck_bus
   - 3: bicycle_motorcycle
   - 4: traffic_sign

3. **이미지 형식**: `.jpg` 또는 `.png` 지원

## 🆘 문제 해결

### 라벨 형식 오류
```bash
# 라벨 형식 확인
python tools/prepare_carla_data.py --carla_dir path/to/carla_data
```

### 데이터 불일치
- 이미지와 라벨 이름이 정확히 일치해야 함
- 확장자 제외하고 이름이 동일해야 함

### 클래스 ID 오류
- 클래스 ID는 0-4 범위여야 함
- `carla_test_adverse.yaml` 참조

## 📞 협업

### 시뮬레이션 담당자에게 확인 요청 사항:

1. **데이터 구조**: `images/`와 `labels/` 디렉토리 포함 여부
2. **라벨 형식**: YOLO 형식인지 확인
3. **클래스 매핑**: 클래스 ID가 프로젝트와 일치하는지
4. **데이터 크기**: 대략적인 이미지 수
5. **FastCut 처리 정보**: 어떤 변환이 적용되었는지

준비되면 알려주시면 바로 통합 작업 진행하겠습니다! 🚀








