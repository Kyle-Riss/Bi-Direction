# 도메인 적응 가이드

CARLA 시뮬레이션 데이터와 리얼 데이터 간 도메인 갭을 줄이기 위한 도메인 적응 기법 적용

## 📋 개요

### 문제
- 시뮬레이션 데이터와 실제 데이터 간 도메인 차이 존재
- 모델이 시뮬레이션 데이터에 과적합될 위험
- 실제 환경에서 성능 저하 발생 가능

### 해결 방법
1. **Adversarial Training**: 도메인 판별기를 통한 도메인 구분 제거
2. **Feature Alignment**: 도메인 간 feature 분포 정렬
3. **Photorealism 변환**: CycleGAN 등을 통한 스타일 변환
4. **혼합 훈련**: 시뮬/실 데이터 비율 조절

## 🔧 구현된 기법

### 1. Adversarial Training

**원리:**
- 도메인 판별기(Discriminator)가 시뮬레이션과 리얼을 구분하려고 함
- 모델은 판별기가 구분 못하도록 학습 (Gradient Reversal)

**구현:**
```python
from models.domain_adaptation import DomainAdaptiveYoloGRU

model = DomainAdaptiveYoloGRU(
    use_adversarial=True,  # Adversarial training 활성화
    use_feature_align=True,  # Feature alignment 활성화
)
```

**Loss:**
```
L_total = L_detection - λ_adv * L_adversarial + λ_align * L_align
```

### 2. Feature Alignment

**원리:**
- MMD (Maximum Mean Discrepancy) 또는 CORAL을 사용하여
- 시뮬레이션과 리얼 도메인의 feature 분포를 정렬

**방법:**
- **MMD**: Gaussian kernel을 사용한 분포 거리 측정
- **CORAL**: Covariance Alignment를 통한 feature 정렬

### 3. Photorealism 변환

**원리:**
- CycleGAN 등을 사용하여 시뮬레이션 이미지를 실제 영상처럼 변환
- 도메인 갭을 사전에 감소

**사용:**
```bash
# CycleGAN 모델이 있다면
python tools/style_transfer.py \
    --input_dir datasets/carla_data/train/images \
    --output_dir datasets/carla_data/train/images_photoreal \
    --generator path/to/cyclegan_generator.pt
```

## 🚀 학습 방법

### 기본 도메인 적응 학습

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

### 단계별 학습 전략

#### 1단계: CARLA 단독 학습 (Pre-training)
```python
# CARLA 데이터만으로 기본 학습
# 이미 구현됨 (main.py 사용)
```

#### 2단계: 도메인 적응 학습
```bash
# Adversarial + Feature Alignment
python tools/train_domain_adaptation.py \
    --carla_train datasets/carla_data/train \
    --real_train datasets/real_data/train \
    --lambda_adv 0.1 \
    --lambda_align 0.1
```

#### 3단계: 혼합 훈련 (Fine-tuning)
```python
# datasets/mixed_dataset.py 사용
from datasets.mixed_dataset import create_mixed_training_dataloaders

train_loader, val_loader = create_mixed_training_dataloaders(
    carla_train_dir='datasets/carla_data/train',
    real_train_dir='datasets/real_data/train',
    carla_weight=0.5,  # 50% : 50%
    real_weight=0.5,
    mode='concat'
)
```

## 📊 평가 메트릭

### mAP (mean Average Precision)

```bash
python tools/evaluation_metrics.py \
    --predictions test/labels \
    --ground_truths test/labels \
    --num_classes 5 \
    --iou_threshold 0.5
```

### Python 코드에서 평가

```python
from tools.evaluation_metrics import evaluate_yolo_predictions, print_evaluation_results

metrics = evaluate_yolo_predictions(
    predictions_dir='test/labels',
    ground_truths_dir='test/labels',  # 실제 GT 디렉토리
    num_classes=5,
    iou_threshold=0.5
)

print_evaluation_results(metrics)
```

## ⚙️ 하이퍼파라미터 튜닝

### Adversarial Loss 가중치 (λ_adv)
- **낮은 값 (0.01-0.05)**: 도메인 적응 약함, detection 성능 유지
- **중간 값 (0.1-0.2)**: 균형잡힌 도메인 적응 (권장)
- **높은 값 (0.5+)**: 강한 도메인 적응, detection 성능 저하 가능

### Feature Alignment 가중치 (λ_align)
- **낮은 값 (0.01-0.05)**: 약한 feature 정렬
- **중간 값 (0.1-0.2)**: 균형잡힌 정렬 (권장)
- **높은 값 (0.5+)**: 강한 정렬, feature 손실 가능

### 혼합 비율 조절

```python
# 시뮬레이션 중심 (초기 학습)
carla_weight=0.7, real_weight=0.3

# 균형 (중간 학습)
carla_weight=0.5, real_weight=0.5

# 리얼 중심 (후기 학습)
carla_weight=0.3, real_weight=0.7
```

## 📈 예상 성능 개선

도메인 적응 기법 적용 시:

| 기법 | mAP 개선 | F1 개선 | 일반화 성능 |
|------|----------|---------|-------------|
| Adversarial Training | +3-5% | +2-4% | ⭐⭐⭐⭐ |
| Feature Alignment | +2-4% | +2-3% | ⭐⭐⭐ |
| Photorealism | +5-8% | +4-6% | ⭐⭐⭐⭐⭐ |
| 혼합 적용 | +8-12% | +6-10% | ⭐⭐⭐⭐⭐ |

## 🔬 실험 제안

### 실험 1: Adversarial Loss 가중치 조절
```bash
for lambda_adv in 0.01 0.05 0.1 0.2 0.5; do
    python tools/train_domain_adaptation.py \
        --lambda_adv $lambda_adv \
        --save_path models/domain_adapt_adv${lambda_adv}.pt
done
```

### 실험 2: 시뮬/실 데이터 비율 조절
```python
ratios = [
    (0.7, 0.3),  # 시뮬 70%
    (0.5, 0.5),  # 균형
    (0.3, 0.7),  # 리얼 70%
]

for carla_w, real_w in ratios:
    # 학습 수행
    pass
```

### 실험 3: Feature Alignment 방법 비교
- MMD vs CORAL
- 각각의 성능 비교

## 📝 체크리스트

- [ ] CARLA 데이터 준비 (시뮬레이션 담당)
- [ ] 리얼 데이터 준비 완료 ✅
- [ ] 도메인 적응 모델 구현 ✅
- [ ] 평가 메트릭 구현 ✅
- [ ] 혼합 학습 파이프라인 구현 ✅
- [ ] Photorealism 변환 적용 (선택)
- [ ] 하이퍼파라미터 튜닝
- [ ] 성능 평가 및 비교

## 💡 참고 자료

1. **Adversarial Domain Adaptation**: 
   - "Unsupervised Domain Adaptation by Backpropagation" (Ganin et al.)

2. **Feature Alignment**:
   - "Deep CORAL: Correlation Alignment for Deep Domain Adaptation" (Sun et al.)

3. **Photorealism**:
   - "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (Zhu et al.)

4. **Mixed Training**:
   - "Synthetic Data for Deep Learning" (Nikolenko, 2021)








