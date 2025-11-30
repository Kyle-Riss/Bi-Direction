# 객체 검출 학습 전략 (표지판, 사람, 자동차)

## 🎯 목표

표지판, 사람, 자동차 객체를 정확하게 인식하기 위한 학습 전략

## 📊 현재 데이터 상황

- **`real_A` (CARLA 원본)**: 라벨 있음 ✅ (1,790개)
- **`fake_B` (FastCut 변환)**: `real_A`와 같은 라벨 사용 가능 ✅
- **`real_B` (실제 도로)**: 라벨 없음 ❌

## 🚀 학습 전략

### 전략: Hybrid Learning (Detection + Domain Adaptation)

```
┌─────────────────────────────────────────────────────────┐
│  Detection Loss (라벨 있음)                              │
│  - real_A: CARLA 원본으로 detection 학습                │
│  - fake_B: FastCut 변환으로 detection 학습              │
│  → 표지판, 사람, 자동차 검출 성능 향상                  │
└─────────────────────────────────────────────────────────┘
                    +
┌─────────────────────────────────────────────────────────┐
│  Domain Adaptation (라벨 없음)                           │
│  - real_A ↔ real_B: Adversarial + Feature Alignment    │
│  → 도메인 갭 감소, 실제 환경 적응                        │
└─────────────────────────────────────────────────────────┘
```

### Loss 구성

```
Total Loss = λ_det * Detection_Loss 
           + λ_adv * Adversarial_Loss 
           + λ_align * Feature_Alignment_Loss
```

- **Detection Loss**: `real_A`와 `fake_B`로 계산 (라벨 사용)
- **Adversarial Loss**: `real_A`와 `real_B` 간 도메인 구분 제거
- **Feature Alignment Loss**: `real_A`와 `real_B` feature 분포 정렬

## 📝 학습 방법

### 1. FastCut 도메인 적응 학습

```bash
python tools/train_fastcut_domain_adaptation.py \
    --real_A_dir carla_datasetv2/real_A \
    --real_B_dir carla_datasetv2/real_B \
    --fake_B_dir carla_datasetv2/fake_B \
    --labels_dir carla_datasetv2/labels \
    --epochs 20 \
    --batch_size 8 \
    --lambda_det 1.0 \
    --lambda_adv 0.1 \
    --lambda_align 0.1 \
    --img_size 320 \
    --device mps
```

### 2. 학습 과정

1. **Detection 학습** (real_A + fake_B)
   - YOLO detection loss 계산
   - 표지판, 사람, 자동차 검출 성능 향상
   - `fake_B`도 같은 라벨로 학습 (FastCut 변환 효과 활용)

2. **도메인 적응** (real_A ↔ real_B)
   - Adversarial loss: 도메인 구분 제거
   - Feature alignment: feature 분포 정렬
   - 실제 도로 환경에 적응

3. **통합 학습**
   - Detection + Domain Adaptation 동시 학습
   - 객체 검출 성능과 도메인 적응 균형

## 🎓 하이퍼파라미터 추천

### 초기 학습 (Detection 중심)
```python
lambda_det = 1.0    # Detection loss 가중치 (높게)
lambda_adv = 0.05   # Adversarial loss 가중치 (낮게)
lambda_align = 0.05 # Feature alignment 가중치 (낮게)
```

### 도메인 적응 강화
```python
lambda_det = 0.8    # Detection loss 가중치
lambda_adv = 0.2    # Adversarial loss 가중치 (높게)
lambda_align = 0.2  # Feature alignment 가중치 (높게)
```

### 균형 학습 (권장)
```python
lambda_det = 1.0    # Detection loss 가중치
lambda_adv = 0.1    # Adversarial loss 가중치
lambda_align = 0.1  # Feature alignment 가중치
```

## 📈 예상 효과

### Detection Loss 사용 시
- ✅ **표지판 검출**: 정확한 위치와 클래스 예측
- ✅ **사람 검출**: 보행자 인식 성능 향상
- ✅ **자동차 검출**: 차량 검출 정확도 향상
- ✅ **FastCut 활용**: 변환된 이미지로 추가 학습 데이터 확보

### Domain Adaptation 효과
- ✅ **실제 환경 적응**: CARLA → 실제 도로 전이
- ✅ **도메인 갭 감소**: 시뮬레이션과 실제 간 차이 최소화
- ✅ **일반화 성능**: 다양한 환경에서 안정적 성능

## 🔄 향후 개선 방안

### Option 1: real_B 라벨링 추가
- `real_B`에 대한 라벨 생성
- `real_B`로도 detection loss 계산
- 더 강력한 도메인 적응 가능

### Option 2: Pseudo-labeling
- 모델이 `real_B`에 대해 예측한 라벨 사용
- 신뢰도 높은 예측만 사용
- 라벨링 작업 없이 detection 학습 가능

### Option 3: Active Learning
- 모델이 불확실한 `real_B` 이미지 선택
- 선택된 이미지만 라벨링
- 효율적인 라벨링 전략

## 💡 핵심 포인트

1. **라벨 활용**: `real_A`와 `fake_B`의 라벨을 최대한 활용
2. **FastCut 효과**: 변환된 이미지로 데이터 증강 효과
3. **도메인 적응**: `real_B`로 실제 환경 적응 (라벨 없이도 가능)
4. **균형 학습**: Detection 성능과 도메인 적응 균형 유지

## 📊 클래스 정보

현재 라벨 클래스:
- **클래스 0**: vehicle (자동차)
- **클래스 1**: pedestrian (사람)

표지판은 현재 라벨에 없지만, 필요시 추가 가능:
- **클래스 2**: traffic_sign (표지판)





