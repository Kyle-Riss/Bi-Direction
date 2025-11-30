# FastCut 데이터 활용 가이드

FastCut으로 변환된 CARLA 데이터를 활용한 도메인 적응 학습

## 📋 데이터 구조

```
carla_datasetv2/
├── real_A/      # CARLA 시뮬레이션 원본 (소스 도메인)
├── real_B/      # 실제 도로 주행 이미지 (타겟 도메인, 별도 데이터셋)
├── fake_B/      # FastCut 변환 이미지 (real_A → real_B 스타일)
├── images/      # 원본 CARLA 이미지
├── labels/      # YOLO 형식 라벨
└── metadata/    # 시나리오 메타데이터
```

### ⚠️ 중요 개념

**`real_B`는 `real_A`와는 완전히 다른 이미지입니다!**

- `real_A`: CARLA 시뮬레이션에서 생성된 원본 이미지
- `real_B`: 실제 도로에서 촬영한 별도의 이미지 (FastCut 학습용 스타일 참조)
- `fake_B`: `real_A`를 `real_B`의 스타일로 변환한 결과

**FastCut의 목적**: `real_A` (시뮬레이션)를 `real_B` (실제 도로)의 스타일로 변환하여 도메인 갭을 줄이는 것

### 데이터 설명

- **`real_A`**: CARLA 시뮬레이션 원본 이미지 (1,790개)
  - 소스 도메인
  - 시뮬레이션 특성 (렌더링, 조명 등)
  
- **`real_B`**: 실제 도로 주행 이미지 (1,790개)
  - **타겟 도메인**
  - **중요**: `real_A`와는 완전히 다른 이미지 (별도의 실제 도로 데이터셋)
  - 실제 환경 특성
  - FastCut 학습 시 스타일 참조용으로 사용
  
- **`fake_B`**: FastCut으로 변환된 이미지 (1,790개)
  - `real_A`를 `real_B` 스타일로 변환한 결과
  - `real_A`와 같은 시나리오/프레임에 대응
  - 도메인 갭을 사전에 감소시킨 합성 이미지

## 🚀 사용 방법

### 1. FastCut 데이터셋 로더 사용

```python
from datasets.fastcut_dataset import create_fastcut_dataloader

# Triplet 모드: real_A, real_B, fake_B 모두 사용
loader = create_fastcut_dataloader(
    real_A_dir='carla_datasetv2/real_A',
    real_B_dir='carla_datasetv2/real_B',
    fake_B_dir='carla_datasetv2/fake_B',
    labels_dir='carla_datasetv2/labels',
    img_size=320,
    batch_size=8,
    mode='triplet'  # 'triplet', 'pair', 'adaptation'
)
```

### 2. 학습 모드 선택

#### Mode 1: Triplet (real_A, real_B, fake_B)
```python
# 3-way 학습: 원본, 타겟, 변환 이미지 모두 활용
loader = create_fastcut_dataloader(
    real_A_dir='carla_datasetv2/real_A',
    real_B_dir='carla_datasetv2/real_B',
    fake_B_dir='carla_datasetv2/fake_B',
    mode='triplet'
)

for real_A, real_B, fake_B, labels in loader:
    # real_A: CARLA 원본
    # real_B: 실제 도로
    # fake_B: FastCut 변환
    # 3가지 모두 활용 가능
    pass
```

#### Mode 2: Pair (real_A, fake_B)
```python
# FastCut 변환 효과 검증
loader = create_fastcut_dataloader(
    real_A_dir='carla_datasetv2/real_A',
    real_B_dir='carla_datasetv2/real_B',
    fake_B_dir='carla_datasetv2/fake_B',
    mode='pair'
)

for real_A, fake_B, labels in loader:
    # real_A와 fake_B 비교
    # FastCut 변환 품질 평가
    pass
```

#### Mode 3: Adaptation (real_A, real_B)
```python
# 기존 도메인 적응 방식
loader = create_fastcut_dataloader(
    real_A_dir='carla_datasetv2/real_A',
    real_B_dir='carla_datasetv2/real_B',
    fake_B_dir='carla_datasetv2/fake_B',
    mode='adaptation'
)

for real_A, real_B, labels in loader:
    # 도메인 적응 학습
    pass
```

## 🎓 학습 전략

### 전략 1: FastCut 변환 데이터 활용

```python
# fake_B를 추가 학습 데이터로 활용
# real_A와 fake_B를 함께 학습하여 도메인 적응 효과 향상

model.train()
for real_A, real_B, fake_B, labels in loader:
    # 1. real_A로 기본 학습
    loss_A = model(real_A, labels)
    
    # 2. fake_B로 변환 효과 학습
    loss_fake = model(fake_B, labels)
    
    # 3. real_B로 타겟 도메인 적응
    loss_B = model(real_B, labels)
    
    total_loss = loss_A + 0.5 * loss_fake + loss_B
    total_loss.backward()
```

### 전략 2: 3-way 도메인 적응

```python
from models.domain_adaptation import DomainAdaptiveYoloGRU

model = DomainAdaptiveYoloGRU(
    use_adversarial=True,
    use_feature_align=True
)

for real_A, real_B, fake_B, labels in loader:
    # 3가지 도메인 모두 활용
    outputs = model(
        x_sim=real_A,      # 시뮬레이션 원본
        x_real=real_B,     # 실제 도로
        x_fake=fake_B,     # FastCut 변환 (새로 추가 가능)
        mode='train'
    )
```

### 전략 3: Progressive Training

```python
# 1단계: real_A로 기본 학습
# 2단계: fake_B로 변환 데이터 학습
# 3단계: real_B로 도메인 적응

# Stage 1: Base training
for real_A, _, _, labels in loader:
    loss = model(real_A, labels)
    loss.backward()

# Stage 2: FastCut adaptation
for _, _, fake_B, labels in loader:
    loss = model(fake_B, labels)
    loss.backward()

# Stage 3: Real domain adaptation
for _, real_B, _, labels in loader:
    loss = model(real_B, labels)
    loss.backward()
```

## 📊 데이터 통계

- **총 이미지**: 5,370개
  - `real_A`: 1,790개
  - `real_B`: 1,790개
  - `fake_B`: 1,790개
  
- **라벨**: 1,790개 (YOLO 형식)
- **클래스**: 2개 (vehicle, pedestrian)

## 🔧 도메인 적응 학습 예제

```python
import torch
from datasets.fastcut_dataset import create_fastcut_dataloader
from models.domain_adaptation import DomainAdaptiveYoloGRU

# 데이터 로더
train_loader = create_fastcut_dataloader(
    real_A_dir='carla_datasetv2/real_A',
    real_B_dir='carla_datasetv2/real_B',
    fake_B_dir='carla_datasetv2/fake_B',
    labels_dir='carla_datasetv2/labels',
    batch_size=8,
    mode='triplet'
)

# 모델
model = DomainAdaptiveYoloGRU(
    use_adversarial=True,
    use_feature_align=True
).to('mps')

# 학습
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    for real_A, real_B, fake_B, labels in train_loader:
        real_A = real_A.to('mps')
        real_B = real_B.to('mps')
        fake_B = fake_B.to('mps')
        
        # Forward
        outputs = model(
            x_sim=real_A,
            x_real=real_B,
            mode='train'
        )
        
        # Loss 계산 및 역전파
        loss = compute_domain_adaptation_loss(outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 💡 활용 팁

1. **FastCut 변환 품질 확인**: `real_A`와 `fake_B`를 비교하여 변환 품질 평가
2. **Progressive Training**: 단계별 학습으로 도메인 적응 효과 향상
3. **Data Augmentation**: `fake_B`를 추가 데이터로 활용하여 데이터 증강 효과
4. **Domain Gap 측정**: `real_A`와 `real_B` 간 도메인 갭 vs `fake_B`와 `real_B` 간 갭 비교

## 📝 참고

- FastCut: 빠른 스타일 변환을 위한 GAN 기반 방법
- 도메인 적응: 시뮬레이션 → 실제 환경 전이 학습
- YOLO-GRU: Temporal 정보를 활용한 객체 검출

