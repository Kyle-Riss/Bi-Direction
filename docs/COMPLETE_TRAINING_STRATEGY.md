# 완전한 학습 전략 (real_B + test/ 함께 활용)

## 🎯 데이터셋 역할 정리

### 1. FastCut 데이터셋 (`carla_datasetv2/`)
- **`real_A`**: CARLA 시뮬레이션 원본 (1,790개, 라벨 있음)
- **`real_B`**: FastCut 스타일 참조용 (1,790개, 라벨 없음)
  - **역할**: 채색, 텍스처 등 스타일 참조
  - **이미 사용됨**: `real_A → fake_B` 변환에 활용
- **`fake_B`**: FastCut 변환 결과 (1,790개, 라벨 있음 - real_A와 동일)

### 2. 실제 도로 데이터셋 (`test/`)
- **이미지**: 1,156개
- **라벨**: 1,156개 (모두 매칭)
- **역할**: 실제 detection 학습 + 성능 평가

## 🚀 최종 학습 전략

### 전략: 3단계 Hybrid Learning

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: FastCut 데이터셋 학습                          │
│  - real_A + fake_B: Detection Loss (라벨 있음)          │
│  - real_B: Domain Adaptation (스타일 참조, 라벨 없음)   │
│  → FastCut 변환 효과 + 도메인 적응                       │
└─────────────────────────────────────────────────────────┘
                    +
┌─────────────────────────────────────────────────────────┐
│  Stage 2: 실제 도로 데이터 학습                          │
│  - test/: Detection Loss (라벨 있음)                    │
│  → 실제 환경에서의 검출 성능 향상                       │
└─────────────────────────────────────────────────────────┘
                    +
┌─────────────────────────────────────────────────────────┐
│  Stage 3: 통합 학습                                      │
│  - FastCut 데이터 + 실제 도로 데이터 혼합               │
│  → 최종 모델 완성                                        │
└─────────────────────────────────────────────────────────┘
```

## 📝 구체적 학습 방법

### Stage 1: FastCut 도메인 적응

```python
# real_A + fake_B: Detection 학습
# real_B: Domain Adaptation (스타일 참조)

from datasets.fastcut_dataset import create_fastcut_dataloader
from models.domain_adaptation import DomainAdaptiveYoloGRU

# FastCut 데이터 로더
fastcut_loader = create_fastcut_dataloader(
    real_A_dir='carla_datasetv2/real_A',
    real_B_dir='carla_datasetv2/real_B',  # 스타일 참조용
    fake_B_dir='carla_datasetv2/fake_B',
    labels_dir='carla_datasetv2/labels',
    mode='triplet'
)

# 학습
for real_A, real_B, fake_B, labels in fastcut_loader:
    # Detection Loss (real_A + fake_B)
    loss_det_A = model(real_A, labels)
    loss_det_fake = model(fake_B, labels)
    
    # Domain Adaptation (real_A ↔ real_B)
    outputs = model(x_sim=real_A, x_real=real_B, mode='train')
    loss_adv = compute_domain_adaptation_loss(outputs)
    
    total_loss = loss_det_A + 0.5 * loss_det_fake + loss_adv
```

### Stage 2: 실제 도로 데이터 학습

```python
# test/ 폴더로 실제 환경 detection 학습

from datasets.dataset import TemporalYOLODataset, create_dataloader

# 실제 도로 데이터 로더
real_loader = create_dataloader(
    img_dir='test',
    label_dir='test/labels',
    batch_size=8
)

# 학습
for images, labels in real_loader:
    loss = model(images, labels)
    loss.backward()
```

### Stage 3: 통합 학습

```python
# FastCut + 실제 도로 데이터 혼합

# 두 데이터 로더를 번갈아가며 학습
fastcut_iter = iter(fastcut_loader)
real_iter = iter(real_loader)

for batch_idx in range(min(len(fastcut_loader), len(real_loader))):
    # FastCut 배치
    real_A, real_B, fake_B, labels_fastcut = next(fastcut_iter)
    loss_fastcut = compute_fastcut_loss(real_A, fake_B, real_B, labels_fastcut)
    
    # 실제 도로 배치
    images_real, labels_real = next(real_iter)
    loss_real = model(images_real, labels_real)
    
    # 통합 loss
    total_loss = 0.7 * loss_fastcut + 0.3 * loss_real
    total_loss.backward()
```

## 🎓 권장 학습 순서

### Option 1: 순차 학습 (권장)

```bash
# 1단계: FastCut 도메인 적응
python tools/train_fastcut_domain_adaptation.py \
    --real_A_dir carla_datasetv2/real_A \
    --real_B_dir carla_datasetv2/real_B \
    --fake_B_dir carla_datasetv2/fake_B \
    --labels_dir carla_datasetv2/labels \
    --epochs 10 \
    --save_path models/fastcut_stage1.pt

# 2단계: 실제 도로 데이터 fine-tuning
python tools/train_real_detection.py \
    --img_dir test \
    --label_dir test/labels \
    --checkpoint models/fastcut_stage1.pt \
    --epochs 5 \
    --save_path models/final_model.pt
```

### Option 2: 통합 학습

```bash
# FastCut + 실제 도로 데이터 동시 학습
python tools/train_hybrid.py \
    --fastcut_real_A carla_datasetv2/real_A \
    --fastcut_real_B carla_datasetv2/real_B \
    --fastcut_fake_B carla_datasetv2/fake_B \
    --fastcut_labels carla_datasetv2/labels \
    --real_img_dir test \
    --real_label_dir test/labels \
    --epochs 15
```

## 📊 데이터 활용 요약

| 데이터셋 | 이미지 | 라벨 | 용도 |
|---------|--------|------|------|
| `real_A` | 1,790 | ✅ | Detection 학습 (CARLA 원본) |
| `real_B` | 1,790 | ❌ | FastCut 스타일 참조 (도메인 적응) |
| `fake_B` | 1,790 | ✅ | Detection 학습 (FastCut 변환) |
| `test/` | 1,156 | ✅ | Detection 학습 (실제 도로) |

## 💡 핵심 포인트

1. **`real_B`는 FastCut 스타일 참조용**
   - 이미 `fake_B` 생성에 사용됨
   - 도메인 적응 학습에 활용 (라벨 불필요)

2. **`test/`는 실제 detection 학습용**
   - 라벨이 있어서 detection loss 계산 가능
   - 실제 환경 검출 성능 향상

3. **함께 활용하는 것이 최선**
   - `real_B`: 스타일 변환 + 도메인 적응
   - `test/`: 실제 검출 성능 향상

## 🎯 최종 권장사항

**함께 활용하는 것을 강력히 권장합니다!**

이유:
- `real_B`는 이미 FastCut 변환에 사용됨 (fake_B 생성)
- `test/`는 실제 도로 데이터로 detection 성능 향상
- 두 데이터셋 모두 다른 목적으로 활용 가능
- 최종 모델 성능 최대화





