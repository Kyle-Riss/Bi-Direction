# 모델 경량화 가이드

현재 프로젝트의 모델 구조 분석 및 경량화 옵션

## 📊 현재 모델 구조

### 1. YoloLSTM (현재 기본 설정)

**구조:**
- CNN 백본: 32 → 64 → 128 채널
- LSTM: `hidden_size=256`, `num_layers=2`, `bidirectional=True`
- 출력 레이어: 512 → 256 → 2

**설정 (`config.py`):**
```python
MODELS = {
    "YoloLSTM": {
        "param": {
            "num_frames": 3,
            "hidden_size": 256,
            "num_layers": 2
        }
    }
}
```

**YOLO 모델:**
- `yolov8n.pt` (nano - 이미 경량 버전)
- 입력: 3 프레임 × 3 채널 = 9 채널

## 🎯 경량화 옵션

### 1. YOLO 모델 경량화

현재 사용 중인 모델: **YOLOv8n (nano)**

**추가 옵션:**

| 모델 | 파라미터 | 속도 | 정확도 |
|------|----------|------|--------|
| YOLOv8n | ~3.2M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| YOLOv8s | ~11.2M | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

**권장:** YOLOv8n 유지 (이미 가장 경량 버전)

**다른 경량 모델 고려:**
- YOLOv5s (Ultralytics)
- YOLOv7-tiny (다른 프레임워크 필요)

### 2. LSTM 크기 조절

#### 옵션 A: Hidden Size 감소
```python
# 현재: hidden_size=256
# 경량화: hidden_size=128
YoloLSTM(param={
    "num_frames": 3,
    "hidden_size": 128,  # 256 → 128
    "num_layers": 2
})
```

**효과:**
- 파라미터 약 50% 감소
- 메모리 사용량 감소

#### 옵션 B: Layer 수 감소
```python
# 현재: num_layers=2
# 경량화: num_layers=1
YoloLSTM(param={
    "num_frames": 3,
    "hidden_size": 256,
    "num_layers": 1  # 2 → 1
})
```

**효과:**
- 파라미터 약 25% 감소
- 연산 속도 향상

#### 옵션 C: Bidirectional 비활성화
```python
# YoloLSTM 수정 필요 (현재 코드에 bidirectional 옵션 없음)
# bidirectional=False로 변경 시
# 파라미터 약 50% 감소 (hidden_size * 2 → hidden_size)
```

### 3. GRU로 전환 (권장)

**GRU (Gated Recurrent Unit) 장점:**
- LSTM 대비 파라미터 약 25% 감소
- 연산 속도 향상
- 메모리 사용량 감소

**사용 방법:**
```python
from models.YoloGRU import YoloGRU

model = YoloGRU(param={
    "num_frames": 3,
    "hidden_size": 256,
    "num_layers": 2,
    "bidirectional": True
})
```

**LightYoloGRU (극도 경량):**
```python
from models.YoloGRU import LightYoloGRU

model = LightYoloGRU(param={
    "num_frames": 3,
    "hidden_size": 128,  # 더 작은 hidden size
    "num_layers": 1,     # 단일 레이어
    "bidirectional": False  # bidirectional 비활성화
})
```

### 4. 이미지 크기 조절

**현재 설정 (`config.py`):**
```python
MULTISCALE = {
    "coarse": {
        "img_size": 128,      # 160 → 128 (메모리 절약)
        "batch_size": 8,
    },
    "fine": {
        "img_size": 320,      # 640 → 320 (메모리 절약)
        "batch_size": 2,
    }
}
```

**추가 경량화:**
```python
MULTISCALE = {
    "coarse": {
        "img_size": 96,       # 128 → 96
        "batch_size": 4,      # 8 → 4
    },
    "fine": {
        "img_size": 224,      # 320 → 224
        "batch_size": 1,      # 2 → 1
    }
}
```

### 5. 배치 크기 조절

**현재 설정:**
- Coarse: batch_size=8
- Fine: batch_size=2

**경량화 옵션:**
- Coarse: batch_size=4 또는 2
- Fine: batch_size=1

**주의:** 배치 크기 감소 시 학습 안정성 저하 가능

### 6. UltraLightYoloLSTM (극도 경량)

**이미 구현됨 (`models/ultra_light_model.py`)**

**특징:**
- CNN 채널: 8 → 16 → 32 (기존: 32 → 64 → 128)
- LSTM: hidden_size=32, num_layers=1
- Bidirectional: False
- 이미지 크기: 64×64

**사용:**
```python
from models.ultra_light_model import UltraLightYoloLSTM

model = UltraLightYoloLSTM(
    num_frames=3,
    hidden_size=32,
    num_layers=1
)
```

## 📈 경량화 비교표

| 모델 변형 | 파라미터 | 메모리 (MB) | 상대 크기 |
|-----------|----------|-------------|-----------|
| YoloLSTM (현재) | ~2.5M | ~50 | 100% |
| YoloLSTM (hidden=128) | ~1.2M | ~25 | 48% |
| YoloGRU | ~1.9M | ~38 | 76% |
| LightYoloGRU | ~0.8M | ~16 | 32% |
| UltraLightYoloLSTM | ~0.2M | ~4 | 8% |

## 🛠️ 모델 분석 실행

모델 크기를 분석하려면:

```bash
python3 tools/analyze_model_size.py
```

이 스크립트는:
- 모든 모델 변형의 파라미터 수 계산
- 메모리 사용량 추정
- 경량화 옵션 비교

## 💡 권장 경량화 전략

### 1단계: 빠른 경량화 (성능 유지)
- LSTM → GRU 전환 (25% 감소)
- Hidden size: 256 → 128 (50% 감소)

### 2단계: 중간 경량화 (성능 일부 희생)
- Layer 수: 2 → 1 (25% 감소)
- Bidirectional: False (50% 감소)
- 이미지 크기: 128 → 96

### 3단계: 극도 경량화 (성능 크게 희생)
- UltraLightYoloLSTM 사용
- 이미지 크기: 64×64
- 배치 크기: 1

## ⚙️ 설정 파일 수정

`config.py`에서 경량화 옵션 적용:

```python
# 경량화 설정 예시
MODELS = {
    "YoloLSTM": {
        "param": {
            "num_frames": 3,
            "hidden_size": 128,  # 256 → 128
            "num_layers": 1      # 2 → 1
        }
    }
}

MULTISCALE = {
    "coarse": {
        "img_size": 96,         # 128 → 96
        "batch_size": 4,        # 8 → 4
        "epochs": 30,
        "lr": 1e-4,
    },
    "fine": {
        "img_size": 224,        # 320 → 224
        "batch_size": 1,        # 2 → 1
        "epochs": 5,
        "lr": 1e-5,
    }
}
```

## 🔬 추가 최적화 기법

### 1. Pruning (학습 후)
```python
import torch.nn.utils.prune as prune

# 모델 pruning 예시
for module in model.modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.2)
```

### 2. Quantization (양자화)
```python
# INT8 양자화
model_quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

### 3. 지연 업데이트 (Gradient Accumulation)
```python
# 작은 배치 크기 + gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 📝 체크리스트

- [ ] 현재 모델 파라미터 수 확인
- [ ] GRU 모델로 전환 테스트
- [ ] Hidden size 감소 테스트
- [ ] Layer 수 감소 테스트
- [ ] 이미지 크기 조절 테스트
- [ ] 배치 크기 조절 테스트
- [ ] 성능 비교 (정확도 vs 속도)
- [ ] 최적 경량화 설정 결정








