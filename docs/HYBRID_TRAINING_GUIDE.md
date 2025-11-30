# Hybrid 학습 가이드 (FastCut + 실제 도로 데이터 번갈아가기)

## 🎯 학습 전략

FastCut 데이터와 실제 도로 데이터를 **번갈아가며** 학습하여 최적의 성능 달성

## 📊 데이터 구성

### FastCut 데이터셋
- **real_A**: CARLA 원본 (1,790개, 라벨 있음)
- **real_B**: 스타일 참조 (1,790개, 라벨 없음)
- **fake_B**: FastCut 변환 (1,790개, 라벨 있음)

### 실제 도로 데이터셋
- **test/**: 실제 도로 이미지 (1,156개, 라벨 있음)

## 🚀 실행 방법

### 기본 실행

```bash
python tools/train_hybrid_fastcut_real.py \
    --real_A_dir carla_datasetv2/real_A \
    --real_B_dir carla_datasetv2/real_B \
    --fake_B_dir carla_datasetv2/fake_B \
    --fastcut_labels_dir carla_datasetv2/labels \
    --real_img_dir test \
    --real_label_dir test/labels \
    --epochs 20 \
    --batch_size 8 \
    --fastcut_weight 0.7 \
    --real_weight 0.3 \
    --device mps
```

### 하이퍼파라미터 조정

```bash
# FastCut 중심 학습 (도메인 적응 강조)
python tools/train_hybrid_fastcut_real.py \
    --fastcut_weight 0.8 \
    --real_weight 0.2 \
    ...

# 실제 도로 중심 학습 (Detection 성능 강조)
python tools/train_hybrid_fastcut_real.py \
    --fastcut_weight 0.5 \
    --real_weight 0.5 \
    ...

# 균형 학습 (권장)
python tools/train_hybrid_fastcut_real.py \
    --fastcut_weight 0.7 \
    --real_weight 0.3 \
    ...
```

## 📝 학습 과정

### 번갈아가며 학습하는 방식

```
배치 1: FastCut 데이터
  - real_A + fake_B: Detection Loss
  - real_B: Domain Adaptation
  
배치 2: 실제 도로 데이터
  - test/: Detection Loss
  
배치 3: FastCut 데이터
  - real_A + fake_B: Detection Loss
  - real_B: Domain Adaptation
  
배치 4: 실제 도로 데이터
  - test/: Detection Loss
  
... (반복)
```

### Loss 구성

```
Total Loss = fastcut_weight × (Detection_FastCut + Domain_Adaptation)
           + real_weight × Detection_Real
```

## 🎓 권장 설정

### 초기 학습 (도메인 적응 중심)
```python
fastcut_weight = 0.8
real_weight = 0.2
lambda_adv = 0.2
lambda_align = 0.2
```

### 중기 학습 (균형)
```python
fastcut_weight = 0.7
real_weight = 0.3
lambda_adv = 0.1
lambda_align = 0.1
```

### 후기 학습 (Detection 성능 중심)
```python
fastcut_weight = 0.5
real_weight = 0.5
lambda_adv = 0.05
lambda_align = 0.05
```

## 💡 핵심 포인트

1. **번갈아가며 학습**: 두 데이터셋의 장점을 모두 활용
2. **가중치 조절**: 학습 단계에 따라 fastcut_weight와 real_weight 조정
3. **도메인 적응**: real_B로 스타일 학습 (라벨 불필요)
4. **Detection 성능**: test/로 실제 검출 성능 향상

## 📈 예상 효과

- ✅ FastCut 변환 효과 활용
- ✅ 도메인 적응 (CARLA → 실제 도로)
- ✅ 실제 환경 검출 성능 향상
- ✅ 표지판, 사람, 자동차 정확한 인식





