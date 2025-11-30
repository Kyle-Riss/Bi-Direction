# 테스트 데이터셋 준비 가이드

FastCut 처리된 CARLA 데이터 + 리얼 데이터를 혼합하여 약 2800개 테스트 데이터셋 준비

## 📋 개요

### 데이터 구성
- **리얼 데이터**: `test/` 폴더 (1156개 이미지)
- **CARLA 데이터**: FastCut 처리된 시뮬레이션 데이터
- **목표**: 총 약 2800개 (리얼 40% + CARLA 60%)

### 사용 목적
- 도메인 적응 모델 검증
- 시뮬레이션과 리얼 데이터 혼합 효과 테스트
- 일반화 성능 평가

## 🚀 사용 방법

### 기본 사용 (CARLA 데이터 있음)

```bash
python tools/prepare_test_dataset.py \
    --real_dir test \
    --carla_dir datasets/carla_data \
    --output_dir datasets/test_mixed \
    --target_total 2800 \
    --real_weight 0.4 \
    --carla_weight 0.6
```

### 리얼 데이터만 사용 (CARLA 데이터 없을 때)

```bash
python tools/prepare_test_dataset.py \
    --real_dir test \
    --output_dir datasets/test_mixed \
    --target_total 1156
```

## 📊 데이터 분할 계획

### 옵션 1: 리얼 40% + CARLA 60% (권장)
```
총 2800개
├── 리얼 데이터: 1120개 (40%)
└── CARLA 데이터: 1680개 (60%)
```

### 옵션 2: 균형 (50% : 50%)
```bash
python tools/prepare_test_dataset.py \
    --real_weight 0.5 \
    --carla_weight 0.5
```

### 옵션 3: 리얼 중심 (60% : 40%)
```bash
python tools/prepare_test_dataset.py \
    --real_weight 0.6 \
    --carla_weight 0.4
```

## 📁 출력 구조

```
datasets/test_mixed/
├── images/
│   ├── image1_real.jpg     (리얼 데이터)
│   ├── image1_carla.jpg    (CARLA 데이터)
│   └── ...
├── labels/
│   ├── image1_real.txt
│   ├── image1_carla.txt
│   └── ...
└── metadata/
    └── dataset_info.txt    (데이터셋 통계)
```

### 파일 이름 규칙
- 리얼 데이터: `{원본이름}_real.jpg` / `{원본이름}_real.txt`
- CARLA 데이터: `{원본이름}_carla.jpg` / `{원본이름}_carla.txt`

이렇게 하면 출처를 구분할 수 있습니다!

## 🧪 테스트 진행 방법

### 1단계: 데이터셋 준비
```bash
python tools/prepare_test_dataset.py \
    --real_dir test \
    --carla_dir datasets/carla_data \
    --output_dir datasets/test_mixed \
    --target_total 2800
```

### 2단계: 데이터셋 로더 생성
```python
from datasets.dataset import TemporalYOLODataset, create_dataloader

test_loader = create_dataloader(
    img_dir='datasets/test_mixed/images',
    label_dir='datasets/test_mixed/labels',
    batch_size=8,
    num_frames=3,
    img_size=320,
    shuffle=False  # 테스트는 shuffle 안함
)
```

### 3단계: 모델 평가
```python
from tools.evaluation_metrics import evaluate_yolo_predictions, print_evaluation_results

metrics = evaluate_yolo_predictions(
    predictions_dir='predictions/test_mixed',
    ground_truths_dir='datasets/test_mixed/labels',
    num_classes=5,
    iou_threshold=0.5
)

print_evaluation_results(metrics)
```

## 💡 데이터 혼합 전략

### 전략 1: 랜덤 혼합 (기본)
- 리얼과 CARLA 데이터를 랜덤하게 섞음
- 가장 간단하고 일반적인 방법

### 전략 2: 도메인별 평가
- 리얼과 CARLA를 분리해서 각각 평가
- 도메인별 성능 비교 가능

### 전략 3: 순차 평가
- 리얼 데이터로 먼저 평가
- CARLA 데이터로 추가 평가
- 혼합 데이터로 최종 평가

## 🔍 검증 체크리스트

- [ ] 리얼 데이터 확인 (`test/` 폴더)
- [ ] CARLA 데이터 확인 (선택적)
- [ ] 혼합 비율 설정 (기본: 40% : 60%)
- [ ] 데이터셋 준비 스크립트 실행
- [ ] 출력 디렉토리 확인
- [ ] 이미지-라벨 매칭 확인
- [ ] 메타데이터 확인
- [ ] 데이터 로더 테스트
- [ ] 평가 스크립트 실행

## ⚠️ 주의사항

1. **파일 이름 충돌 방지**
   - 리얼과 CARLA에 같은 이름의 파일이 있을 수 있음
   - 따라서 `_real`과 `_carla` 접미사 추가

2. **라벨 형식 통일**
   - 리얼과 CARLA 라벨이 모두 YOLO 형식이어야 함
   - 클래스 ID가 일치해야 함

3. **데이터 개수 조절**
   - 리얼 데이터가 1156개이므로, CARLA가 충분히 있어야 목표 개수 달성
   - CARLA 데이터가 부족하면 리얼 데이터만 사용

## 📈 예상 결과

### 혼합 데이터셋 (2800개)
- 도메인 다양성 증가
- 일반화 성능 향상 기대
- 시뮬/실 데이터 균형

### 평가 메트릭
- mAP@0.5
- Precision, Recall, F1
- 도메인별 성능 비교

## 🔄 워크플로우

```
1. 리얼 데이터 확인 (test/)
   ↓
2. CARLA 데이터 준비 (선택)
   ↓
3. 혼합 데이터셋 생성
   python tools/prepare_test_dataset.py
   ↓
4. 데이터 로더 생성
   ↓
5. 모델 평가
   ↓
6. 결과 분석
```

## 💬 Q&A

### Q: CARLA 데이터가 없어도 되나요?
A: 네! 리얼 데이터만으로도 테스트 가능합니다.

### Q: 비율을 조절할 수 있나요?
A: 네! `--real_weight`와 `--carla_weight`로 조절 가능합니다.

### Q: 목표 개수를 변경할 수 있나요?
A: 네! `--target_total`로 조절 가능합니다.

준비되면 바로 테스트 시작하세요! 🚀








