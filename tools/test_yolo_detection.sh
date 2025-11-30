#!/bin/bash
# YOLO 라벨링 테스트 스크립트

echo "🔍 YOLO 라벨링 테스트 시작..."
echo "=================================="

# 디렉토리 설정
TEST_DIR="test_yolo_detection"
IMAGE_DIR="$TEST_DIR"
LABEL_DIR="$TEST_DIR/labels"
VIS_DIR="$TEST_DIR/visualized"

# 테스트 디렉토리 생성
mkdir -p "$LABEL_DIR"
mkdir -p "$VIS_DIR"

# 테스트 이미지 준비 (처음 5개만)
echo "📸 테스트 이미지 준비 중..."
if [ ! -d "$IMAGE_DIR" ] || [ -z "$(ls -A $IMAGE_DIR/*.jpg 2>/dev/null)" ]; then
    echo "   test 폴더에서 이미지 복사 중..."
    mkdir -p "$IMAGE_DIR"
    ls test/*.jpg 2>/dev/null | head -5 | xargs -I {} cp {} "$IMAGE_DIR/" 2>/dev/null
fi

echo "✅ 준비된 이미지: $(ls $IMAGE_DIR/*.jpg 2>/dev/null | wc -l | tr -d ' ') 개"

# YOLO 실행
echo ""
echo "🤖 YOLO 검출 실행 중..."
python3 tools/detect_with_yolo.py \
    --model yolov8n.pt \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$LABEL_DIR" \
    --conf 0.25

# 결과 확인
echo ""
echo "=================================="
echo "📊 결과 확인"
echo "=================================="
echo "생성된 라벨 파일:"
ls -lh "$LABEL_DIR"/*.txt 2>/dev/null | head -5 || echo "   라벨 파일 없음"

echo ""
echo "생성된 시각화 이미지:"
ls -lh "$VIS_DIR"/*.jpg 2>/dev/null | head -5 || echo "   시각화 이미지 없음"

echo ""
echo "라벨 파일 샘플 (첫 번째 파일):"
if [ -f "$(ls $LABEL_DIR/*.txt 2>/dev/null | head -1)" ]; then
    FIRST_LABEL=$(ls $LABEL_DIR/*.txt | head -1)
    echo "   파일: $(basename $FIRST_LABEL)"
    head -5 "$FIRST_LABEL"
else
    echo "   라벨 파일이 생성되지 않았습니다."
fi

echo ""
echo "✅ 테스트 완료!"
echo "   라벨 파일: $LABEL_DIR"
echo "   시각화 이미지: $VIS_DIR"








