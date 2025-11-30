#!/bin/bash
# 프로젝트 정리 스크립트

echo "🧹 프로젝트 정리 시작..."
echo ""

# 1. 중복 압축 파일 정리 (최종 버전만 유지)
echo "📦 1. 중복 압축 파일 정리"
rm -f training_dataset.tar.gz training_dataset_full.tar.gz
echo "   ✅ 중복 압축 파일 삭제 완료 (training_dataset_final.tar.gz만 유지)"
echo ""

# 2. 임시 라벨 디렉토리 정리 (최종 labels에 이미 복사됨)
echo "📁 2. 임시 라벨 디렉토리 정리"
rm -rf carla_datasetv2/realB_split/train/labels_with_traffic
rm -rf carla_datasetv2/realB_split/train/labels_filtered
rm -rf carla_datasetv2/fakeB_split/train/labels_with_traffic
rm -rf carla_datasetv2/fakeB_split/train/labels_filtered
echo "   ✅ 임시 라벨 디렉토리 삭제 완료"
echo ""

# 3. 로그 파일 정리
echo "📝 3. 로그 파일 정리"
rm -f train.log train_stage2.log
echo "   ✅ 로그 파일 삭제 완료"
echo ""

# 4. 캐시 파일 정리 (재생성 가능)
echo "🗑️  4. 캐시 파일 정리"
find carla_datasetv2 -name "*.cache" -type f -delete
echo "   ✅ 캐시 파일 삭제 완료"
echo ""

# 5. 사용하지 않는 small 버전 디렉토리 정리
echo "📂 5. 사용하지 않는 small 버전 디렉토리 정리"
rm -rf carla_datasetv2/realA_split_small
rm -rf carla_datasetv2/realB_split_small
rm -rf carla_datasetv2/fakeB_split_small
echo "   ✅ small 버전 디렉토리 삭제 완료"
echo ""

# 6. 임시 변환 파일 정리
echo "🎬 6. 임시 변환 파일 정리"
rm -rf data/video_frames_1280
echo "   ✅ 임시 변환 파일 삭제 완료"
echo ""

# 7. __pycache__ 정리
echo "🐍 7. Python 캐시 파일 정리"
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "   ✅ Python 캐시 파일 삭제 완료"
echo ""

# 8. 빈 디렉토리 정리
echo "📁 8. 빈 디렉토리 정리"
find . -type d -empty -delete 2>/dev/null || true
echo "   ✅ 빈 디렉토리 삭제 완료"
echo ""

echo "✅ 프로젝트 정리 완료!"
echo ""
echo "📊 정리 후 상태:"
du -sh . 2>/dev/null | head -1

