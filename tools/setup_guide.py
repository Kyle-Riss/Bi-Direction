"""
동영상 데이터 처리 가이드
"""
import os

def show_instructions():
    print("🎥 동영상 데이터 처리 가이드")
    print("=" * 50)
    
    print("\n📁 1단계: 동영상 파일 복사")
    print("다음 명령어로 동영상들을 프로젝트로 복사하세요:")
    print()
    print("cp /Users/hayubin/Downloads/archive/*.mp4 /Users/hayubin/Bi-Direction/data/videos/")
    print("cp /Users/hayubin/Downloads/archive/*.avi /Users/hayubin/Bi-Direction/data/videos/")
    print("cp /Users/hayubin/Downloads/archive/*.mov /Users/hayubin/Bi-Direction/data/videos/")
    print()
    
    print("📊 2단계: 프레임 추출")
    print("동영상들을 프레임으로 변환:")
    print()
    print("python tools/video_processor.py --video_dir data/videos --frames_dir data/frames --dataset_dir data")
    print()
    
    print("🔧 3단계: 데이터셋 구조 확인")
    print("생성된 데이터셋 구조:")
    print()
    print("data/")
    print("├── train/")
    print("│   ├── images/  # 학습용 이미지")
    print("│   └── labels/  # 학습용 라벨 (비어있음)")
    print("├── val/")
    print("│   ├── images/  # 검증용 이미지")
    print("│   └── labels/  # 검증용 라벨 (비어있음)")
    print("└── test/")
    print("    ├── images/  # 테스트용 이미지")
    print("    └── labels/  # 테스트용 라벨 (비어있음)")
    print()
    
    print("🎯 4단계: 라벨링 (필요시)")
    print("YOLO 형식 라벨 파일 생성:")
    print("- 각 이미지에 대해 .txt 파일 생성")
    print("- 형식: class_id x_center y_center width height")
    print("- 좌표는 이미지 크기 대비 정규화된 값 (0-1)")
    print()
    
    print("🚀 5단계: 학습 실행")
    print("데이터 준비 완료 후:")
    print()
    print("python main.py")
    print()
    
    print("💡 팁:")
    print("- 동영상이 많으면 frame_interval을 조정하여 프레임 수 조절")
    print("- 예: --frame_interval 5 (5프레임마다 1개씩 추출)")
    print("- 메모리 부족 시 배치 크기 줄이기")

if __name__ == "__main__":
    show_instructions()

