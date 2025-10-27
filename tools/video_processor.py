"""
비디오 데이터 처리 도구
동영상을 프레임으로 분할하고 YOLO 형식 데이터셋을 생성
"""
import os
import cv2
import glob
import shutil
from pathlib import Path
import argparse


def extract_frames_from_video(video_path, output_dir, frame_interval=1):
    """
    비디오에서 프레임을 추출하여 이미지로 저장
    
    Args:
        video_path (str): 비디오 파일 경로
        output_dir (str): 프레임 저장 디렉토리
        frame_interval (int): 프레임 추출 간격 (1 = 모든 프레임)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    frame_count = 0
    saved_count = 0
    
    # 비디오 이름으로 디렉토리 생성
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # 프레임을 이미지로 저장
            frame_filename = f"frame_{saved_count:06d}.jpg"
            frame_path = os.path.join(video_output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path}")
    return True


def process_videos_to_frames(video_dir, output_dir, frame_interval=1):
    """
    디렉토리 내 모든 비디오를 프레임으로 변환
    
    Args:
        video_dir (str): 비디오 파일들이 있는 디렉토리
        output_dir (str): 프레임 저장 디렉토리
        frame_interval (int): 프레임 추출 간격
    """
    # 지원하는 비디오 포맷
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
    
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
        video_files.extend(glob.glob(os.path.join(video_dir, ext.upper())))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for video_path in video_files:
        print(f"Processing: {os.path.basename(video_path)}")
        extract_frames_from_video(video_path, output_dir, frame_interval)


def create_yolo_dataset_structure(frames_dir, output_dir):
    """
    프레임들을 YOLO 데이터셋 구조로 정리
    
    Args:
        frames_dir (str): 프레임들이 있는 디렉토리
        output_dir (str): YOLO 데이터셋 출력 디렉토리
    """
    # YOLO 데이터셋 구조 생성
    train_img_dir = os.path.join(output_dir, 'train', 'images')
    train_label_dir = os.path.join(output_dir, 'train', 'labels')
    val_img_dir = os.path.join(output_dir, 'val', 'images')
    val_label_dir = os.path.join(output_dir, 'val', 'labels')
    test_img_dir = os.path.join(output_dir, 'test', 'images')
    test_label_dir = os.path.join(output_dir, 'test', 'labels')
    
    for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir, 
                     test_img_dir, test_label_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 모든 프레임 파일 수집
    frame_files = []
    for video_dir in os.listdir(frames_dir):
        video_path = os.path.join(frames_dir, video_dir)
        if os.path.isdir(video_path):
            for frame_file in os.listdir(video_path):
                if frame_file.endswith('.jpg'):
                    frame_files.append(os.path.join(video_path, frame_file))
    
    frame_files.sort()
    
    # 데이터 분할 (80% train, 10% val, 10% test)
    total_frames = len(frame_files)
    train_end = int(total_frames * 0.8)
    val_end = int(total_frames * 0.9)
    
    train_frames = frame_files[:train_end]
    val_frames = frame_files[train_end:val_end]
    test_frames = frame_files[val_end:]
    
    # 프레임 복사
    for i, frame_path in enumerate(train_frames):
        filename = f"train_{i:06d}.jpg"
        shutil.copy2(frame_path, os.path.join(train_img_dir, filename))
    
    for i, frame_path in enumerate(val_frames):
        filename = f"val_{i:06d}.jpg"
        shutil.copy2(frame_path, os.path.join(val_img_dir, filename))
    
    for i, frame_path in enumerate(test_frames):
        filename = f"test_{i:06d}.jpg"
        shutil.copy2(frame_path, os.path.join(test_img_dir, filename))
    
    print(f"Dataset created:")
    print(f"  Train: {len(train_frames)} frames")
    print(f"  Val: {len(val_frames)} frames")
    print(f"  Test: {len(test_frames)} frames")


def main():
    parser = argparse.ArgumentParser(description='Process videos to YOLO dataset')
    parser.add_argument('--video_dir', type=str, default='data/videos',
                       help='Directory containing video files')
    parser.add_argument('--frames_dir', type=str, default='data/frames',
                       help='Directory to save extracted frames')
    parser.add_argument('--dataset_dir', type=str, default='data',
                       help='Directory to save YOLO dataset')
    parser.add_argument('--frame_interval', type=int, default=1,
                       help='Frame extraction interval (1 = every frame)')
    
    args = parser.parse_args()
    
    print("🎥 Video Processing Pipeline")
    print("=" * 50)
    
    # 1단계: 비디오를 프레임으로 변환
    print("Step 1: Extracting frames from videos...")
    process_videos_to_frames(args.video_dir, args.frames_dir, args.frame_interval)
    
    # 2단계: YOLO 데이터셋 구조 생성
    print("\nStep 2: Creating YOLO dataset structure...")
    create_yolo_dataset_structure(args.frames_dir, args.dataset_dir)
    
    print("\n✅ Video processing completed!")
    print(f"Dataset saved to: {args.dataset_dir}")


if __name__ == "__main__":
    main()

