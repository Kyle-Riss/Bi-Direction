"""
비디오 프레임을 학습 이미지 형식(1280x1280 정사각형)으로 변환

학습 이미지와 동일한 형식으로 변환하여:
- 종횡비 문제 해결
- 해상도 일치
- 도메인 적응 용이
"""
import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def resize_with_letterbox(img, target_size=1280):
    """
    종횡비를 유지하면서 정사각형으로 리사이즈 (letterbox padding)
    
    Args:
        img: OpenCV 이미지 (BGR)
        target_size: 목표 크기 (정사각형)
    
    Returns:
        resized_img: 리사이즈된 이미지
        scale: 스케일 비율
        pad: 패딩 정보 (top, left)
    """
    h, w = img.shape[:2]
    
    # 스케일 계산 (긴 변을 기준)
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 리사이즈
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 패딩 계산 (중앙 정렬)
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    
    # 패딩 추가 (검은색)
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    return padded, scale, (pad_top, pad_left)


def extract_and_convert_video(video_path, output_dir, target_size=1280, frame_interval=30):
    """
    비디오에서 프레임을 추출하고 학습 이미지 형식으로 변환
    
    Args:
        video_path: 비디오 파일 경로
        output_dir: 출력 디렉토리
        target_size: 목표 이미지 크기 (기본: 1280x1280)
        frame_interval: 프레임 추출 간격 (기본: 30 = 1초마다 1프레임)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return False
    
    # 비디오 정보
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📹 비디오 정보:")
    print(f"   해상도: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   총 프레임: {total_frames}")
    print(f"   추출 간격: {frame_interval} 프레임 (약 {frame_interval/fps:.1f}초)")
    
    # 출력 디렉토리 생성
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    
    print(f"\n🔄 프레임 추출 및 변환 중...")
    
    with tqdm(total=total_frames // frame_interval) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # BGR -> RGB 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 1280x1280으로 변환 (letterbox padding)
                resized_frame, scale, pad = resize_with_letterbox(frame_rgb, target_size)
                
                # RGB -> BGR (저장용)
                resized_frame_bgr = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
                
                # 파일명: video_name_frame_000000.png
                frame_filename = f"{video_name}_frame_{saved_count:06d}.png"
                frame_path = os.path.join(video_output_dir, frame_filename)
                
                # PNG로 저장 (무손실)
                cv2.imwrite(frame_path, resized_frame_bgr)
                saved_count += 1
                pbar.update(1)
            
            frame_count += 1
    
    cap.release()
    
    print(f"\n✅ 완료!")
    print(f"   추출된 프레임: {saved_count}개")
    print(f"   저장 위치: {video_output_dir}")
    print(f"   형식: {target_size}x{target_size} PNG (정사각형)")
    
    return True


def process_video_directory(video_dir, output_dir, target_size=1280, frame_interval=30):
    """
    디렉토리 내 모든 비디오 처리
    """
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
    
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(ext))
        video_files.extend(Path(video_dir).glob(ext.upper()))
    
    if not video_files:
        print(f"❌ No video files found in {video_dir}")
        return
    
    print(f"📁 Found {len(video_files)} video files")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")
        print("-" * 70)
        extract_and_convert_video(
            str(video_path),
            output_dir,
            target_size,
            frame_interval
        )


def main():
    parser = argparse.ArgumentParser(
        description='비디오를 학습 이미지 형식(1280x1280)으로 변환'
    )
    parser.add_argument(
        '--video',
        type=str,
        help='비디오 파일 경로 (단일 파일)'
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        help='비디오 파일들이 있는 디렉토리'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/video_frames_1280',
        help='출력 디렉토리 (기본: data/video_frames_1280)'
    )
    parser.add_argument(
        '--target_size',
        type=int,
        default=1280,
        help='목표 이미지 크기 (기본: 1280, 학습 이미지와 동일)'
    )
    parser.add_argument(
        '--frame_interval',
        type=int,
        default=30,
        help='프레임 추출 간격 (기본: 30 = 약 1초마다 1프레임)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎬 비디오 → 학습 이미지 형식 변환")
    print("=" * 70)
    print(f"목표 크기: {args.target_size}x{args.target_size} (정사각형)")
    print(f"프레임 간격: {args.frame_interval}")
    print(f"출력 디렉토리: {args.output_dir}")
    print("=" * 70)
    
    if args.video:
        # 단일 비디오 처리
        extract_and_convert_video(
            args.video,
            args.output_dir,
            args.target_size,
            args.frame_interval
        )
    elif args.video_dir:
        # 디렉토리 내 모든 비디오 처리
        process_video_directory(
            args.video_dir,
            args.output_dir,
            args.target_size,
            args.frame_interval
        )
    else:
        print("❌ Error: --video 또는 --video_dir 중 하나를 지정해야 합니다.")
        parser.print_help()
        return
    
    print("\n" + "=" * 70)
    print("✅ 모든 변환 완료!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("   1. 변환된 이미지로 추론 테스트")
    print("   2. 필요시 라벨링 후 추가 학습 데이터로 사용")
    print("=" * 70)


if __name__ == "__main__":
    main()

