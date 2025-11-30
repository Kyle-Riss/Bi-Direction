"""
비디오 추론: 종횡비 + 크롭 정리

720×1280 → 도로 중심 720×720 크롭 → 192×192 리사이즈 → 추론

학습 이미지와 동일한 종횡비(정사각형)로 맞춰서 왜곡 최소화
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import os
from tqdm import tqdm


def crop_to_square(frame, crop_mode='center'):
    """
    프레임을 정사각형으로 크롭
    
    Args:
        frame: OpenCV 이미지 (BGR)
        crop_mode: 'center' (중앙) 또는 'bottom' (하단, 도로 중심)
    
    Returns:
        cropped_frame: 정사각형 크롭된 이미지
        crop_info: (x, y, w, h) 크롭 정보
    """
    h, w = frame.shape[:2]
    
    # 정사각형 크기 결정 (짧은 변 기준)
    size = min(h, w)
    
    if crop_mode == 'bottom':
        # 하단 중심 크롭 (도로가 보통 하단에 있음)
        x = (w - size) // 2  # 가로 중앙
        y = h - size  # 하단부터
    else:  # center
        # 중앙 크롭
        x = (w - size) // 2
        y = (h - size) // 2
    
    cropped = frame[y:y+size, x:x+size]
    
    return cropped, (x, y, size, size)


def resize_to_model_input(img, target_size=192):
    """
    정사각형 이미지를 모델 입력 크기로 리사이즈
    """
    return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def process_video_with_crop(model_path, video_path, output_path, conf=0.20,
                             crop_mode='bottom', model_input_size=192):
    """
    비디오를 크롭 후 추론
    
    Args:
        model_path: 모델 파일 경로
        video_path: 입력 비디오 경로
        output_path: 출력 비디오 경로
        conf: confidence threshold
        crop_mode: 'center' 또는 'bottom' (도로 중심)
        model_input_size: 모델 입력 크기 (192)
    """
    print("=" * 70)
    print("🎬 비디오 추론: 종횡비 + 크롭 정리")
    print("=" * 70)
    print(f"모델: {model_path}")
    print(f"입력 비디오: {video_path}")
    print(f"크롭 모드: {crop_mode} (도로 중심)")
    print(f"크롭: 720×720 정사각형")
    print(f"모델 입력: {model_input_size}×{model_input_size}")
    print(f"Confidence: {conf}")
    print("=" * 70)
    
    # 모델 로드
    model = YOLO(model_path)
    
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # 비디오 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 비디오 정보:")
    print(f"   해상도: {width}×{height}")
    print(f"   FPS: {fps}")
    print(f"   총 프레임: {total_frames}")
    
    # 크롭 크기 결정
    crop_size = min(width, height)
    print(f"\n✂️  크롭 설정:")
    print(f"   크롭 크기: {crop_size}×{crop_size}")
    if crop_mode == 'bottom':
        print(f"   위치: 하단 중심 (도로 영역 포함)")
    else:
        print(f"   위치: 중앙")
    
    # 출력 비디오 설정 (원본 해상도 유지)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    frames_with_detections = 0
    
    print(f"\n🔄 프레임 처리 중...")
    
    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1단계: 정사각형 크롭 (720×720)
            cropped_frame, crop_info = crop_to_square(frame, crop_mode)
            
            # 2단계: 모델 입력 크기로 리사이즈 (192×192)
            model_input = resize_to_model_input(cropped_frame, model_input_size)
            
            # 3단계: 추론
            results = model.predict(
                source=model_input,
                conf=conf,
                verbose=False,
                imgsz=model_input_size
            )
            
            result = results[0]
            
            # 4단계: 결과를 원본 프레임에 그리기
            if result.boxes is not None and len(result.boxes) > 0:
                frames_with_detections += 1
                
                for box in result.boxes:
                    # 192×192 좌표를 크롭된 720×720 좌표로 변환
                    x1_192, y1_192, x2_192, y2_192 = box.xyxy[0].cpu().numpy()
                    
                    # 192 → 720 스케일링
                    scale = crop_size / model_input_size
                    x1_crop = int(x1_192 * scale)
                    y1_crop = int(y1_192 * scale)
                    x2_crop = int(x2_192 * scale)
                    y2_crop = int(y2_192 * scale)
                    
                    # 크롭된 좌표를 원본 프레임 좌표로 변환
                    crop_x, crop_y, _, _ = crop_info
                    x1_orig = x1_crop + crop_x
                    y1_orig = y1_crop + crop_y
                    x2_orig = x2_crop + crop_x
                    y2_orig = y2_crop + crop_y
                    
                    # 클래스 및 신뢰도
                    cls = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    class_name = model.names[cls]
                    
                    # 바운딩 박스 그리기
                    color = (0, 255, 0) if class_name == 'vehicle' else (255, 0, 0)
                    cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)
                    
                    # 라벨
                    label = f"{class_name} {conf_score:.2f}"
                    cv2.putText(frame, label, (x1_orig, y1_orig - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                total_detections += len(result.boxes)
            
            # 크롭 영역 표시 (선택사항, 디버깅용)
            # crop_x, crop_y, crop_w, crop_h = crop_info
            # cv2.rectangle(frame, (crop_x, crop_y), (crop_x+crop_w, crop_y+crop_h), (255, 255, 0), 1)
            
            # 프레임 저장
            out.write(frame)
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    
    print(f"\n✅ 완료!")
    print(f"   처리된 프레임: {frame_count}")
    print(f"   탐지된 프레임: {frames_with_detections} ({frames_with_detections/frame_count*100:.1f}%)")
    print(f"   총 탐지 수: {total_detections}")
    print(f"   평균 탐지/프레임: {total_detections/frame_count:.2f}")
    print(f"   탐지된 프레임당 평균: {total_detections/max(frames_with_detections, 1):.2f}")
    print(f"   출력 비디오: {output_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='비디오를 크롭 후 추론 (종횡비 정리)'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='모델 파일 경로'
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='입력 비디오 파일 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='출력 비디오 경로 (기본: 입력 비디오명_cropped.mp4)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.20,
        help='Confidence threshold (기본: 0.20)'
    )
    parser.add_argument(
        '--crop_mode',
        type=str,
        default='bottom',
        choices=['center', 'bottom'],
        help='크롭 모드: center (중앙) 또는 bottom (하단, 도로 중심) (기본: bottom)'
    )
    parser.add_argument(
        '--model_input_size',
        type=int,
        default=192,
        help='모델 입력 크기 (기본: 192)'
    )
    
    args = parser.parse_args()
    
    # 출력 경로 설정
    if args.output is None:
        video_name = Path(args.video).stem
        output_dir = Path(args.video).parent / "inference_results"
        output_dir.mkdir(exist_ok=True)
        args.output = str(output_dir / f"{video_name}_cropped.mp4")
    
    process_video_with_crop(
        args.model,
        args.video,
        args.output,
        args.conf,
        args.crop_mode,
        args.model_input_size
    )


if __name__ == "__main__":
    main()

