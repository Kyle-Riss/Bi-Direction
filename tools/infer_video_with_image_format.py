"""
비디오 추론 시 프레임을 학습 이미지 형식(1280x1280)으로 변환 후 추론

비디오 프레임 → 1280x1280 변환 → 모델 입력(192x192) → 추론
이렇게 하면 학습 이미지와 동일한 전처리 파이프라인을 거침
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import os
from tqdm import tqdm


def resize_with_letterbox(img, target_size=1280):
    """
    종횡비를 유지하면서 정사각형으로 리사이즈 (letterbox padding)
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded


def process_video_with_image_format(model_path, video_path, output_path, conf=0.20, 
                                     intermediate_size=1280, model_input_size=192):
    """
    비디오를 학습 이미지 형식으로 변환하면서 추론
    
    Args:
        model_path: 모델 파일 경로
        video_path: 입력 비디오 경로
        output_path: 출력 비디오 경로
        conf: confidence threshold
        intermediate_size: 중간 변환 크기 (1280x1280, 학습 이미지와 동일)
        model_input_size: 모델 입력 크기 (192x192)
    """
    print("=" * 70)
    print("🎬 비디오 추론 (학습 이미지 형식으로 전처리)")
    print("=" * 70)
    print(f"모델: {model_path}")
    print(f"입력 비디오: {video_path}")
    print(f"중간 변환: {intermediate_size}x{intermediate_size} (학습 이미지 형식)")
    print(f"모델 입력: {model_input_size}x{model_input_size}")
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
    print(f"   해상도: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   총 프레임: {total_frames}")
    
    # 출력 비디오 설정 (원본 해상도 유지)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    
    print(f"\n🔄 프레임 처리 중...")
    
    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1단계: 1280x1280으로 변환 (학습 이미지 형식)
            frame_1280 = resize_with_letterbox(frame, intermediate_size)
            
            # 2단계: 모델 입력 크기로 리사이즈 (letterbox 유지)
            frame_192 = resize_with_letterbox(frame_1280, model_input_size)
            
            # 3단계: 추론
            results = model.predict(
                source=frame_192,
                conf=conf,
                verbose=False,
                imgsz=model_input_size
            )
            
            result = results[0]
            
            # 4단계: 결과를 원본 프레임에 그리기
            if result.boxes is not None and len(result.boxes) > 0:
                # 바운딩 박스 그리기
                for box in result.boxes:
                    # 192x192 좌표를 원본 해상도로 변환
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    class_name = model.names[cls]
                    
                    # 좌표 스케일링 (192x192 → 1280x1280 → 원본 해상도)
                    scale_to_1280 = intermediate_size / model_input_size
                    scale_to_original = width / intermediate_size  # 가로 기준
                    
                    x1 = int(x1 * scale_to_1280 * scale_to_original)
                    y1 = int(y1 * scale_to_1280 * scale_to_original)
                    x2 = int(x2 * scale_to_1280 * scale_to_original)
                    y2 = int(y2 * scale_to_1280 * scale_to_original)
                    
                    # 바운딩 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf_score:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                total_detections += len(result.boxes)
            
            # 프레임 저장
            out.write(frame)
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    out.release()
    
    print(f"\n✅ 완료!")
    print(f"   처리된 프레임: {frame_count}")
    print(f"   총 탐지 수: {total_detections}")
    print(f"   평균 탐지/프레임: {total_detections/frame_count:.2f}")
    print(f"   출력 비디오: {output_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='비디오를 학습 이미지 형식으로 변환하면서 추론'
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
        help='출력 비디오 경로 (기본: 입력 비디오명_inferred.mp4)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.20,
        help='Confidence threshold (기본: 0.20)'
    )
    parser.add_argument(
        '--intermediate_size',
        type=int,
        default=1280,
        help='중간 변환 크기 (기본: 1280, 학습 이미지와 동일)'
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
        args.output = str(output_dir / f"{video_name}_inferred.mp4")
    
    process_video_with_image_format(
        args.model,
        args.video,
        args.output,
        args.conf,
        args.intermediate_size,
        args.model_input_size
    )


if __name__ == "__main__":
    main()

