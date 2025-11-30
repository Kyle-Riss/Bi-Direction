"""
YOLO를 사용하여 이미지에서 바운딩 박스 자동 검출 및 라벨 생성

test 폴더의 이미지들에 대해 YOLOv8로 객체 검출을 수행하고,
YOLO 형식의 라벨 파일(.txt)을 생성합니다.
"""
import os
import glob
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from ultralytics import YOLO
import argparse


def map_coco_to_custom_class(coco_class_id):
    """
    COCO 데이터셋 클래스 ID를 우리의 커스텀 클래스로 매핑
    
    COCO 클래스:
    0: person (pedestrian)
    1: bicycle
    2: car
    3: motorcycle
    5: bus
    7: truck
    
    우리 클래스:
    0: pedestrian
    1: car
    2: truck_bus
    3: bicycle_motorcycle
    4: traffic_sign
    """
    mapping = {
        0: 0,   # person -> pedestrian
        1: 3,   # bicycle -> bicycle_motorcycle
        2: 1,   # car -> car
        3: 3,   # motorcycle -> bicycle_motorcycle
        5: 2,   # bus -> truck_bus
        7: 2,   # truck -> truck_bus
    }
    return mapping.get(coco_class_id, None)


def convert_bbox_to_yolo_format(bbox, img_width, img_height):
    """
    Ultralytics YOLO 결과 바운딩 박스를 YOLO 형식으로 변환
    
    Args:
        bbox: [x1, y1, x2, y2] (픽셀 좌표)
        img_width: 이미지 너비
        img_height: 이미지 높이
    
    Returns:
        (x_center_norm, y_center_norm, width_norm, height_norm)
    """
    x1, y1, x2, y2 = bbox
    
    # 중심 좌표 계산
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    
    # 너비, 높이 계산
    width = x2 - x1
    height = y2 - y1
    
    # 정규화 (0-1 범위)
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    # 범위 검증 및 클리핑
    x_center_norm = max(0, min(1, x_center_norm))
    y_center_norm = max(0, min(1, y_center_norm))
    width_norm = max(0.001, min(1, width_norm))
    height_norm = max(0.001, min(1, height_norm))
    
    return x_center_norm, y_center_norm, width_norm, height_norm


def draw_bbox_on_image(image_path, detections, class_names, output_path=None):
    """
    이미지에 바운딩 박스 그리기
    
    Args:
        image_path: 원본 이미지 경로
        detections: [(class_id, x_center_norm, y_center_norm, width_norm, height_norm, conf), ...]
        class_names: 클래스 이름 리스트
        output_path: 저장할 경로 (None이면 원본 이미지와 같은 위치에 저장)
    """
    # 이미지 로드
    img = Image.open(image_path)
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)
    
    # 색상 정의 (클래스별)
    colors = {
        0: (255, 0, 0),      # pedestrian - 빨강
        1: (0, 255, 0),      # car - 초록
        2: (0, 0, 255),      # truck_bus - 파랑
        3: (255, 255, 0),    # bicycle_motorcycle - 노랑
        4: (255, 0, 255),    # traffic_sign - 마젠타
    }
    
    # 폰트 (시스템 기본 폰트 사용)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    # 바운딩 박스 그리기
    for detection in detections:
        if len(detection) == 5:
            # conf 값이 없는 경우 기본값 1.0 사용
            class_id, x_center_norm, y_center_norm, width_norm, height_norm = detection
            conf = 1.0
        else:
            # conf 값이 있는 경우
            class_id, x_center_norm, y_center_norm, width_norm, height_norm, conf = detection
        # 정규화된 좌표를 픽셀 좌표로 변환
        x_center = x_center_norm * img_width
        y_center = y_center_norm * img_height
        width = width_norm * img_width
        height = height_norm * img_height
        
        # 왼쪽 위, 오른쪽 아래 좌표 계산
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # 색상 가져오기
        color = colors.get(class_id, (255, 255, 255))
        
        # 바운딩 박스 그리기
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 클래스 이름 및 신뢰도 표시
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        text = f"{class_name} {conf:.2f}"
        
        # 텍스트 배경 (가독성 향상)
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_bg = [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2]
        draw.rectangle(text_bg, fill=color)
        draw.text((x1, y1), text, fill=(0, 0, 0), font=font)
    
    # 저장
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(os.path.dirname(image_path), f"{base_name}_detected.jpg")
    
    img.save(output_path, quality=95)
    return output_path


def detect_and_save_labels(model_path, image_dir, output_label_dir, conf_threshold=0.25, iou_threshold=0.45, save_visualization=True):
    """
    YOLO 모델을 사용하여 이미지에서 객체 검출하고 YOLO 형식 라벨 파일 생성
    
    Args:
        model_path: YOLO 모델 경로 (예: 'yolov8n.pt')
        image_dir: 이미지 디렉토리
        output_label_dir: 라벨 파일 출력 디렉토리
        conf_threshold: 신뢰도 임계값 (기본: 0.25)
        iou_threshold: IoU 임계값 (기본: 0.45)
        save_visualization: 시각화 이미지 저장 여부 (기본: True)
    """
    # 출력 디렉토리 생성
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 시각화 이미지 저장 디렉토리
    if save_visualization:
        vis_dir = os.path.join(os.path.dirname(output_label_dir), 'visualized')
        os.makedirs(vis_dir, exist_ok=True)
    
    # 클래스 이름 정의
    class_names = ['pedestrian', 'car', 'truck_bus', 'bicycle_motorcycle', 'traffic_sign']
    
    # YOLO 모델 로드
    print(f"📦 Loading YOLO model from {model_path}...")
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # 이미지 파일 목록 가져오기
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')) + 
                        glob.glob(os.path.join(image_dir, '*.png')) +
                        glob.glob(os.path.join(image_dir, '*.jpeg')))
    
    print(f"\n🔍 Detecting objects in {len(image_files)} images...")
    print(f"   Confidence threshold: {conf_threshold}")
    print(f"   IoU threshold: {iou_threshold}")
    
    total_detections = 0
    processed_count = 0
    skipped_count = 0
    
    for img_path in image_files:
        try:
            # 이미지 로드하여 크기 확인
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # YOLO 추론 수행 (MPS 대신 CPU 사용 - 안정성)
            results = model.predict(
                source=img_path,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
                device='cpu'  # MPS는 일부 환경에서 문제가 있을 수 있음
            )
            
            # 결과 파싱
            result = results[0]  # 첫 번째 결과만 사용 (단일 이미지)
            
            # 라벨 파일 생성
            img_basename = os.path.basename(img_path)
            label_filename = os.path.splitext(img_basename)[0] + '.txt'
            label_path = os.path.join(output_label_dir, label_filename)
            
            detections_count = 0
            detections_list = []  # 시각화를 위한 리스트
            
            with open(label_path, 'w') as f:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # COCO 클래스 ID 가져오기
                        coco_class_id = int(box.cls[0])
                        
                        # 커스텀 클래스로 매핑
                        custom_class_id = map_coco_to_custom_class(coco_class_id)
                        
                        if custom_class_id is None:
                            continue  # 매핑되지 않은 클래스는 스킵
                        
                        # 바운딩 박스 좌표 가져오기 (xyxy 형식)
                        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                        
                        # YOLO 형식으로 변환
                        x_center, y_center, width, height = convert_bbox_to_yolo_format(
                            bbox, img_width, img_height
                        )
                        
                        # 신뢰도 가져오기
                        conf = float(box.conf[0])
                        
                        # 라벨 파일에 저장
                        f.write(f"{custom_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        detections_list.append((custom_class_id, x_center, y_center, width, height, conf))
                        detections_count += 1
                        total_detections += 1
            
            # 시각화 이미지 저장
            if save_visualization and detections_count > 0:
                vis_path = os.path.join(vis_dir, f"{os.path.splitext(img_basename)[0]}_detected.jpg")
                draw_bbox_on_image(img_path, detections_list, class_names, vis_path)
            
            if detections_count > 0:
                processed_count += 1
            else:
                # 검출된 객체가 없으면 빈 라벨 파일 생성 (또는 생성 안 함)
                if os.path.exists(label_path) and os.path.getsize(label_path) == 0:
                    os.remove(label_path)
                    skipped_count += 1
                else:
                    processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"   Processed {processed_count}/{len(image_files)} images...")
        
        except Exception as e:
            print(f"⚠️  Error processing {img_path}: {e}")
            skipped_count += 1
            continue
    
    print(f"\n✅ Detection completed!")
    print(f"   Total images: {len(image_files)}")
    print(f"   Processed: {processed_count}")
    print(f"   Skipped: {skipped_count}")
    print(f"   Total detections: {total_detections}")
    print(f"   Average detections per image: {total_detections/processed_count if processed_count > 0 else 0:.2f}")
    print(f"\n📁 Labels saved to: {output_label_dir}")
    if save_visualization:
        print(f"📸 Visualization images saved to: {vis_dir}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Detect objects using YOLO and generate YOLO format labels')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Path to YOLO model file (default: yolov8n.pt)')
    parser.add_argument('--image_dir', type=str, default='test',
                       help='Directory containing images (default: test)')
    parser.add_argument('--output_dir', type=str, default='test/labels',
                       help='Output directory for label files (default: test/labels)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--no-vis', action='store_true',
                       help='Disable visualization image saving')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found at {args.model}")
        print("   Available models: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt")
        print("   Downloading yolov8n.pt if not found...")
        return
    
    if not os.path.exists(args.image_dir):
        print(f"❌ Error: Image directory not found at {args.image_dir}")
        return
    
    detect_and_save_labels(
        model_path=args.model,
        image_dir=args.image_dir,
        output_label_dir=args.output_dir,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        save_visualization=not args.no_vis
    )


if __name__ == '__main__':
    main()

