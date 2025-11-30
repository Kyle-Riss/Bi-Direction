"""
표지판/신호등 자동 라벨링

YOLOv8 사전 학습 모델을 사용하여 표지판/신호등을 탐지하고
기존 라벨 파일에 추가
"""
import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm


def load_existing_labels(label_path):
    """기존 라벨 파일 로드"""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    labels.append((class_id, x_center, y_center, width, height))
    return labels


def save_labels(label_path, labels):
    """라벨 파일 저장"""
    with open(label_path, 'w') as f:
        for class_id, x_center, y_center, width, height in labels:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def detect_traffic_objects(model, image_path, conf_threshold=0.25):
    """
    이미지에서 표지판/신호등 탐지
    
    Returns:
        list of (class_id, x_center, y_center, width, height)
        class_id: 2 (traffic_sign) or 3 (traffic_light)
    """
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        verbose=False
    )
    
    detections = []
    result = results[0]
    
    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # COCO 클래스에서 표지판/신호등 매핑
            # COCO에는 직접적인 표지판/신호등 클래스가 없으므로
            # 다른 방법 필요
            
            # 일단 모든 탐지를 반환 (나중에 필터링)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # 이미지 크기 로드
            img = cv2.imread(image_path)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            
            # YOLO 형식으로 변환
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h
            
            # COCO 클래스를 우리 클래스로 매핑
            # 주의: COCO에는 표지판/신호등이 없으므로
            # 다른 모델이나 방법 필요
            detections.append((cls, x_center, y_center, width, height, conf))
    
    return detections


def auto_label_traffic(image_dir, label_dir, output_label_dir=None, 
                       conf_threshold=0.25, model_name='yolov8n.pt'):
    """
    이미지에서 표지판/신호등 자동 탐지하여 라벨 추가
    
    주의: YOLOv8 기본 모델에는 표지판/신호등 클래스가 없으므로
    이 스크립트는 구조만 제공합니다.
    실제로는 표지판/신호등 전용 모델이 필요합니다.
    """
    if output_label_dir is None:
        output_label_dir = label_dir
    
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 모델 로드
    print(f"📦 모델 로드: {model_name}")
    model = YOLO(model_name)
    
    # 이미지 파일 목록
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\n📁 이미지 파일: {len(image_files)}개")
    print(f"📁 라벨 디렉토리: {label_dir}")
    print(f"📁 출력 디렉토리: {output_label_dir}")
    print(f"🔍 Confidence threshold: {conf_threshold}")
    print("\n⚠️  주의: YOLOv8 기본 모델에는 표지판/신호등 클래스가 없습니다!")
    print("   이 스크립트는 구조만 제공하며, 실제 사용 시 전용 모델이 필요합니다.")
    print("=" * 70)
    
    total_added = 0
    total_skipped = 0
    
    for image_file in tqdm(image_files, desc="라벨링 중"):
        image_path = os.path.join(image_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        output_path = os.path.join(output_label_dir, label_file)
        
        # 기존 라벨 로드
        existing_labels = load_existing_labels(label_path)
        
        # 표지판/신호등 탐지 (현재는 구조만)
        # TODO: 표지판/신호등 전용 모델 사용
        traffic_detections = []  # detect_traffic_objects(model, image_path, conf_threshold)
        
        # 기존 라벨 + 새로운 표지판/신호등 라벨 결합
        all_labels = existing_labels.copy()
        
        for cls, x_center, y_center, width, height, conf in traffic_detections:
            # 클래스 매핑: COCO → 우리 클래스
            # 2: traffic_sign, 3: traffic_light
            if cls in [2, 3]:  # 예시 (실제 매핑 필요)
                yolo_class = cls
                all_labels.append((yolo_class, x_center, y_center, width, height))
                total_added += 1
        
        # 라벨 저장
        save_labels(output_path, all_labels)
        
        if not traffic_detections:
            total_skipped += 1
    
    print(f"\n✅ 완료!")
    print(f"   추가된 표지판/신호등: {total_added}개")
    print(f"   스킵된 이미지: {total_skipped}개")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='표지판/신호등 자동 라벨링'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help='이미지 디렉토리'
    )
    parser.add_argument(
        '--label_dir',
        type=str,
        required=True,
        help='기존 라벨 디렉토리'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='출력 라벨 디렉토리 (기본: label_dir와 동일)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (기본: 0.25)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='모델 파일 (기본: yolov8n.pt)'
    )
    
    args = parser.parse_args()
    
    auto_label_traffic(
        args.image_dir,
        args.label_dir,
        args.output_dir,
        args.conf,
        args.model
    )


if __name__ == "__main__":
    main()

