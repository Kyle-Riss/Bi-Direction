"""
표지판/신호등 자동 라벨링

방법 1: YOLOv8 기본 모델로 탐지 후 필터링 (제한적)
방법 2: 표지판/신호등 전용 모델 사용 (권장)
방법 3: COCO 모델 + 추가 필터링
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


def detect_with_custom_model(model, image_path, conf_threshold=0.25):
    """
    커스텀 모델로 표지판/신호등 탐지
    
    Returns:
        list of (yolo_class, x_center, y_center, width, height, confidence)
        yolo_class: 2 (traffic_sign) or 3 (traffic_light)
    """
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        verbose=False
    )
    
    detections = []
    result = results[0]
    
    if result.boxes is not None:
        img = cv2.imread(image_path)
        if img is None:
            return detections
        img_h, img_w = img.shape[:2]
        
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            
            # 클래스 이름 기반 필터링
            yolo_class = None
            if 'sign' in class_name.lower() or 'traffic_sign' in class_name.lower():
                yolo_class = 2  # traffic_sign
            elif 'light' in class_name.lower() or 'traffic_light' in class_name.lower() or 'signal' in class_name.lower():
                yolo_class = 3  # traffic_light
            
            if yolo_class is not None:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # YOLO 형식으로 변환
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                
                detections.append((yolo_class, x_center, y_center, width, height, conf))
    
    return detections


def detect_with_heuristic(image_path, conf_threshold=0.25):
    """
    휴리스틱 방법: 색상 기반 표지판/신호등 탐지
    
    이 방법은 제한적이지만, 기본적인 탐지는 가능합니다.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    img_h, img_w = img.shape[:2]
    detections = []
    
    # HSV로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 빨간색 영역 (신호등, 정지 표지판)
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # 노란색 영역 (경고 표지판)
    yellow_lower = np.array([20, 50, 50])
    yellow_upper = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    # 초록색 영역 (신호등)
    green_lower = np.array([50, 50, 50])
    green_upper = np.array([70, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # 컨투어 찾기
    for mask, yolo_class in [(red_mask, 3), (yellow_mask, 2), (green_mask, 3)]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # 너무 작은 영역 제외
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # YOLO 형식으로 변환
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            width = w / img_w
            height = h / img_h
            
            # 신뢰도는 낮게 설정 (휴리스틱이므로)
            conf = 0.3
            detections.append((yolo_class, x_center, y_center, width, height, conf))
    
    return detections


def auto_label_traffic(image_dir, label_dir, output_label_dir=None, 
                       conf_threshold=0.25, method='heuristic', model_path=None):
    """
    표지판/신호등 자동 라벨링
    
    Args:
        image_dir: 이미지 디렉토리
        label_dir: 기존 라벨 디렉토리
        output_label_dir: 출력 라벨 디렉토리
        conf_threshold: confidence threshold
        method: 'heuristic' (색상 기반) 또는 'model' (모델 기반)
        model_path: 모델 파일 경로 (method='model'일 때)
    """
    if output_label_dir is None:
        output_label_dir = label_dir
    
    os.makedirs(output_label_dir, exist_ok=True)
    
    print("=" * 70)
    print("🚦 표지판/신호등 자동 라벨링")
    print("=" * 70)
    print(f"이미지 디렉토리: {image_dir}")
    print(f"라벨 디렉토리: {label_dir}")
    print(f"출력 디렉토리: {output_label_dir}")
    print(f"방법: {method}")
    print(f"Confidence threshold: {conf_threshold}")
    print("=" * 70)
    
    # 모델 로드 (method='model'일 때)
    model = None
    if method == 'model':
        if model_path is None:
            print("⚠️  모델 경로가 지정되지 않았습니다. 휴리스틱 방법을 사용합니다.")
            method = 'heuristic'
        else:
            if os.path.exists(model_path):
                print(f"📦 모델 로드: {model_path}")
                model = YOLO(model_path)
            else:
                print(f"⚠️  모델 파일을 찾을 수 없습니다: {model_path}")
                print("   휴리스틱 방법을 사용합니다.")
                method = 'heuristic'
    
    # 이미지 파일 목록
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\n📁 이미지 파일: {len(image_files)}개")
    print(f"🔄 라벨링 시작...\n")
    
    total_added_signs = 0
    total_added_lights = 0
    total_processed = 0
    
    for image_file in tqdm(image_files, desc="라벨링"):
        image_path = os.path.join(image_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        output_path = os.path.join(output_label_dir, label_file)
        
        # 기존 라벨 로드
        existing_labels = load_existing_labels(label_path)
        
        # 표지판/신호등 탐지
        if method == 'model' and model is not None:
            traffic_detections = detect_with_custom_model(model, image_path, conf_threshold)
        else:
            traffic_detections = detect_with_heuristic(image_path, conf_threshold)
        
        # 기존 라벨 + 새로운 표지판/신호등 라벨 결합
        all_labels = existing_labels.copy()
        
        for yolo_class, x_center, y_center, width, height, conf in traffic_detections:
            # 중복 제거: 기존 라벨과 IoU가 높으면 제외
            is_duplicate = False
            for existing_class, ex_x, ex_y, ex_w, ex_h in existing_labels:
                # 간단한 IoU 계산
                iou = calculate_iou(
                    (x_center, y_center, width, height),
                    (ex_x, ex_y, ex_w, ex_h)
                )
                if iou > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_labels.append((yolo_class, x_center, y_center, width, height))
                if yolo_class == 2:
                    total_added_signs += 1
                elif yolo_class == 3:
                    total_added_lights += 1
        
        # 라벨 저장
        save_labels(output_path, all_labels)
        total_processed += 1
    
    print(f"\n✅ 완료!")
    print(f"   처리된 이미지: {total_processed}개")
    print(f"   추가된 표지판: {total_added_signs}개")
    print(f"   추가된 신호등: {total_added_lights}개")
    print(f"   총 추가: {total_added_signs + total_added_lights}개")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("   라벨 파일을 확인하고 필요시 수동으로 수정하세요.")
    print("   그 다음 추가 학습을 진행하세요:")
    print("   python tools/train_mixed_with_traffic.py")
    print("=" * 70)


def calculate_iou(box1, box2):
    """두 바운딩 박스의 IoU 계산"""
    x1_center, y1_center, w1, h1 = box1
    x2_center, y2_center, w2, h2 = box2
    
    # 좌표 변환
    x1_min = x1_center - w1 / 2
    x1_max = x1_center + w1 / 2
    y1_min = y1_center - h1 / 2
    y1_max = y1_center + h1 / 2
    
    x2_min = x2_center - w2 / 2
    x2_max = x2_center + w2 / 2
    y2_min = y2_center - h2 / 2
    y2_max = y2_center + h2 / 2
    
    # 교집합
    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


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
        '--method',
        type=str,
        default='heuristic',
        choices=['heuristic', 'model'],
        help='라벨링 방법: heuristic (색상 기반) 또는 model (모델 기반)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='모델 파일 경로 (method=model일 때 필요)'
    )
    
    args = parser.parse_args()
    
    auto_label_traffic(
        args.image_dir,
        args.label_dir,
        args.output_dir,
        args.conf,
        args.method,
        args.model
    )


if __name__ == "__main__":
    main()

