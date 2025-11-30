"""
표지판/신호등 라벨 필터링

신뢰도, 크기, 위치 기준으로 false positive 제거
"""
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np


def load_labels(label_path):
    """라벨 파일 로드"""
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
                    conf = float(parts[5]) if len(parts) > 5 else 1.0
                    labels.append((class_id, x_center, y_center, width, height, conf))
    return labels


def save_labels(label_path, labels):
    """라벨 파일 저장 (confidence 제외)"""
    with open(label_path, 'w') as f:
        for class_id, x_center, y_center, width, height, _ in labels:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def calculate_iou(box1, box2):
    """두 바운딩 박스의 IoU 계산"""
    x1_center, y1_center, w1, h1 = box1[:4]
    x2_center, y2_center, w2, h2 = box2[:4]
    
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


def filter_labels(labels, 
                  min_conf=0.4,
                  max_conf=1.0,
                  min_size=0.01,  # 최소 크기 (이미지의 1%)
                  max_size=0.3,   # 최대 크기 (이미지의 30%)
                  min_aspect_ratio=0.3,  # 최소 종횡비
                  max_aspect_ratio=3.0,   # 최대 종횡비
                  edge_margin=0.02,  # 경계 마진 (2%)
                  max_iou=0.7,  # 중복 제거 IoU 임계값
                  max_traffic_signs=10,  # 이미지당 최대 표지판 수
                  max_traffic_lights=15):  # 이미지당 최대 신호등 수
    """
    라벨 필터링
    
    Args:
        labels: [(class_id, x_center, y_center, width, height, conf), ...]
        min_conf: 최소 신뢰도
        max_conf: 최대 신뢰도
        min_size: 최소 박스 크기 (정규화)
        max_size: 최대 박스 크기 (정규화)
        min_aspect_ratio: 최소 종횡비
        max_aspect_ratio: 최대 종횡비
        edge_margin: 경계 마진 (경계 근처 박스 제거)
        max_iou: 중복 제거 IoU 임계값
        max_traffic_signs: 이미지당 최대 표지판 수
        max_traffic_lights: 이미지당 최대 신호등 수
    """
    filtered = []
    
    # 1단계: 신뢰도 필터링
    for label in labels:
        class_id, x_center, y_center, width, height, conf = label
        
        # 신뢰도 체크
        if conf < min_conf or conf > max_conf:
            continue
        
        # 크기 체크
        box_area = width * height
        if box_area < min_size or box_area > max_size:
            continue
        
        # 종횡비 체크
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
        
        # 경계 체크 (경계 근처 박스 제거)
        if (x_center - width/2 < edge_margin or 
            x_center + width/2 > 1 - edge_margin or
            y_center - height/2 < edge_margin or
            y_center + height/2 > 1 - edge_margin):
            continue
        
        filtered.append(label)
    
    # 2단계: 클래스별로 분리
    vehicle_labels = [l for l in filtered if l[0] == 0]
    pedestrian_labels = [l for l in filtered if l[0] == 1]
    traffic_sign_labels = [l for l in filtered if l[0] == 2]
    traffic_light_labels = [l for l in filtered if l[0] == 3]
    
    # 3단계: 표지판/신호등 신뢰도 순으로 정렬 후 상위 N개만 선택
    traffic_sign_labels.sort(key=lambda x: x[5], reverse=True)
    traffic_light_labels.sort(key=lambda x: x[5], reverse=True)
    
    traffic_sign_labels = traffic_sign_labels[:max_traffic_signs]
    traffic_light_labels = traffic_light_labels[:max_traffic_lights]
    
    # 4단계: 중복 제거 (NMS 스타일)
    def remove_duplicates(label_list, iou_threshold):
        if len(label_list) == 0:
            return []
        
        # 신뢰도 순으로 정렬
        label_list.sort(key=lambda x: x[5], reverse=True)
        kept = []
        
        for label in label_list:
            is_duplicate = False
            for kept_label in kept:
                iou = calculate_iou(label, kept_label)
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(label)
        
        return kept
    
    traffic_sign_labels = remove_duplicates(traffic_sign_labels, max_iou)
    traffic_light_labels = remove_duplicates(traffic_light_labels, max_iou)
    
    # 5단계: 모든 라벨 합치기
    final_labels = vehicle_labels + pedestrian_labels + traffic_sign_labels + traffic_light_labels
    
    return final_labels


def filter_label_files(input_label_dir, output_label_dir, **filter_kwargs):
    """
    라벨 파일들 필터링
    
    Args:
        input_label_dir: 입력 라벨 디렉토리
        output_label_dir: 출력 라벨 디렉토리
        **filter_kwargs: 필터링 파라미터
    """
    os.makedirs(output_label_dir, exist_ok=True)
    
    label_files = sorted([f for f in os.listdir(input_label_dir) if f.endswith('.txt')])
    
    print("=" * 70)
    print("🔍 표지판/신호등 라벨 필터링")
    print("=" * 70)
    print(f"입력 디렉토리: {input_label_dir}")
    print(f"출력 디렉토리: {output_label_dir}")
    print(f"라벨 파일 수: {len(label_files)}")
    print("\n필터링 기준:")
    print(f"  • 최소 신뢰도: {filter_kwargs.get('min_conf', 0.4)}")
    print(f"  • 최소 크기: {filter_kwargs.get('min_size', 0.01)}")
    print(f"  • 최대 크기: {filter_kwargs.get('max_size', 0.3)}")
    print(f"  • 최대 표지판 수: {filter_kwargs.get('max_traffic_signs', 10)}")
    print(f"  • 최대 신호등 수: {filter_kwargs.get('max_traffic_lights', 15)}")
    print("=" * 70)
    print()
    
    total_before = 0
    total_after = 0
    total_removed = 0
    
    stats = {
        0: {'before': 0, 'after': 0},  # vehicle
        1: {'before': 0, 'after': 0},  # pedestrian
        2: {'before': 0, 'after': 0},  # traffic_sign
        3: {'before': 0, 'after': 0},  # traffic_light
    }
    
    for label_file in tqdm(label_files, desc="필터링"):
        input_path = os.path.join(input_label_dir, label_file)
        output_path = os.path.join(output_label_dir, label_file)
        
        # 라벨 로드
        labels = load_labels(input_path)
        
        # 통계
        for label in labels:
            class_id = label[0]
            if class_id in stats:
                stats[class_id]['before'] += 1
        total_before += len(labels)
        
        # 필터링
        filtered_labels = filter_labels(labels, **filter_kwargs)
        
        # 통계
        for label in filtered_labels:
            class_id = label[0]
            if class_id in stats:
                stats[class_id]['after'] += 1
        total_after += len(filtered_labels)
        total_removed += len(labels) - len(filtered_labels)
        
        # 저장
        save_labels(output_path, filtered_labels)
    
    print("\n" + "=" * 70)
    print("✅ 필터링 완료!")
    print("=" * 70)
    print(f"\n📊 통계:")
    print(f"   총 라벨 수: {total_before:,} → {total_after:,} (제거: {total_removed:,}, {total_removed/total_before*100:.1f}%)")
    print(f"\n   클래스별:")
    class_names = {0: 'vehicle', 1: 'pedestrian', 2: 'traffic_sign', 3: 'traffic_light'}
    for class_id, name in class_names.items():
        before = stats[class_id]['before']
        after = stats[class_id]['after']
        removed = before - after
        pct = (removed / before * 100) if before > 0 else 0
        print(f"   {name}: {before:,} → {after:,} (제거: {removed:,}, {pct:.1f}%)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='표지판/신호등 라벨 필터링'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='입력 라벨 디렉토리'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='출력 라벨 디렉토리'
    )
    parser.add_argument(
        '--min_conf',
        type=float,
        default=0.4,
        help='최소 신뢰도 (기본: 0.4)'
    )
    parser.add_argument(
        '--min_size',
        type=float,
        default=0.01,
        help='최소 박스 크기 (정규화, 기본: 0.01)'
    )
    parser.add_argument(
        '--max_size',
        type=float,
        default=0.3,
        help='최대 박스 크기 (정규화, 기본: 0.3)'
    )
    parser.add_argument(
        '--max_signs',
        type=int,
        default=10,
        help='이미지당 최대 표지판 수 (기본: 10)'
    )
    parser.add_argument(
        '--max_lights',
        type=int,
        default=15,
        help='이미지당 최대 신호등 수 (기본: 15)'
    )
    parser.add_argument(
        '--iou_threshold',
        type=float,
        default=0.7,
        help='중복 제거 IoU 임계값 (기본: 0.7)'
    )
    
    args = parser.parse_args()
    
    filter_kwargs = {
        'min_conf': args.min_conf,
        'min_size': args.min_size,
        'max_size': args.max_size,
        'max_traffic_signs': args.max_signs,
        'max_traffic_lights': args.max_lights,
        'max_iou': args.iou_threshold,
    }
    
    filter_label_files(args.input_dir, args.output_dir, **filter_kwargs)


if __name__ == "__main__":
    main()

