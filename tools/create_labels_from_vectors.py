"""
vectors.tsv를 사용하여 YOLO 형식 라벨 파일 생성

vectors.tsv가 중심 좌표 (x_center, y_center)라고 가정하고,
클래스별 기본 width/height를 사용하여 바운딩 박스 생성

주의: 이는 임시 해결책입니다. 정확한 바운딩 박스 정보는 데이터 담당자에게 확인 필요
"""
import os
import csv
import shutil
from collections import defaultdict
from PIL import Image


def parse_metadata(metadata_path):
    """metadata.tsv 파싱"""
    image_labels = defaultdict(list)
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for idx, row in enumerate(reader):
            try:
                label = int(row['label'])
                meta = row['meta']
                
                # meta에서 이미지명 추출
                parts = meta.split('-')
                if len(parts) >= 3:
                    image_name = '-'.join(parts[:-1]) + '.jpg'
                elif len(parts) == 2:
                    if parts[1].endswith('.jpg'):
                        image_name = parts[0] + '-' + parts[1]
                    elif parts[0].endswith('.jpg'):
                        image_name = parts[0]
                    else:
                        image_name = parts[0] + '.jpg'
                else:
                    image_name = parts[0]
                    if not image_name.endswith('.jpg'):
                        image_name = parts[0] + '.jpg'
                
                image_labels[image_name].append((label, idx))
            except Exception as e:
                print(f"⚠️  Warning: Error parsing row {idx}: {e}")
                continue
    
    return image_labels


def parse_vectors(vectors_path):
    """vectors.tsv 파싱"""
    vectors = []
    with open(vectors_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                try:
                    x = float(row[0])
                    y = float(row[1])
                    vectors.append((x, y))
                except ValueError:
                    continue
    return vectors


def map_label_to_yolo_class(label):
    """metadata.tsv의 label을 YOLO 클래스로 매핑"""
    label_mapping = {
        0: 0,  # pedestrian
        1: 1,  # car
        2: 2,  # truck_bus
        3: 3,  # bicycle_motorcycle
        4: 4,  # traffic_sign
        5: 4,  # traffic_sign으로 가정 (확인 필요)
        6: 4,  # traffic_sign으로 가정 (확인 필요)
        7: 4,  # traffic_sign으로 가정 (확인 필요)
    }
    return label_mapping.get(label, None)


def get_default_bbox_size(class_id, img_width, img_height):
    """
    클래스별 기본 바운딩 박스 크기 반환 (정규화)
    
    참고: 이는 추정값이며, 실제 객체 크기에 맞게 조정 필요
    """
    # 클래스별 평균 크기 (정규화된 값, 이미지 크기 기준)
    default_sizes = {
        0: (0.05, 0.15),   # pedestrian: 작고 세로로 긴 박스
        1: (0.20, 0.15),   # car: 가로로 긴 박스
        2: (0.25, 0.20),   # truck_bus: 큰 박스
        3: (0.10, 0.15),   # bicycle_motorcycle: 중간 크기
        4: (0.08, 0.10),   # traffic_sign: 작은 박스
    }
    
    width_norm, height_norm = default_sizes.get(class_id, (0.15, 0.15))
    return width_norm, height_norm


def create_yolo_labels_from_vectors(metadata_path, vectors_path, image_dir, output_label_dir, 
                                    img_width=1280, img_height=720):
    """
    vectors.tsv의 좌표를 사용하여 YOLO 형식 라벨 파일 생성
    
    Args:
        metadata_path: metadata.tsv 경로
        vectors_path: vectors.tsv 경로
        image_dir: 이미지 디렉토리
        output_label_dir: 라벨 파일 출력 디렉토리
        img_width: 이미지 너비 (픽셀)
        img_height: 이미지 높이 (픽셀)
    """
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 데이터 파싱
    print("📖 Parsing metadata.tsv...")
    image_labels = parse_metadata(metadata_path)
    print(f"   Found {len(image_labels)} images with labels")
    
    print("📖 Parsing vectors.tsv...")
    vectors = parse_vectors(vectors_path)
    print(f"   Found {len(vectors)} vectors")
    
    # 이미지별로 객체 그룹화
    image_to_objects = defaultdict(list)
    for image_name, label_indices in image_labels.items():
        for label, obj_idx in label_indices:
            if obj_idx < len(vectors):
                x_pixel, y_pixel = vectors[obj_idx]
                yolo_class = map_label_to_yolo_class(label)
                if yolo_class is not None:
                    image_to_objects[image_name].append((yolo_class, label, x_pixel, y_pixel, obj_idx))
    
    print(f"\n🔄 Creating YOLO label files...")
    converted_count = 0
    skipped_count = 0
    error_count = 0
    
    for image_name, objects in image_to_objects.items():
        image_path = os.path.join(image_dir, image_name)
        
        # 실제 이미지 파일 확인
        actual_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        matched_image = None
        
        # 정확한 매칭
        if image_name in actual_images:
            matched_image = image_name
        else:
            # prefix 매칭 시도
            img_prefix = image_name.replace('.jpg', '').split('-')[0]
            matched = [f for f in actual_images if f.startswith(img_prefix)]
            if matched:
                matched_image = matched[0]
        
        if not matched_image:
            skipped_count += 1
            continue
        
        # 이미지 크기 가져오기
        try:
            img_path = os.path.join(image_dir, matched_image)
            img = Image.open(img_path)
            actual_width, actual_height = img.size
        except Exception as e:
            print(f"⚠️  Warning: Could not open image {matched_image}: {e}")
            skipped_count += 1
            continue
        
        # YOLO 라벨 파일 생성
        label_file = os.path.join(output_label_dir, matched_image.replace('.jpg', '.txt'))
        
        try:
            with open(label_file, 'w') as f:
                for yolo_class, orig_label, x_pixel, y_pixel, obj_idx in objects:
                    # 픽셀 좌표를 정규화
                    # vectors.tsv의 좌표가 이미지 크기 범위 내인지 확인
                    if not (0 <= x_pixel <= actual_width and 0 <= y_pixel <= actual_height):
                        # 범위를 벗어나면 클리핑 또는 스킵
                        if x_pixel < 0 or x_pixel > actual_width or y_pixel < 0 or y_pixel > actual_height:
                            # 이미지 크기로 나누어 정규화 시도 (vectors가 다른 스케일일 수 있음)
                            if x_pixel > 10 or y_pixel > 10:
                                # 작은 값이면 이미 정규화된 좌표일 수 있음
                                x_norm = max(0, min(1, x_pixel / actual_width))
                                y_norm = max(0, min(1, y_pixel / actual_height))
                            else:
                                # 픽셀 좌표 범위를 벗어난 경우 클리핑
                                x_pixel = max(0, min(actual_width, x_pixel))
                                y_pixel = max(0, min(actual_height, y_pixel))
                                x_norm = x_pixel / actual_width
                                y_norm = y_pixel / actual_height
                        else:
                            x_norm = x_pixel / actual_width
                            y_norm = y_pixel / actual_height
                    else:
                        x_norm = x_pixel / actual_width
                        y_norm = y_pixel / actual_height
                    
                    # 클래스별 기본 크기 가져오기
                    width_norm, height_norm = get_default_bbox_size(yolo_class, actual_width, actual_height)
                    
                    # 좌표가 박스 범위를 벗어나지 않도록 조정
                    x_norm = max(width_norm/2, min(1 - width_norm/2, x_norm))
                    y_norm = max(height_norm/2, min(1 - height_norm/2, y_norm))
                    
                    # YOLO 형식으로 저장: class_id x_center y_center width height
                    f.write(f"{yolo_class} {x_norm:.6f} {y_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
            
            converted_count += 1
        except Exception as e:
            print(f"⚠️  Error creating label for {matched_image}: {e}")
            error_count += 1
    
    print(f"\n✅ Conversion completed!")
    print(f"   Converted: {converted_count} images")
    print(f"   Skipped: {skipped_count} images")
    print(f"   Errors: {error_count} images")
    print(f"\n⚠️  IMPORTANT NOTES:")
    print(f"   1. This script assumes vectors.tsv contains center coordinates")
    print(f"   2. Default bounding box sizes are used (may not be accurate)")
    print(f"   3. Please verify the generated labels and adjust if necessary")
    print(f"   4. Labels 5, 6, 7 are mapped to class 4 (traffic_sign) - verify with data team")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create YOLO labels from vectors.tsv')
    parser.add_argument('--metadata', type=str, default='test/metadata.tsv',
                       help='Path to metadata.tsv')
    parser.add_argument('--vectors', type=str, default='test/vectors.tsv',
                       help='Path to vectors.tsv')
    parser.add_argument('--image_dir', type=str, default='test',
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for label files')
    parser.add_argument('--img_width', type=int, default=1280,
                       help='Image width in pixels')
    parser.add_argument('--img_height', type=int, default=720,
                       help='Image height in pixels')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.metadata):
        print(f"❌ Error: metadata.tsv not found at {args.metadata}")
        return
    
    if not os.path.exists(args.vectors):
        print(f"❌ Error: vectors.tsv not found at {args.vectors}")
        return
    
    if not os.path.exists(args.image_dir):
        print(f"❌ Error: Image directory not found at {args.image_dir}")
        return
    
    create_yolo_labels_from_vectors(
        args.metadata,
        args.vectors,
        args.image_dir,
        args.output_dir,
        args.img_width,
        args.img_height
    )


if __name__ == '__main__':
    main()









