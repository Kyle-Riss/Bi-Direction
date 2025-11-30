"""
리얼 이미지 데이터(test 폴더)를 YOLO 형식으로 변환하는 스크립트

metadata.tsv와 vectors.tsv를 파싱하여 YOLO 형식의 라벨 파일(.txt) 생성
"""
import os
import csv
from collections import defaultdict
from PIL import Image


def parse_metadata(metadata_path):
    """metadata.tsv를 파싱하여 이미지별 라벨 정보 추출"""
    image_labels = defaultdict(list)  # {image_name: [(label_id, object_id), ...]}
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for idx, row in enumerate(reader):
            label = int(row['label'])
            meta = row['meta']  # format: "image_name-label_id"
            
            # meta에서 이미지명과 객체 ID 추출
            parts = meta.split('-')
            image_name = parts[0]  # 예: "68930b82-89f59985.jpg"
            
            # 이미지명에 .jpg가 포함되어 있는지 확인
            if not image_name.endswith('.jpg'):
                image_name = parts[0] + '.jpg'
            
            image_labels[image_name].append((label, idx))
    
    return image_labels


def parse_vectors(vectors_path):
    """vectors.tsv를 파싱하여 객체별 좌표 정보 추출"""
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


def convert_to_yolo_format(vectors, image_labels, image_dir, output_label_dir):
    """
    vectors.tsv와 metadata.tsv를 결합하여 YOLO 형식의 라벨 파일 생성
    
    주의: vectors.tsv가 바운딩 박스 좌표인지 임베딩 벡터인지 확인 필요
    만약 임베딩 벡터라면, 이 스크립트는 수정이 필요합니다.
    """
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 이미지별로 그룹화
    image_to_objects = defaultdict(list)  # {image_name: [(label, x, y), ...]}
    
    for image_name, label_indices in image_labels.items():
        for label, obj_idx in label_indices:
            if obj_idx < len(vectors):
                x, y = vectors[obj_idx]
                image_to_objects[image_name].append((label, x, y))
    
    # 각 이미지에 대한 YOLO 라벨 파일 생성
    converted_count = 0
    skipped_count = 0
    
    for image_name, objects in image_to_objects.items():
        image_path = os.path.join(image_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"⚠️  Warning: Image not found: {image_path}")
            skipped_count += 1
            continue
        
        # 이미지 크기 가져오기
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
        except Exception as e:
            print(f"⚠️  Warning: Could not open image {image_path}: {e}")
            skipped_count += 1
            continue
        
        # YOLO 라벨 파일 생성
        label_file = os.path.join(output_label_dir, image_name.replace('.jpg', '.txt'))
        
        with open(label_file, 'w') as f:
            for label, x, y in objects:
                # vectors.tsv의 값이 정규화되어 있는지 확인 필요
                # 일반적으로 YOLO 형식: class_id x_center y_center width height (0-1 normalized)
                
                # 만약 vectors가 임베딩 벡터라면, 바운딩 박스 정보가 없으므로
                # 더미 바운딩 박스를 생성하거나 다른 방법이 필요합니다
                
                # vectors.tsv의 의미 확인 필요:
                # - 만약 바운딩 박스 좌표라면, x, y가 중심 좌표일 수도 있고
                # - 임베딩 벡터라면 바운딩 박스 정보가 없음
                
                # 이미지 크기 (1280x720) 기준으로 정규화
                # vectors 값 범위를 확인하여 좌표인지 벡터인지 판단
                # 일반적으로 임베딩 벡터는 더 큰 범위의 값을 가짐
                
                # 임시 해결책: vectors가 임베딩 벡터라면 바운딩 박스를 생성할 수 없음
                # 대신 이미지 전체를 객체 영역으로 가정하거나, 
                # 이미지의 작은 영역을 객체로 가정
                
                # vectors.tsv 값이 정규화 좌표인지 확인 (0-1 범위)
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # 이미 정규화된 좌표로 가정
                    x_norm = x
                    y_norm = y
                elif 0 <= x <= img_width and 0 <= y <= img_height:
                    # 픽셀 좌표로 가정 - 정규화
                    x_norm = x / img_width
                    y_norm = y / img_height
                else:
                    # 임베딩 벡터로 보임 - 바운딩 박스 정보 없음
                    # 임시로 이미지 중심에 작은 박스 생성
                    print(f"⚠️  Warning: vectors.tsv may contain embedding vectors, not bounding box coordinates")
                    x_norm = 0.5  # 이미지 중심
                    y_norm = 0.5
                
                # 기본 크기의 바운딩 박스 (실제로는 width, height 정보 필요)
                # 임시로 작은 박스 생성 (이미지의 10%)
                # 실제로는 바운딩 박스 정보를 데이터 담당자에게 요청해야 함
                w = 0.1
                h = 0.1
                
                # 좌표가 박스 범위를 벗어나지 않도록 조정
                x_norm = max(w/2, min(1 - w/2, x_norm))
                y_norm = max(h/2, min(1 - h/2, y_norm))
                
                # YOLO 형식으로 저장: class_id x_center y_center width height
                # label이 5, 6, 7 등이 있는데, 이것들은 클래스 매핑이 필요할 수 있습니다
                # 기본 클래스: 0: pedestrian, 1: car, 2: truck_bus, 3: bicycle_motorcycle, 4: traffic_sign
                
                # label 매핑 (5, 6, 7 등은 적절한 클래스로 매핑 필요)
                mapped_label = map_label_to_yolo_class(label)
                
                if mapped_label is not None:
                    f.write(f"{mapped_label} {x_norm:.6f} {y_norm:.6f} {w:.6f} {h:.6f}\n")
        
        converted_count += 1
    
    print(f"✅ Converted {converted_count} images")
    print(f"⚠️  Skipped {skipped_count} images")
    print(f"📁 Labels saved to: {output_label_dir}")


def map_label_to_yolo_class(label):
    """
    metadata.tsv의 label을 YOLO 클래스로 매핑
    
    현재 클래스 정의 (carla_data.yaml 참고):
    0: pedestrian
    1: car
    2: truck_bus
    3: bicycle_motorcycle
    4: traffic_sign
    """
    # 기본 매핑 (실제 매핑은 데이터 담당자와 협의 필요)
    label_mapping = {
        0: 0,  # pedestrian
        1: 1,  # car
        2: 2,  # truck_bus
        3: 3,  # bicycle_motorcycle
        4: 4,  # traffic_sign (없는 경우도 있음)
        5: 4,  # traffic_sign으로 가정 (실제로는 확인 필요)
        6: 4,  # traffic_sign으로 가정 (실제로는 확인 필요)
        7: 4,  # traffic_sign으로 가정 (실제로는 확인 필요)
    }
    
    return label_mapping.get(label, None)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert real image data to YOLO format')
    parser.add_argument('--test_dir', type=str, default='test',
                       help='Directory containing test images, metadata.tsv, and vectors.tsv')
    parser.add_argument('--output_label_dir', type=str, default='test/labels',
                       help='Output directory for YOLO label files')
    
    args = parser.parse_args()
    
    # 경로 설정
    metadata_path = os.path.join(args.test_dir, 'metadata.tsv')
    vectors_path = os.path.join(args.test_dir, 'vectors.tsv')
    image_dir = args.test_dir
    
    # 파일 존재 확인
    if not os.path.exists(metadata_path):
        print(f"❌ Error: metadata.tsv not found at {metadata_path}")
        return
    
    if not os.path.exists(vectors_path):
        print(f"❌ Error: vectors.tsv not found at {vectors_path}")
        return
    
    print("📖 Parsing metadata.tsv...")
    image_labels = parse_metadata(metadata_path)
    print(f"   Found {len(image_labels)} images with labels")
    
    print("📖 Parsing vectors.tsv...")
    vectors = parse_vectors(vectors_path)
    print(f"   Found {len(vectors)} vectors")
    
    print("🔄 Converting to YOLO format...")
    convert_to_yolo_format(vectors, image_labels, image_dir, args.output_label_dir)
    
    print("✅ Conversion completed!")
    print("\n⚠️  Note: This script assumes vectors.tsv contains bounding box coordinates.")
    print("   If vectors.tsv contains embedding vectors, the script needs to be modified.")
    print("   Please verify the output labels and adjust if necessary.")


if __name__ == '__main__':
    main()

