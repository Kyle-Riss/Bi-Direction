"""
리얼 데이터 준비 스크립트 (모델 담당)

1. test 폴더의 리얼 이미지 데이터를 분석
2. metadata.tsv를 파싱하여 라벨 정보 확인
3. YOLO 형식으로 변환 가능한 데이터 구조 정리
4. datasets 폴더로 데이터 분할 및 이동

주의: vectors.tsv는 임베딩 벡터로 보이며 바운딩 박스 정보가 아닐 수 있음
      실제 바운딩 박스 정보는 데이터 담당자에게 확인 필요
"""
import os
import shutil
import csv
from collections import defaultdict
from pathlib import Path
import random
from PIL import Image

def parse_metadata(metadata_path):
    """
    metadata.tsv를 파싱하여 이미지별 라벨 정보 추출
    
    Returns:
        dict: {image_name: [(label_id, object_id), ...]}
    """
    image_labels = defaultdict(list)
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for idx, row in enumerate(reader):
            try:
                label = int(row['label'])
                meta = row['meta']  # format: "image_name-label_id"
                
                # meta에서 이미지명 추출
                # 형식: "68930b82-89f59985.jpg-2" 또는 "68930b82-89f59985.jpg"
                parts = meta.split('-')
                if len(parts) >= 3:
                    # "68930b82-89f59985.jpg-2" 형식
                    image_name = '-'.join(parts[:-1]) + '.jpg'
                elif len(parts) == 2:
                    # "68930b82-89f59985.jpg" 또는 "68930b82.jpg" 형식
                    if parts[1].endswith('.jpg'):
                        image_name = parts[0] + '-' + parts[1]
                    elif parts[0].endswith('.jpg'):
                        image_name = parts[0]
                    else:
                        image_name = parts[0] + '.jpg'
                else:
                    # "68930b82.jpg" 형식
                    image_name = parts[0]
                    if not image_name.endswith('.jpg'):
                        image_name = parts[0] + '.jpg'
                
                image_labels[image_name].append((label, idx))
            except Exception as e:
                print(f"⚠️  Warning: Error parsing row {idx}: {e}")
                continue
    
    return image_labels

def analyze_labels(image_labels):
    """라벨 분포 분석"""
    label_counts = defaultdict(int)
    image_counts = defaultdict(int)
    
    for image_name, labels in image_labels.items():
        unique_labels = set(label for label, _ in labels)
        image_counts[image_name] = len(labels)
        
        for label in unique_labels:
            label_counts[label] += 1
    
    print("\n📊 Label Distribution:")
    print("=" * 50)
    print(f"Total images: {len(image_labels)}")
    print(f"Total objects: {sum(len(labels) for labels in image_labels.values())}")
    print(f"\nLabel counts:")
    for label in sorted(label_counts.keys()):
        print(f"  Label {label}: {label_counts[label]} images")
    
    print(f"\nObjects per image:")
    obj_per_img = list(image_counts.values())
    if obj_per_img:
        print(f"  Min: {min(obj_per_img)}")
        print(f"  Max: {max(obj_per_img)}")
        print(f"  Avg: {sum(obj_per_img) / len(obj_per_img):.2f}")
    
    return label_counts

def map_label_to_yolo_class(label):
    """
    metadata.tsv의 label을 YOLO 클래스로 매핑
    
    현재 클래스 정의 (carla_test_adverse.yaml 참고):
    0: pedestrian
    1: car
    2: truck_bus
    3: bicycle_motorcycle
    4: traffic_sign
    """
    # 기본 매핑 (실제 매핑은 데이터 담당자와 협의 필요)
    # metadata.tsv의 label: 0, 2, 5, 7 등
    # YOLO class: 0-4
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

def check_image_sizes(test_dir):
    """이미지 크기 확인"""
    img_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    
    sizes = {}
    for img_file in img_files[:10]:  # 샘플 10개만
        try:
            img_path = os.path.join(test_dir, img_file)
            img = Image.open(img_path)
            sizes[img_file] = img.size
        except Exception as e:
            print(f"⚠️  Warning: Could not open {img_file}: {e}")
    
    if sizes:
        unique_sizes = set(sizes.values())
        print(f"\n📐 Image Sizes (sample):")
        for size in unique_sizes:
            count = sum(1 for s in sizes.values() if s == size)
            print(f"  {size}: {count} images")
    
    return sizes

def create_dataset_structure(output_base_dir='datasets/real_data'):
    """데이터셋 디렉토리 구조 생성"""
    splits = ['train', 'val', 'test']
    
    for split in splits:
        img_dir = os.path.join(output_base_dir, split, 'images')
        label_dir = os.path.join(output_base_dir, split, 'labels')
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
    
    print(f"✅ Created dataset structure: {output_base_dir}")
    return output_base_dir

def split_and_copy_images(test_dir, output_base_dir, image_labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    이미지를 train/val/test로 분할하고 복사
    
    주의: 현재는 바운딩 박스 정보가 없으므로 이미지만 복사
          라벨 파일은 나중에 바운딩 박스 정보를 받으면 생성
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # 실제 이미지 파일 목록 가져오기
    actual_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    actual_files_set = set(actual_files)
    
    # metadata.tsv의 이미지 이름과 실제 파일 매칭
    # metadata.tsv에는 짧은 해시가 있을 수 있으므로 prefix 매칭 시도
    matched_files = []
    for img_name in image_labels.keys():
        # 정확한 매칭 시도
        if img_name in actual_files_set:
            matched_files.append(img_name)
        else:
            # prefix 매칭 시도 (이미지 이름이 파일명의 prefix인 경우)
            img_prefix = img_name.replace('.jpg', '')
            matched = [f for f in actual_files if f.startswith(img_prefix)]
            if matched:
                # 첫 번째 매칭된 파일 사용
                matched_files.append(matched[0])
    
    # 중복 제거 및 정렬
    image_files = sorted(list(set(matched_files)))
    
    # 셔플
    random.seed(42)
    random.shuffle(image_files)
    
    # 분할
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    train_images = image_files[:train_end]
    val_images = image_files[train_end:val_end]
    test_images = image_files[val_end:]
    
    if total == 0:
        print(f"\n⚠️  Warning: No matching images found!")
        print(f"   Metadata images: {len(image_labels)}")
        print(f"   Actual files: {len(actual_files)}")
        print(f"   Matched: {len(matched_files)}")
        print(f"\n   Trying to match all actual files instead...")
        # 매칭 실패 시 모든 이미지 파일 사용
        image_files = sorted(actual_files)
        total = len(image_files)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        train_images = image_files[:train_end]
        val_images = image_files[train_end:val_end]
        test_images = image_files[val_end:]
    
    print(f"\n📦 Splitting {total} images:")
    if total > 0:
        print(f"  Train: {len(train_images)} ({len(train_images)/total*100:.1f}%)")
        print(f"  Val: {len(val_images)} ({len(val_images)/total*100:.1f}%)")
        print(f"  Test: {len(test_images)} ({len(test_images)/total*100:.1f}%)")
    else:
        print(f"  No images to split!")
    
    # 복사
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    for split, img_list in splits.items():
        img_dir = os.path.join(output_base_dir, split, 'images')
        
        for img_file in img_list:
            src = os.path.join(test_dir, img_file)
            dst = os.path.join(img_dir, img_file)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"⚠️  Warning: Could not copy {img_file}: {e}")
        
        print(f"  ✅ Copied {len(img_list)} images to {split}/images")
    
    # 라벨 정보 저장 (나중에 바운딩 박스 정보 추가 시 사용)
    label_info_path = os.path.join(output_base_dir, 'label_info.txt')
    with open(label_info_path, 'w') as f:
        f.write("# Label information for real data\n")
        f.write("# Format: image_name: [(yolo_class, original_label), ...]\n")
        f.write("# NOTE: Bounding box information is missing - needs to be added\n\n")
        
        for img_file in image_files:
            if img_file in image_labels:
                labels = image_labels[img_file]
                yolo_labels = []
                for orig_label, obj_id in labels:
                    yolo_class = map_label_to_yolo_class(orig_label)
                    if yolo_class is not None:
                        yolo_labels.append((yolo_class, orig_label))
                
                if yolo_labels:
                    f.write(f"{img_file}: {yolo_labels}\n")
    
    print(f"\n✅ Label info saved to: {label_info_path}")
    
    return splits

def create_summary_report(test_dir, output_base_dir, image_labels):
    """요약 보고서 생성"""
    report_path = os.path.join(output_base_dir, 'PREPARATION_REPORT.md')
    
    with open(report_path, 'w') as f:
        f.write("# Real Data Preparation Report\n\n")
        f.write("## 📋 Summary\n\n")
        f.write(f"- Total images: {len(image_labels)}\n")
        f.write(f"- Total objects: {sum(len(labels) for labels in image_labels.values())}\n")
        f.write(f"- Source directory: {test_dir}\n")
        f.write(f"- Output directory: {output_base_dir}\n\n")
        
        f.write("## ⚠️  Important Notes\n\n")
        f.write("1. **Missing Bounding Box Information**: vectors.tsv contains embedding vectors, not bounding box coordinates.\n")
        f.write("2. **YOLO Label Files**: Currently not created due to missing bbox info.\n")
        f.write("3. **Label Mapping**: Some labels (5, 6, 7) are mapped to class 4 (traffic_sign) - needs verification.\n")
        f.write("4. **Next Steps**:\n")
        f.write("   - Request bounding box information from data team\n")
        f.write("   - Verify label mappings (especially labels 5, 6, 7)\n")
        f.write("   - Generate YOLO format label files (.txt) once bbox info is available\n\n")
        
        f.write("## 📁 Dataset Structure\n\n")
        f.write("```\n")
        f.write(f"{output_base_dir}/\n")
        f.write("├── train/\n")
        f.write("│   ├── images/\n")
        f.write("│   └── labels/  (empty - needs bbox info)\n")
        f.write("├── val/\n")
        f.write("│   ├── images/\n")
        f.write("│   └── labels/  (empty - needs bbox info)\n")
        f.write("├── test/\n")
        f.write("│   ├── images/\n")
        f.write("│   └── labels/  (empty - needs bbox info)\n")
        f.write("├── label_info.txt\n")
        f.write("└── PREPARATION_REPORT.md\n")
        f.write("```\n")
    
    print(f"\n📄 Report saved to: {report_path}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare real image data for training')
    parser.add_argument('--test_dir', type=str, default='test',
                       help='Directory containing test images, metadata.tsv, and vectors.tsv')
    parser.add_argument('--output_dir', type=str, default='datasets/real_data',
                       help='Output directory for organized dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Ratio of validation data')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Ratio of test data')
    
    args = parser.parse_args()
    
    # 경로 확인
    metadata_path = os.path.join(args.test_dir, 'metadata.tsv')
    if not os.path.exists(metadata_path):
        print(f"❌ Error: metadata.tsv not found at {metadata_path}")
        return
    
    print("🔄 Starting real data preparation...")
    print("=" * 60)
    
    # 1. 메타데이터 파싱
    print("\n📖 Step 1: Parsing metadata.tsv...")
    image_labels = parse_metadata(metadata_path)
    print(f"   ✅ Found {len(image_labels)} images with labels")
    
    # 2. 라벨 분석
    print("\n📊 Step 2: Analyzing labels...")
    label_counts = analyze_labels(image_labels)
    
    # 3. 이미지 크기 확인
    print("\n📐 Step 3: Checking image sizes...")
    check_image_sizes(args.test_dir)
    
    # 4. 데이터셋 구조 생성
    print("\n📁 Step 4: Creating dataset structure...")
    output_base_dir = create_dataset_structure(args.output_dir)
    
    # 5. 이미지 분할 및 복사
    print("\n📦 Step 5: Splitting and copying images...")
    splits = split_and_copy_images(
        args.test_dir, 
        output_base_dir, 
        image_labels,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )
    
    # 6. 보고서 생성
    print("\n📄 Step 6: Creating summary report...")
    create_summary_report(args.test_dir, output_base_dir, image_labels)
    
    print("\n" + "=" * 60)
    print("✅ Real data preparation completed!")
    print("\n⚠️  IMPORTANT:")
    print("   - Bounding box information is missing from vectors.tsv")
    print("   - Label files (.txt) cannot be created without bbox info")
    print("   - Please request bounding box data from data team")
    print(f"\n📁 Output directory: {output_base_dir}")

if __name__ == '__main__':
    main()

