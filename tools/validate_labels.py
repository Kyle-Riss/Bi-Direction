"""
라벨 파일 검증 스크립트 (모델 담당)

1. YOLO 형식 라벨 파일 검증
2. 클래스 ID 범위 확인
3. 좌표 범위 확인 (0-1 정규화)
4. 이미지-라벨 매칭 확인
5. 라벨 통계 생성
"""
import os
import glob
from collections import defaultdict
from pathlib import Path
from PIL import Image


def validate_yolo_label(label_path, num_classes=5):
    """
    단일 YOLO 라벨 파일 검증
    
    Args:
        label_path: 라벨 파일 경로
        num_classes: 예상되는 클래스 수
    
    Returns:
        dict: 검증 결과 (is_valid, errors, warnings, stats)
    """
    errors = []
    warnings = []
    stats = {
        'num_objects': 0,
        'class_ids': [],
        'bboxes': []
    }
    
    if not os.path.exists(label_path):
        return {
            'is_valid': False,
            'errors': [f'Label file not found: {label_path}'],
            'warnings': warnings,
            'stats': stats
        }
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line_idx, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                errors.append(f'Line {line_idx}: Expected 5 values, got {len(parts)}')
                continue
            
            try:
                class_id = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError as e:
                errors.append(f'Line {line_idx}: Invalid number format: {e}')
                continue
            
            # 클래스 ID 검증
            if class_id < 0 or class_id >= num_classes:
                errors.append(f'Line {line_idx}: Class ID {class_id} out of range [0, {num_classes-1}]')
                continue
            
            # 좌표 범위 검증 (0-1 정규화)
            if not (0 <= x_center <= 1):
                errors.append(f'Line {line_idx}: x_center {x_center} out of range [0, 1]')
            if not (0 <= y_center <= 1):
                errors.append(f'Line {line_idx}: y_center {y_center} out of range [0, 1]')
            if not (0 < width <= 1):
                errors.append(f'Line {line_idx}: width {width} out of range (0, 1]')
            if not (0 < height <= 1):
                errors.append(f'Line {line_idx}: height {height} out of range (0, 1]')
            
            # 바운딩 박스 범위 검증
            x_min = x_center - width / 2
            x_max = x_center + width / 2
            y_min = y_center - height / 2
            y_max = y_center + height / 2
            
            if not (0 <= x_min < x_max <= 1):
                errors.append(f'Line {line_idx}: Invalid x coordinates (x_min={x_min:.3f}, x_max={x_max:.3f})')
            if not (0 <= y_min < y_max <= 1):
                errors.append(f'Line {line_idx}: Invalid y coordinates (y_min={y_min:.3f}, y_max={y_max:.3f})')
            
            # 박스 크기 경고
            if width < 0.01 or height < 0.01:
                warnings.append(f'Line {line_idx}: Very small bounding box (w={width:.3f}, h={height:.3f})')
            if width > 0.9 or height > 0.9:
                warnings.append(f'Line {line_idx}: Very large bounding box (w={width:.3f}, h={height:.3f})')
            
            stats['num_objects'] += 1
            stats['class_ids'].append(class_id)
            stats['bboxes'].append((x_center, y_center, width, height))
    
    except Exception as e:
        errors.append(f'Error reading label file: {e}')
    
    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'stats': stats
    }


def validate_dataset(img_dir, label_dir, num_classes=5):
    """
    데이터셋 전체 검증
    
    Returns:
        dict: 검증 결과 통계
    """
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    label_files = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
    
    # 이미지-라벨 매칭
    img_basenames = {os.path.basename(f).replace('.jpg', '') for f in img_files}
    label_basenames = {os.path.basename(f).replace('.txt', '') for f in label_files}
    
    missing_labels = img_basenames - label_basenames
    missing_images = label_basenames - img_basenames
    matched = img_basenames & label_basenames
    
    print(f"\n📊 Dataset Validation:")
    print("=" * 60)
    print(f"Total images: {len(img_files)}")
    print(f"Total labels: {len(label_files)}")
    print(f"Matched pairs: {len(matched)}")
    print(f"Images without labels: {len(missing_labels)}")
    print(f"Labels without images: {len(missing_images)}")
    
    # 각 라벨 파일 검증
    total_errors = 0
    total_warnings = 0
    class_distribution = defaultdict(int)
    bbox_sizes = []
    
    valid_labels = 0
    invalid_labels = 0
    
    for label_file in label_files:
        result = validate_yolo_label(label_file, num_classes)
        
        if result['is_valid']:
            valid_labels += 1
        else:
            invalid_labels += 1
            total_errors += len(result['errors'])
            if result['errors']:
                print(f"\n❌ {os.path.basename(label_file)}:")
                for error in result['errors'][:3]:  # 처음 3개만 표시
                    print(f"   {error}")
        
        total_warnings += len(result['warnings'])
        
        # 통계 수집
        for class_id in result['stats']['class_ids']:
            class_distribution[class_id] += 1
        
        for x, y, w, h in result['stats']['bboxes']:
            bbox_sizes.append(w * h)  # 면적
    
    print(f"\n✅ Validation Results:")
    print(f"   Valid labels: {valid_labels}")
    print(f"   Invalid labels: {invalid_labels}")
    print(f"   Total errors: {total_errors}")
    print(f"   Total warnings: {total_warnings}")
    
    print(f"\n📈 Class Distribution:")
    for class_id in sorted(class_distribution.keys()):
        print(f"   Class {class_id}: {class_distribution[class_id]} objects")
    
    if bbox_sizes:
        print(f"\n📐 Bounding Box Sizes:")
        print(f"   Min area: {min(bbox_sizes):.4f}")
        print(f"   Max area: {max(bbox_sizes):.4f}")
        print(f"   Avg area: {sum(bbox_sizes) / len(bbox_sizes):.4f}")
    
    # 이미지 크기 확인
    if img_files:
        sample_sizes = []
        for img_file in img_files[:10]:
            try:
                img = Image.open(img_file)
                sample_sizes.append(img.size)
            except:
                pass
        
        if sample_sizes:
            unique_sizes = set(sample_sizes)
            print(f"\n🖼️  Image Sizes (sample):")
            for size in unique_sizes:
                count = sum(1 for s in sample_sizes if s == size)
                print(f"   {size}: {count} images")
    
    return {
        'total_images': len(img_files),
        'total_labels': len(label_files),
        'matched_pairs': len(matched),
        'missing_labels': len(missing_labels),
        'missing_images': len(missing_images),
        'valid_labels': valid_labels,
        'invalid_labels': invalid_labels,
        'total_errors': total_errors,
        'total_warnings': total_warnings,
        'class_distribution': dict(class_distribution),
        'bbox_areas': bbox_sizes
    }


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate YOLO label files')
    parser.add_argument('--img_dir', type=str, required=True,
                       help='Image directory')
    parser.add_argument('--label_dir', type=str, required=True,
                       help='Label directory')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of classes (default: 5)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.img_dir):
        print(f"❌ Error: Image directory not found: {args.img_dir}")
        return
    
    if not os.path.exists(args.label_dir):
        print(f"❌ Error: Label directory not found: {args.label_dir}")
        return
    
    validate_dataset(args.img_dir, args.label_dir, args.num_classes)


if __name__ == '__main__':
    main()









