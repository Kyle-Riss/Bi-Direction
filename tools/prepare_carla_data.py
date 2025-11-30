"""
CARLA 시뮬레이션 데이터 준비 스크립트

FastCut으로 처리된 CARLA 데이터를 받아서 학습 가능한 형태로 준비
"""
import os
import shutil
import glob
import argparse
from pathlib import Path
from collections import defaultdict


def validate_carla_structure(data_dir):
    """
    CARLA 데이터 구조 검증
    
    예상 구조:
    carla_data/
    ├── images/
    │   ├── *.jpg
    │   └── ...
    └── labels/
        ├── *.txt  (YOLO 형식)
        └── ...
    """
    required_dirs = ['images', 'labels']
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = os.path.join(data_dir, dir_name)
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"⚠️  경고: 다음 디렉토리가 없습니다: {missing_dirs}")
        return False
    
    # 이미지 파일 확인 (.jpg, .png 모두 지원)
    image_files_jpg = glob.glob(os.path.join(data_dir, 'images', '*.jpg'))
    image_files_png = glob.glob(os.path.join(data_dir, 'images', '*.png'))
    image_files = image_files_jpg + image_files_png
    label_files = glob.glob(os.path.join(data_dir, 'labels', '*.txt'))
    
    print(f"📊 데이터 통계:")
    print(f"   이미지: {len(image_files)}개")
    print(f"   라벨: {len(label_files)}개")
    
    if len(image_files) == 0:
        print("❌ 이미지 파일이 없습니다!")
        return False
    
    # 이미지-라벨 매칭 확인
    image_names = {os.path.basename(f).replace('.jpg', '').replace('.png', '') for f in image_files}
    label_names = {os.path.basename(f).replace('.txt', '') for f in label_files}
    
    matched = image_names & label_names
    unmatched_images = image_names - label_names
    unmatched_labels = label_names - image_names
    
    print(f"   매칭된 이미지-라벨: {len(matched)}개")
    
    if unmatched_images:
        print(f"⚠️  라벨 없는 이미지: {len(unmatched_images)}개")
    
    if unmatched_labels:
        print(f"⚠️  이미지 없는 라벨: {len(unmatched_labels)}개")
    
    return True


def split_carla_data(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    CARLA 데이터를 train/val/test로 분할
    
    Args:
        data_dir: CARLA 데이터 디렉토리
        output_dir: 출력 디렉토리
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
    """
    import random
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # 이미지 파일 목록 가져오기 (.jpg, .png 모두 지원)
    image_files_jpg = glob.glob(os.path.join(data_dir, 'images', '*.jpg'))
    image_files_png = glob.glob(os.path.join(data_dir, 'images', '*.png'))
    image_files = image_files_jpg + image_files_png
    image_names = [os.path.basename(f).replace('.jpg', '').replace('.png', '') for f in image_files]
    
    # 매칭된 이미지만 사용 (라벨이 있는 것)
    label_dir = os.path.join(data_dir, 'labels')
    matched_names = []
    for name in image_names:
        label_path = os.path.join(label_dir, f"{name}.txt")
        if os.path.exists(label_path):
            matched_names.append(name)
    
    print(f"\n📦 총 {len(matched_names)}개 이미지-라벨 쌍 분할 시작...")
    
    # 셔플
    random.seed(42)
    random.shuffle(matched_names)
    
    # 분할
    total = len(matched_names)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    train_names = matched_names[:train_end]
    val_names = matched_names[train_end:val_end]
    test_names = matched_names[val_end:]
    
    splits = {
        'train': train_names,
        'val': val_names,
        'test': test_names
    }
    
    print(f"   학습: {len(train_names)}개 ({len(train_names)/total*100:.1f}%)")
    print(f"   검증: {len(val_names)}개 ({len(val_names)/total*100:.1f}%)")
    print(f"   테스트: {len(test_names)}개 ({len(test_names)/total*100:.1f}%)")
    
    # 디렉토리 생성 및 파일 복사
    for split_name, names in splits.items():
        split_output_dir = os.path.join(output_dir, split_name)
        os.makedirs(os.path.join(split_output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_output_dir, 'labels'), exist_ok=True)
        
        for name in names:
            # 이미지 복사 (.jpg 또는 .png)
            src_img_jpg = os.path.join(data_dir, 'images', f"{name}.jpg")
            src_img_png = os.path.join(data_dir, 'images', f"{name}.png")
            if os.path.exists(src_img_jpg):
            dst_img = os.path.join(split_output_dir, 'images', f"{name}.jpg")
                shutil.copy2(src_img_jpg, dst_img)
            elif os.path.exists(src_img_png):
                dst_img = os.path.join(split_output_dir, 'images', f"{name}.png")
                shutil.copy2(src_img_png, dst_img)
            
            # 라벨 복사
            src_label = os.path.join(data_dir, 'labels', f"{name}.txt")
            dst_label = os.path.join(split_output_dir, 'labels', f"{name}.txt")
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
        
        print(f"   ✅ {split_name} 복사 완료: {split_output_dir}")
    
    return splits


def check_label_format(label_dir, num_classes=5):
    """
    라벨 형식 검증
    
    YOLO 형식: class_id x_center y_center width height (normalized)
    """
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    
    errors = []
    class_counts = defaultdict(int)
    
    for label_file in label_files[:100]:  # 샘플 100개만 확인
        try:
            with open(label_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        errors.append(f"{os.path.basename(label_file)}:{line_num}: 5개 값 필요 (현재 {len(parts)}개)")
                        continue
                    
                    try:
                        class_id = int(float(parts[0]))
                        x, y, w, h = map(float, parts[1:])
                        
                        # 범위 검증
                        if class_id < 0 or class_id >= num_classes:
                            errors.append(f"{os.path.basename(label_file)}:{line_num}: 잘못된 클래스 ID: {class_id}")
                        
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            errors.append(f"{os.path.basename(label_file)}:{line_num}: 좌표가 [0, 1] 범위를 벗어남")
                        
                        class_counts[class_id] += 1
                    
                    except ValueError as e:
                        errors.append(f"{os.path.basename(label_file)}:{line_num}: 숫자 변환 오류: {e}")
        
        except Exception as e:
            errors.append(f"{os.path.basename(label_file)}: 파일 읽기 오류: {e}")
    
    if errors:
        print(f"\n⚠️  라벨 형식 오류 ({len(errors)}개 발견):")
        for error in errors[:10]:  # 처음 10개만 출력
            print(f"   {error}")
        if len(errors) > 10:
            print(f"   ... 외 {len(errors) - 10}개")
    else:
        print("\n✅ 라벨 형식 검증 통과!")
    
    print(f"\n📊 클래스별 객체 수:")
    for class_id in sorted(class_counts.keys()):
        print(f"   클래스 {class_id}: {class_counts[class_id]}개")
    
    return len(errors) == 0


def prepare_carla_dataset(carla_data_dir, output_dir='datasets/carla_data', 
                         train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    CARLA 데이터셋 준비 메인 함수
    
    Args:
        carla_data_dir: FastCut으로 처리된 CARLA 데이터 디렉토리
        output_dir: 출력 디렉토리
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
    """
    print("=" * 60)
    print("🚗 CARLA 데이터셋 준비")
    print("=" * 60)
    
    if not os.path.exists(carla_data_dir):
        print(f"❌ 오류: CARLA 데이터 디렉토리가 없습니다: {carla_data_dir}")
        return False
    
    # 1. 데이터 구조 검증
    print("\n1️⃣  데이터 구조 검증...")
    if not validate_carla_structure(carla_data_dir):
        print("❌ 데이터 구조 검증 실패!")
        return False
    
    # 2. 라벨 형식 검증
    print("\n2️⃣  라벨 형식 검증...")
    label_dir = os.path.join(carla_data_dir, 'labels')
    check_label_format(label_dir)
    
    # 3. 데이터 분할
    print("\n3️⃣  데이터 분할 (train/val/test)...")
    os.makedirs(output_dir, exist_ok=True)
    splits = split_carla_data(carla_data_dir, output_dir, train_ratio, val_ratio, test_ratio)
    
    # 4. 요약 리포트
    print("\n4️⃣  요약 리포트 생성...")
    total_images = sum(len(names) for names in splits.values())
    
    print(f"\n{'='*60}")
    print("✅ CARLA 데이터셋 준비 완료!")
    print(f"{'='*60}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"총 이미지: {total_images}개")
    print(f"  - 학습: {len(splits['train'])}개")
    print(f"  - 검증: {len(splits['val'])}개")
    print(f"  - 테스트: {len(splits['test'])}개")
    print(f"\n다음 단계:")
    print(f"  1. 도메인 적응 학습: python tools/train_domain_adaptation.py")
    print(f"  2. 혼합 학습: datasets/mixed_dataset.py 사용")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare CARLA simulation data for training')
    parser.add_argument('--carla_dir', type=str, required=True,
                       help='CARLA data directory (should contain images/ and labels/ subdirectories)')
    parser.add_argument('--output_dir', type=str, default='datasets/carla_data',
                       help='Output directory for organized dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test data ratio')
    
    args = parser.parse_args()
    
    prepare_carla_dataset(
        args.carla_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )


if __name__ == '__main__':
    main()




