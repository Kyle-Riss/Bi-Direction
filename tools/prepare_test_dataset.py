"""
테스트 데이터셋 준비 스크립트

FastCut 처리된 CARLA 데이터 + 리얼 데이터를 혼합하여
총 2800개 정도의 테스트 데이터셋 준비

구조:
- 리얼 데이터: test/ (1156개)
- CARLA 데이터: FastCut 처리된 합성 데이터
- 혼합 비율: 리얼 데이터 + CARLA 데이터 = 약 2800개
"""
import os
import shutil
import glob
import random
import argparse
from pathlib import Path
from collections import defaultdict


def count_data(data_dir, split='train'):
    """데이터 개수 계산"""
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        return 0
    
    images_dir = os.path.join(split_dir, 'images')
    if os.path.exists(images_dir):
        return len(glob.glob(os.path.join(images_dir, '*.jpg'))) + \
               len(glob.glob(os.path.join(images_dir, '*.png')))
    return 0


def prepare_test_dataset(real_data_dir='test', 
                         carla_data_dir=None,
                         target_total=2800,
                         output_dir='datasets/test_mixed',
                         real_weight=0.4,  # 리얼 데이터 40%
                         carla_weight=0.6):  # CARLA 데이터 60%
    """
    테스트 데이터셋 준비
    
    Args:
        real_data_dir: 리얼 데이터 디렉토리 (test/)
        carla_data_dir: CARLA 데이터 디렉토리
        target_total: 목표 총 데이터 개수 (2800개)
        output_dir: 출력 디렉토리
        real_weight: 리얼 데이터 가중치
        carla_weight: CARLA 데이터 가중치
    """
    print("=" * 60)
    print("🧪 테스트 데이터셋 준비 (Mixed: CARLA + Real)")
    print("=" * 60)
    
    # 1. 리얼 데이터 확인
    print("\n1️⃣  리얼 데이터 확인...")
    real_images = glob.glob(os.path.join(real_data_dir, '*.jpg'))
    real_labels_dir = os.path.join(real_data_dir, 'labels')
    real_labels = glob.glob(os.path.join(real_labels_dir, '*.txt')) if os.path.exists(real_labels_dir) else []
    
    print(f"   리얼 이미지: {len(real_images)}개")
    print(f"   리얼 라벨: {len(real_labels)}개")
    
    if len(real_images) == 0:
        print("❌ 리얼 데이터가 없습니다!")
        return False
    
    # 리얼 데이터 이름 매칭
    real_image_names = {os.path.basename(f).replace('.jpg', '') for f in real_images}
    real_label_names = {os.path.basename(f).replace('.txt', '') for f in real_labels}
    real_matched = sorted(list(real_image_names & real_label_names))
    
    print(f"   매칭된 이미지-라벨: {len(real_matched)}개")
    
    # 2. CARLA 데이터 확인
    carla_count = 0
    carla_matched = []
    
    if carla_data_dir and os.path.exists(carla_data_dir):
        print("\n2️⃣  CARLA 데이터 확인...")
        
        # CARLA 데이터가 이미 분할되어 있는지 확인
        carla_train_images = glob.glob(os.path.join(carla_data_dir, 'train', 'images', '*.jpg'))
        carla_train_labels = glob.glob(os.path.join(carla_data_dir, 'train', 'labels', '*.txt'))
        
        if len(carla_train_images) > 0:
            # 이미 분할된 형태
            carla_image_names = {os.path.basename(f).replace('.jpg', '') for f in carla_train_images}
            carla_label_names = {os.path.basename(f).replace('.txt', '') for f in carla_train_labels}
            carla_matched = sorted(list(carla_image_names & carla_label_names))
            carla_count = len(carla_matched)
            carla_base_dir = os.path.join(carla_data_dir, 'train')
        else:
            # 아직 분할 안된 형태
            carla_images = glob.glob(os.path.join(carla_data_dir, 'images', '*.jpg'))
            carla_labels = glob.glob(os.path.join(carla_data_dir, 'labels', '*.txt'))
            
            if len(carla_images) > 0:
                carla_image_names = {os.path.basename(f).replace('.jpg', '') for f in carla_images}
                carla_label_names = {os.path.basename(f).replace('.jpg', '') for f in carla_labels}
                carla_matched = sorted(list(carla_image_names & carla_label_names))
                carla_count = len(carla_matched)
                carla_base_dir = carla_data_dir
        
        print(f"   CARLA 이미지-라벨: {carla_count}개")
    
    # 3. 혼합 비율 계산
    print("\n3️⃣  혼합 비율 계산...")
    
    real_available = len(real_matched)
    carla_available = carla_count
    
    if carla_available == 0:
        print("⚠️  CARLA 데이터가 없습니다. 리얼 데이터만 사용합니다.")
        use_real_only = True
        selected_real = real_matched
        selected_carla = []
    else:
        use_real_only = False
        
        # 목표 개수에 맞춰 샘플링
        # 리얼: 약 40% (1120개)
        # CARLA: 약 60% (1680개)
        target_real = int(target_total * real_weight)
        target_carla = target_total - target_real
        
        # 실제 사용 가능한 개수 고려
        selected_real_count = min(target_real, real_available)
        selected_carla_count = min(target_carla, carla_available)
        
        # 랜덤 샘플링
        random.seed(42)
        selected_real = random.sample(real_matched, selected_real_count)
        selected_carla = random.sample(carla_matched, selected_carla_count) if carla_available > 0 else []
        
        total_selected = len(selected_real) + len(selected_carla)
        
        print(f"   목표 총 개수: {target_total}개")
        print(f"   리얼 데이터: {len(selected_real)}개 ({len(selected_real)/total_selected*100:.1f}%)")
        print(f"   CARLA 데이터: {len(selected_carla)}개 ({len(selected_carla)/total_selected*100:.1f}%)")
        print(f"   실제 총 개수: {total_selected}개")
    
    # 4. 출력 디렉토리 생성
    print("\n4️⃣  출력 디렉토리 생성...")
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)
    
    # 5. 리얼 데이터 복사
    print("\n5️⃣  리얼 데이터 복사...")
    copied_real = 0
    
    for name in selected_real:
        # 이미지 복사
        src_img = os.path.join(real_data_dir, f"{name}.jpg")
        dst_img = os.path.join(output_dir, 'images', f"{name}_real.jpg")
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
            copied_real += 1
        
        # 라벨 복사
        src_label = os.path.join(real_labels_dir, f"{name}.txt")
        dst_label = os.path.join(output_dir, 'labels', f"{name}_real.txt")
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
    
    print(f"   ✅ 리얼 데이터 {copied_real}개 복사 완료")
    
    # 6. CARLA 데이터 복사
    copied_carla = 0
    
    if not use_real_only:
        print("\n6️⃣  CARLA 데이터 복사...")
        
        carla_images_dir = os.path.join(carla_base_dir, 'images')
        carla_labels_dir = os.path.join(carla_base_dir, 'labels')
        
        for name in selected_carla:
            # 이미지 복사
            src_img = os.path.join(carla_images_dir, f"{name}.jpg")
            if not os.path.exists(src_img):
                src_img = os.path.join(carla_images_dir, f"{name}.png")
            
            dst_img = os.path.join(output_dir, 'images', f"{name}_carla.jpg")
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
                copied_carla += 1
            
            # 라벨 복사
            src_label = os.path.join(carla_labels_dir, f"{name}.txt")
            dst_label = os.path.join(output_dir, 'labels', f"{name}_carla.txt")
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
        
        print(f"   ✅ CARLA 데이터 {copied_carla}개 복사 완료")
    
    # 7. 메타데이터 생성
    print("\n7️⃣  메타데이터 생성...")
    
    metadata = {
        'total': copied_real + copied_carla,
        'real': copied_real,
        'carla': copied_carla,
        'real_ratio': copied_real / (copied_real + copied_carla) if (copied_real + copied_carla) > 0 else 0,
        'carla_ratio': copied_carla / (copied_real + copied_carla) if (copied_real + copied_carla) > 0 else 0
    }
    
    metadata_file = os.path.join(output_dir, 'metadata', 'dataset_info.txt')
    with open(metadata_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("테스트 데이터셋 정보\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"총 데이터 개수: {metadata['total']}개\n")
        f.write(f"  - 리얼 데이터: {metadata['real']}개 ({metadata['real_ratio']*100:.1f}%)\n")
        f.write(f"  - CARLA 데이터: {metadata['carla']}개 ({metadata['carla_ratio']*100:.1f}%)\n\n")
        f.write(f"리얼 데이터 출처: {real_data_dir}\n")
        if carla_data_dir:
            f.write(f"CARLA 데이터 출처: {carla_data_dir}\n")
        f.write(f"출력 디렉토리: {output_dir}\n")
    
    print(f"   ✅ 메타데이터 저장: {metadata_file}")
    
    # 8. 최종 요약
    print("\n" + "=" * 60)
    print("✅ 테스트 데이터셋 준비 완료!")
    print("=" * 60)
    print(f"출력 디렉토리: {output_dir}")
    print(f"총 데이터 개수: {metadata['total']}개")
    print(f"  - 리얼 데이터: {metadata['real']}개 ({metadata['real_ratio']*100:.1f}%)")
    print(f"  - CARLA 데이터: {metadata['carla']}개 ({metadata['carla_ratio']*100:.1f}%)")
    print(f"\n📁 구조:")
    print(f"   {output_dir}/")
    print(f"   ├── images/      (리얼: *_real.jpg, CARLA: *_carla.jpg)")
    print(f"   ├── labels/      (리얼: *_real.txt, CARLA: *_carla.txt)")
    print(f"   └── metadata/    (데이터셋 정보)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare mixed test dataset (CARLA + Real)')
    parser.add_argument('--real_dir', type=str, default='test',
                       help='Real data directory (test/)')
    parser.add_argument('--carla_dir', type=str, default=None,
                       help='CARLA data directory (optional)')
    parser.add_argument('--output_dir', type=str, default='datasets/test_mixed',
                       help='Output directory for mixed dataset')
    parser.add_argument('--target_total', type=int, default=2800,
                       help='Target total number of samples')
    parser.add_argument('--real_weight', type=float, default=0.4,
                       help='Real data weight (default: 0.4 = 40%%)')
    parser.add_argument('--carla_weight', type=float, default=0.6,
                       help='CARLA data weight (default: 0.6 = 60%%)')
    
    args = parser.parse_args()
    
    # 가중치 정규화
    total_weight = args.real_weight + args.carla_weight
    if total_weight != 1.0:
        args.real_weight /= total_weight
        args.carla_weight /= total_weight
        print(f"⚠️  가중치 정규화: real={args.real_weight:.2f}, carla={args.carla_weight:.2f}")
    
    prepare_test_dataset(
        real_data_dir=args.real_dir,
        carla_data_dir=args.carla_dir,
        target_total=args.target_total,
        output_dir=args.output_dir,
        real_weight=args.real_weight,
        carla_weight=args.carla_weight
    )


if __name__ == '__main__':
    main()








