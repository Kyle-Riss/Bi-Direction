"""
CARLA + 리얼 데이터 통합 스크립트

FastCut으로 처리된 CARLA 데이터와 리얼 데이터를 통합하여
도메인 적응 학습에 사용할 수 있는 형태로 준비
"""
import os
import argparse
from pathlib import Path
import sys

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from datasets.mixed_dataset import create_mixed_training_dataloaders


def check_data_availability(carla_dir=None, real_dir=None):
    """
    데이터 가용성 확인
    
    Returns:
        available: dict with availability status
    """
    available = {
        'carla': False,
        'real': False
    }
    
    if carla_dir:
        carla_train = os.path.join(carla_dir, 'train')
        if os.path.exists(carla_train):
            carla_images = os.path.join(carla_train, 'images')
            carla_labels = os.path.join(carla_train, 'labels')
            if os.path.exists(carla_images) and os.path.exists(carla_labels):
                import glob
                images = glob.glob(os.path.join(carla_images, '*.jpg'))
                if len(images) > 0:
                    available['carla'] = True
    
    if real_dir:
        real_train = os.path.join(real_dir, 'train')
        if os.path.exists(real_train):
            real_images = os.path.join(real_train, 'images')
            real_labels = os.path.join(real_train, 'labels')
            if os.path.exists(real_images) and os.path.exists(real_labels):
                import glob
                images = glob.glob(os.path.join(real_images, '*.jpg'))
                if len(images) > 0:
                    available['real'] = True
    
    return available


def print_integration_guide(carla_dir=None, real_dir=None):
    """
    통합 가이드 출력
    """
    print("=" * 60)
    print("🔄 CARLA + 리얼 데이터 통합 가이드")
    print("=" * 60)
    
    available = check_data_availability(carla_dir, real_dir)
    
    print("\n📊 데이터 가용성:")
    print(f"   CARLA 데이터: {'✅ 사용 가능' if available['carla'] else '❌ 없음'}")
    print(f"   리얼 데이터: {'✅ 사용 가능' if available['real'] else '❌ 없음'}")
    
    if not available['carla'] and not available['real']:
        print("\n⚠️  사용 가능한 데이터가 없습니다!")
        print("\n다음 단계:")
        print("1. CARLA 데이터 준비:")
        print("   python tools/prepare_carla_data.py --carla_dir /path/to/carla_data")
        print("2. 리얼 데이터 준비 (이미 완료): datasets/real_data")
        return
    
    if available['carla'] and available['real']:
        print("\n✅ 두 데이터 모두 사용 가능! 도메인 적응 학습 가능")
        print("\n📋 사용 가능한 학습 방법:")
        print("\n1️⃣  도메인 적응 학습 (Adversarial + Feature Alignment):")
        print(f"   python tools/train_domain_adaptation.py \\")
        print(f"       --carla_train {carla_dir}/train \\")
        print(f"       --real_train {real_dir}/train \\")
        print(f"       --epochs 20 \\")
        print(f"       --lambda_adv 0.1 \\")
        print(f"       --lambda_align 0.1")
        
        print("\n2️⃣  혼합 학습 (비율 조절):")
        print("   from datasets.mixed_dataset import create_mixed_training_dataloaders")
        print("   train_loader, val_loader = create_mixed_training_dataloaders(")
        print(f"       carla_train_dir='{carla_dir}/train',")
        print(f"       real_train_dir='{real_dir}/train',")
        print("       carla_weight=0.5,  # 시뮬 50%")
        print("       real_weight=0.5,   # 리얼 50%")
        print("       mode='concat'")
        print("   )")
        
        print("\n3️⃣  단계별 학습 전략:")
        print("   a) CARLA 단독 학습 (Pre-training)")
        print("   b) 도메인 적응 학습 (Adversarial + Alignment)")
        print("   c) 혼합 학습 Fine-tuning")
    
    elif available['carla']:
        print("\n⚠️  CARLA 데이터만 사용 가능 - 단독 학습만 가능")
        print("   리얼 데이터를 준비하면 도메인 적응 학습 가능")
    
    elif available['real']:
        print("\n⚠️  리얼 데이터만 사용 가능")
        print("   CARLA 데이터를 준비하면 도메인 적응 학습 가능")


def main():
    parser = argparse.ArgumentParser(description='Integrate CARLA and real data for domain adaptation')
    parser.add_argument('--carla_dir', type=str, default='datasets/carla_data',
                       help='CARLA data directory')
    parser.add_argument('--real_dir', type=str, default='datasets/real_data',
                       help='Real data directory')
    
    args = parser.parse_args()
    
    print_integration_guide(args.carla_dir, args.real_dir)


if __name__ == '__main__':
    main()








