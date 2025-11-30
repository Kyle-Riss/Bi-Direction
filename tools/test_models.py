"""
두 모델 비교 테스트 스크립트
- Baseline: robust_model.pt
- Current: simple_mixed_model.pt
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 모듈 import
import importlib.util

# dataset import
dataset_spec = importlib.util.spec_from_file_location(
    "dataset",
    project_root / "datasets" / "dataset.py"
)
dataset_module = importlib.util.module_from_spec(dataset_spec)
sys.modules["datasets.dataset"] = dataset_module
dataset_spec.loader.exec_module(dataset_module)
TemporalYOLODataset = dataset_module.TemporalYOLODataset
yolo_collate_fn = dataset_module.yolo_collate_fn

from models.model import create_model
from models.TemporalYoloGRU_v2 import TemporalYoloGRU_v2
from helper.yolo_loss import compute_yolo_loss_simple


def test_model(model, dataloader, device, model_name="Model"):
    """모델 테스트 및 성능 측정"""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    num_samples = 0
    
    print(f"\n📊 {model_name} 테스트 중...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc=f"Testing {model_name}")):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # 모델 출력 형식에 따라 처리
            if isinstance(outputs, tuple):
                temporal_output, yolo_output = outputs
                outputs = temporal_output
            
            # Loss 계산
            if len(targets) > 0:
                try:
                    loss = compute_yolo_loss_simple(outputs, targets, model)
                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    print(f"⚠️ Loss 계산 실패: {e}")
            
            num_samples += images.size(0)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    print(f"\n✅ {model_name} 결과:")
    print(f"   - 평균 Loss: {avg_loss:.4f}")
    print(f"   - 테스트 샘플 수: {num_samples}개")
    print(f"   - 배치 수: {num_batches}개")
    
    return {
        'model_name': model_name,
        'avg_loss': avg_loss,
        'num_samples': num_samples,
        'num_batches': num_batches
    }


def load_baseline_model(device):
    """Baseline 모델 로드"""
    print("\n📦 Baseline 모델 로드 중...")
    print("   파일: models/robust_model.pt")
    
    # TemporalYoloGRU 모델 생성 (baseline은 이 모델 사용)
    from models.TemporalYoloGRU import TemporalYoloGRU
    
    model = TemporalYoloGRU(
        num_frames=3,
        yolo_checkpoint='yolov8n.pt',
        gru_hidden_size=256,
        gru_num_layers=2,
        gru_bidirectional=True,
        feature_size=1024
    )
    
    try:
        model.load_state_dict(torch.load('models/robust_model.pt', map_location=device))
        model = model.to(device)
        print("   ✅ Baseline 모델 로드 완료")
        return model
    except Exception as e:
        print(f"   ❌ Baseline 모델 로드 실패: {e}")
        return None


def load_current_model(device):
    """Current 모델 로드"""
    print("\n📦 Current 모델 로드 중...")
    print("   파일: models/simple_mixed_model.pt")
    
    model = TemporalYoloGRU_v2(
        num_frames=3,
        yolo_checkpoint='yolov8n.pt',
        gru_hidden_size=256,
        gru_num_layers=2,
        gru_bidirectional=True,
        feature_size=1024
    )
    
    try:
        state_dict = torch.load('models/simple_mixed_model.pt', map_location=device)
        # feature_proj 키가 있으면 제거 (모델 구조에 없을 수 있음)
        if 'feature_proj.weight' in state_dict:
            print("   ⚠️ feature_proj 키 발견, 제거 중...")
            state_dict = {k: v for k, v in state_dict.items() if 'feature_proj' not in k}
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        print("   ✅ Current 모델 로드 완료")
        return model
    except Exception as e:
        print(f"   ❌ Current 모델 로드 실패: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='두 모델 비교 테스트')
    parser.add_argument('--test_img_dir', type=str, default='test',
                       help='테스트 이미지 디렉토리')
    parser.add_argument('--test_label_dir', type=str, default='test/labels',
                       help='테스트 라벨 디렉토리')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='배치 크기')
    parser.add_argument('--img_size', type=int, default=128,
                       help='이미지 크기')
    parser.add_argument('--num_frames', type=int, default=3,
                       help='Temporal frame 수')
    parser.add_argument('--device', type=str, default='mps',
                       help='디바이스 (mps, cuda, cpu)')
    parser.add_argument('--baseline_only', action='store_true',
                       help='Baseline 모델만 테스트')
    parser.add_argument('--current_only', action='store_true',
                       help='Current 모델만 테스트')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"🔧 디바이스: {device}")
    
    # 테스트 데이터셋 준비
    print(f"\n📦 테스트 데이터셋 준비 중...")
    print(f"   이미지: {args.test_img_dir}")
    print(f"   라벨: {args.test_label_dir}")
    
    from datasets.dataset import get_coarsening_transform
    
    transform = get_coarsening_transform(img_size=args.img_size, normalize=True)
    
    test_dataset = TemporalYOLODataset(
        img_dir=args.test_img_dir,
        label_dir=args.test_label_dir,
        num_frames=args.num_frames,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=yolo_collate_fn
    )
    
    print(f"   ✅ 테스트 데이터: {len(test_dataset)}개")
    
    results = []
    
    # Baseline 모델 테스트
    if not args.current_only:
        baseline_model = load_baseline_model(device)
        if baseline_model is not None:
            baseline_result = test_model(baseline_model, test_loader, device, "Baseline (robust_model.pt)")
            results.append(baseline_result)
    
    # Current 모델 테스트
    if not args.baseline_only:
        current_model = load_current_model(device)
        if current_model is not None:
            current_result = test_model(current_model, test_loader, device, "Current (simple_mixed_model.pt)")
            results.append(current_result)
    
    # 결과 비교
    if len(results) == 2:
        print("\n" + "="*60)
        print("📊 모델 비교 결과")
        print("="*60)
        
        baseline = results[0]
        current = results[1]
        
        print(f"\nBaseline 모델:")
        print(f"   Loss: {baseline['avg_loss']:.4f}")
        print(f"\nCurrent 모델:")
        print(f"   Loss: {current['avg_loss']:.4f}")
        
        loss_diff = current['avg_loss'] - baseline['avg_loss']
        loss_improvement = (loss_diff / baseline['avg_loss']) * 100 if baseline['avg_loss'] > 0 else 0
        
        print(f"\n차이:")
        if loss_diff < 0:
            print(f"   ✅ Current 모델이 {abs(loss_diff):.4f} 더 낮은 Loss (개선: {abs(loss_improvement):.2f}%)")
        elif loss_diff > 0:
            print(f"   ⚠️ Current 모델이 {loss_diff:.4f} 더 높은 Loss (악화: {loss_improvement:.2f}%)")
        else:
            print(f"   ➡️ Loss 동일")
    
    print("\n✅ 테스트 완료!")


if __name__ == '__main__':
    main()

