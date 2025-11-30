"""
간단한 혼합 학습: FastCut 데이터 + 실제 도로 데이터

FastCut이 이미 도메인 적응을 해줬으므로,
그냥 섞어서 일반 YOLO 학습만 하면 됨!
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import os
import argparse
from pathlib import Path
import sys
from tqdm import tqdm

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# HuggingFace datasets 패키지와 충돌 방지
import importlib.util

# fastcut_dataset import
fastcut_dataset_spec = importlib.util.spec_from_file_location(
    "fastcut_dataset",
    project_root / "datasets" / "fastcut_dataset.py"
)
fastcut_dataset = importlib.util.module_from_spec(fastcut_dataset_spec)
sys.modules["datasets.fastcut_dataset"] = fastcut_dataset
fastcut_dataset_spec.loader.exec_module(fastcut_dataset)
FastCutDataset = fastcut_dataset.FastCutDataset

# dataset import
dataset_spec = importlib.util.spec_from_file_location(
    "dataset",
    project_root / "datasets" / "dataset.py"
)
dataset_module = importlib.util.module_from_spec(dataset_spec)
sys.modules["datasets.dataset"] = dataset_module
dataset_spec.loader.exec_module(dataset_module)
TemporalYOLODataset = dataset_module.TemporalYOLODataset

from config import get_config
from models.TemporalYoloGRU_v2 import TemporalYoloGRU_v2
from helper.yolo_loss import compute_yolo_loss_simple


class SimpleMixedDataset(torch.utils.data.Dataset):
    """
    real_A (원본 시뮬레이션), fake_B (FastCut 변환 결과), test/ (실제 도로) 데이터를 섞어서 사용하는 간단한 데이터셋
    
    중요:
    - real_A: 원본 시뮬레이션 (소스)
    - real_B: 실제 도로 이미지 (real world, 별도 데이터셋, 스타일 참조용)
    - fake_B: real_A와 real_B를 합쳐서 나온 결과 (FastCut 변환)
              → real_A를 real_B 스타일로 변환한 것
              → fake_B와 real_A는 같은 라벨 사용 (타겟 지점이 맞음)
    - test/: 실제 도로 이미지 (라벨 있음)
    """
    def __init__(self, real_A_dir, fake_B_dir, labels_dir,
                 real_img_dir, real_label_dir,
                 img_size=128, num_frames=3, transform=None):
        self.num_frames = num_frames
        self.img_size = img_size
        
        # transform이 None이면 기본 transform 사용
        if transform is None:
            from datasets.dataset import get_coarsening_transform
            transform = get_coarsening_transform(img_size=img_size, normalize=True)
        
        # real_A 데이터셋 (원본 시뮬레이션, 소스)
        self.real_A_dataset = TemporalYOLODataset(
            img_dir=real_A_dir,
            label_dir=labels_dir,  # real_A가 원본이므로 라벨은 여기 있음
            num_frames=num_frames,
            transform=transform
        )
        
        # fake_B 데이터셋 (real_A를 real_B 스타일로 변환한 결과, 타겟 지점은 맞음)
        self.fake_B_dataset = TemporalYOLODataset(
            img_dir=fake_B_dir,
            label_dir=labels_dir,  # fake_B는 real_A를 변환한 것이므로 같은 라벨 사용
            num_frames=num_frames,
            transform=transform
        )
        
        # 실제 도로 데이터셋 (test/)
        self.real_dataset = TemporalYOLODataset(
            img_dir=real_img_dir,
            label_dir=real_label_dir,
            num_frames=num_frames,
            transform=transform
        )
        
        # 데이터셋 합치기 전략
        # Option 1: fake_B + test/ 만 사용 (실제 도로 스타일만, 권장) ✅
        # Option 2: real_A + fake_B + test/ 모두 사용 (다양성)
        
        # fake_B + test/ 만 사용 (실제 도로 스타일 중심)
        use_real_A = False  # False: fake_B + test/만 사용 (권장)
        
        if use_real_A:
            self.combined_dataset = ConcatDataset([
                self.real_A_dataset,  # 원본 시뮬레이션 (선택적)
                self.fake_B_dataset,  # FastCut 변환 결과 (실제 도로 스타일)
                self.real_dataset     # 실제 도로
            ])
            print(f"✅ 데이터셋 준비 완료 (real_A 포함):")
            print(f"   - real_A (원본 시뮬레이션): {len(self.real_A_dataset)}개")
            print(f"   - fake_B (FastCut 변환, 실제 도로 스타일): {len(self.fake_B_dataset)}개")
            print(f"   - 실제 도로 (test/): {len(self.real_dataset)}개")
        else:
            # fake_B + test/ 만 사용 (실제 도로 스타일 중심, 권장)
            self.combined_dataset = ConcatDataset([
                self.fake_B_dataset,  # FastCut 변환 결과 (실제 도로 스타일)
                self.real_dataset     # 실제 도로
            ])
            print(f"✅ 데이터셋 준비 완료 (fake_B + test/ 중심):")
            print(f"   - fake_B (FastCut 변환, 실제 도로 스타일): {len(self.fake_B_dataset)}개")
            print(f"   - 실제 도로 (test/): {len(self.real_dataset)}개")
        
        print(f"   - 총합: {len(self.combined_dataset)}개")
    
    def __len__(self):
        return len(self.combined_dataset)
    
    def __getitem__(self, idx):
        return self.combined_dataset[idx]


def train_epoch(model, dataloader, optimizer, device, epoch):
    """간단한 학습 에폭"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        # TemporalYoloGRU_v2는 targets를 받지 않으므로, 일반 forward만
        outputs = model(images)
        
        # 모델 출력 형식에 따라 처리
        if isinstance(outputs, tuple):
            # (temporal_output, yolo_output) 형식인 경우
            temporal_output, yolo_output = outputs
            outputs = temporal_output
        
        # YOLO Loss 계산
        # outputs: (batch_size, 5*85) = (batch_size, 425)
        # targets: (num_targets, 6) = (batch_idx, class, x, y, w, h)
        try:
            loss = compute_yolo_loss_simple(outputs, targets, model)
        except Exception as e:
            # YOLO loss 계산 실패 시, 폴백으로 간단한 loss 사용
            print(f"⚠️ YOLO loss 계산 실패: {e}, 폴백 loss 사용")
            if len(targets) > 0:
                # 간단한 MSE loss (폴백)
                target_coords = targets[:, 2:4]  # x, y 좌표만 추출
                if outputs.shape[-1] == 425:
                    outputs_coords = outputs[:, :2]  # 첫 2개 값 사용
                    loss = nn.MSELoss()(outputs_coords, target_coords)
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='간단한 혼합 학습 (FastCut + 실제 도로)')
    parser.add_argument('--real_A_dir', type=str, required=True,
                       help='real_A 이미지 디렉토리 (원본 시뮬레이션, 소스)')
    parser.add_argument('--fake_B_dir', type=str, required=True,
                       help='fake_B 이미지 디렉토리 (real_A를 real_B 스타일로 변환한 결과)')
    parser.add_argument('--fastcut_labels_dir', type=str, required=True,
                       help='FastCut 라벨 디렉토리 (real_A와 fake_B 공통)')
    parser.add_argument('--real_img_dir', type=str, default='data/real_world/test',
                       help='실제 도로 이미지 디렉토리 (기본: data/real_world/test)')
    parser.add_argument('--real_label_dir', type=str, default='data/real_world/test/labels',
                       help='실제 도로 라벨 디렉토리 (기본: data/real_world/test/labels)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='학습률')
    parser.add_argument('--img_size', type=int, default=128,
                       help='이미지 크기')
    parser.add_argument('--num_frames', type=int, default=3,
                       help='Temporal frame 수')
    parser.add_argument('--device', type=str, default='mps',
                       help='디바이스 (mps, cuda, cpu)')
    parser.add_argument('--save_path', type=str, default='models/simple_mixed_model.pt',
                       help='모델 저장 경로')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device(args.device if torch.backends.mps.is_available() and args.device == 'mps' 
                         else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 디바이스: {device}")
    
    # 데이터셋 생성
    print("\n📦 데이터셋 준비 중...")
    dataset = SimpleMixedDataset(
        real_A_dir=args.real_A_dir,
        fake_B_dir=args.fake_B_dir,
        labels_dir=args.fastcut_labels_dir,  # 인자 이름 수정
        real_img_dir=args.real_img_dir,
        real_label_dir=args.real_label_dir,
        img_size=args.img_size,
        num_frames=args.num_frames
    )
    
    # collate_fn import
    yolo_collate_fn = dataset_module.yolo_collate_fn
    
    dataloader = DataLoader(
        dataset.combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # macOS에서 multiprocessing 이슈 방지
        pin_memory=False,
        collate_fn=yolo_collate_fn  # YOLO 형식 라벨 처리
    )
    
    # 모델 생성
    print("\n🤖 모델 생성 중...")
    model = TemporalYoloGRU_v2(
        num_frames=args.num_frames,
        yolo_checkpoint='yolov8n.pt',
        gru_hidden_size=256,
        gru_num_layers=2,
        gru_bidirectional=True,
        feature_size=1024
    )
    model = model.to(device)
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//2, gamma=0.1)
    
    # 학습 루프
    print(f"\n🎓 학습 시작 (총 {args.epochs} 에폭)...")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, device, epoch)
        scheduler.step()
        
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")
        
        # 최고 성능 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"✅ 모델 저장: {args.save_path} (Loss: {avg_loss:.4f})")
        
        print("-" * 60)
    
    print(f"\n🎉 학습 완료!")
    print(f"   최종 Loss: {best_loss:.4f}")
    print(f"   모델 저장: {args.save_path}")


if __name__ == '__main__':
    main()

