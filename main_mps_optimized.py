"""
MPS 최적화된 학습 스크립트
Apple Silicon에서 최대 성능을 위한 설정
"""
import torch
import os
import gc
from tools.mps_optimizer import (
    optimize_mps_settings, 
    get_mps_optimized_dataloader_kwargs,
    mps_training_step,
    mps_memory_cleanup,
    get_mps_device
)

# 모듈 임포트
from config import get_config
from models.model import create_model
from datasets.dataset_utils import load_temporal_yolo_data
from helper.train_utils import train_one_epoch

def mps_optimized_train_one_epoch(model, dataloader, optimizer, device, epoch_desc):
    """MPS 최적화된 학습 루프"""
    model.train()
    total_loss = 0.0
    
    print(f"Epoch {epoch_desc}")
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        # MPS 최적화된 학습 스텝
        loss = mps_training_step(model, images, targets, optimizer, device)
        total_loss += loss
        
        # 진행 상황 출력 (매 10 배치마다)
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: Loss = {loss:.4f}")
    
    return total_loss / len(dataloader)

def main():
    print("🍎 Apple Silicon MPS Optimized Training")
    print("=" * 50)
    
    # MPS 최적화 적용
    optimize_mps_settings()
    
    # 설정 로드
    cfg_gen = get_config("general")
    cfg_data = get_config("data")
    cfg_ms = get_config("multiscale")
    cfg_wandb = get_config("wandb")
    
    # WandB 초기화 (선택)
    if cfg_wandb["use"]:
        try:
            import wandb
            wandb.init(
                project=cfg_wandb["project"],
                name=cfg_wandb["run_name"],
                config={**cfg_gen, **cfg_data, **cfg_ms}
            )
        except ImportError:
            print("⚠️ WandB not installed. Continuing without logging...")
            cfg_wandb["use"] = False
    
    # MPS 디바이스 설정
    device = get_mps_device()
    print(f"Using device: {device}")
    
    # Coarse Training
    print(f"--- Starting Coarse Training (Size: {cfg_ms['coarse']['img_size']}) ---")
    cfg_coarse = cfg_ms['coarse']
    
    # 모델 생성 (MPS 최적화)
    model = create_model("yolo_lstm", num_frames=cfg_gen["num_frames"], hidden_size=256, num_layers=2)
    model = model.to(device)
    
    # MPS 최적화된 DataLoader 설정
    mps_dataloader_kwargs = get_mps_optimized_dataloader_kwargs()
    mps_dataloader_kwargs['batch_size'] = cfg_coarse["batch_size"]
    
    # 데이터 로더 생성 (MPS 최적화 설정 적용)
    train_loader, _ = load_temporal_yolo_data(
        cfg_coarse["batch_size"], 
        cfg_gen["num_workers"], 
        cfg_gen["random_state"]
    )
    
    # 옵티마이저 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_coarse["lr"])
    
    # 학습 루프
    for epoch in range(cfg_coarse["epochs"]):
        avg_loss = mps_optimized_train_one_epoch(
            model, train_loader, optimizer, device,
            f"[COARSE {epoch+1}/{cfg_coarse['epochs']}]"
        )
        
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
        
        # WandB 로깅
        if cfg_wandb["use"]:
            try:
                import wandb
                wandb.log({"coarse_loss": avg_loss, "coarse_epoch": epoch+1})
            except ImportError:
                pass
        
        # 메모리 정리
        mps_memory_cleanup()
    
    # 모델 저장
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), cfg_coarse["save_path"])
    print(f"Coarse model saved to {cfg_coarse['save_path']}")
    
    # Fine-Tuning
    print(f"\n--- Starting Fine-Tuning (Size: {cfg_ms['fine']['img_size']}) ---")
    cfg_fine = cfg_ms['fine']
    
    # 모델 가중치 로드
    model.load_state_dict(torch.load(cfg_coarse["save_path"]))
    model = model.to(device)
    
    # Fine-tuning 데이터 로더
    train_loader, _ = load_temporal_yolo_data(
        cfg_fine["batch_size"], 
        cfg_gen["num_workers"], 
        cfg_gen["random_state"]
    )
    
    # 옵티마이저 재정의
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_fine["lr"])
    
    # Fine-tuning 학습 루프
    for epoch in range(cfg_fine["epochs"]):
        avg_loss = mps_optimized_train_one_epoch(
            model, train_loader, optimizer, device,
            f"[FINE {epoch+1}/{cfg_fine['epochs']}]"
        )
        
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
        
        # WandB 로깅
        if cfg_wandb["use"]:
            try:
                import wandb
                wandb.log({
                    "fine_loss": avg_loss, 
                    "fine_epoch": epoch+1,
                    "total_epoch": cfg_coarse["epochs"] + epoch + 1
                })
            except ImportError:
                pass
        
        # 메모리 정리
        mps_memory_cleanup()
    
    # 최종 모델 저장
    torch.save(model.state_dict(), cfg_fine["save_path"])
    print(f"Final robust model saved to {cfg_fine['save_path']}")
    
    # WandB 종료
    if cfg_wandb["use"]:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass
    
    print("✅ Training completed!")

if __name__ == "__main__":
    main()

