"""
Bi-Direction 프로젝트 메인 실행 파일
간단하고 효율적인 구조로 개선
"""
import torch
import os
# import wandb # WandB 추가 (선택사항)

# 모듈 임포트
from config import get_config
from models.model import create_model
from datasets.dataset_utils import load_temporal_yolo_data
from helper.train_utils import train_one_epoch

def main():
    # --- 0. 설정 로드 ---
    cfg_gen = get_config("general")
    cfg_data = get_config("data")
    cfg_ms = get_config("multiscale")
    cfg_wandb = get_config("wandb")

    # --- WandB 초기화 (선택) ---
    if cfg_wandb["use"]:
        try:
            import wandb
            wandb.init(
                project=cfg_wandb["project"],
                name=cfg_wandb["run_name"],
                config={**cfg_gen, **cfg_data, **cfg_ms} # 모든 설정을 WandB에 기록
            )
        except ImportError:
            print("⚠️ WandB not installed. Continuing without logging...")
            cfg_wandb["use"] = False

    # --- 1. 디바이스 설정 ---
    scaler = None  # scaler 초기화
    if cfg_gen["device"] == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif cfg_gen["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        # Mixed Precision을 위한 Scaler (CUDA에서만)
        scaler = torch.cuda.amp.GradScaler()
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- 2. Coarse Training (Hot-Start) ---
    print(f"--- Starting Coarse Training (Size: {cfg_ms['coarse']['img_size']}) ---")
    cfg_coarse = cfg_ms['coarse']

    model = create_model("yolo_lstm", num_frames=cfg_gen["num_frames"], hidden_size=256, num_layers=2)
    model.to(device)

    train_loader, _ = load_temporal_yolo_data(
        cfg_coarse["batch_size"], 
        cfg_gen["num_workers"], 
        cfg_gen["random_state"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_coarse["lr"])

    for epoch in range(cfg_coarse["epochs"]):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device,
                                   f"[COARSE {epoch+1}/{cfg_coarse['epochs']}]", scaler)
        if cfg_wandb["use"]:
            try:
                import wandb
                wandb.log({"coarse_loss": avg_loss, "coarse_epoch": epoch+1}) # WandB 로깅
            except ImportError:
                pass

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), cfg_coarse["save_path"])
    print(f"Coarse model saved to {cfg_coarse['save_path']}")

    # --- 3. Fine-Tuning ---
    print(f"\n--- Starting Fine-Tuning (Size: {cfg_ms['fine']['img_size']}) ---")
    cfg_fine = cfg_ms['fine']

    # 모델 가중치만 로드 (구조는 이미 동일)
    model.load_state_dict(torch.load(cfg_coarse["save_path"]))
    model.to(device) # 다시 device로 보내야 할 수도 있음

    train_loader, _ = load_temporal_yolo_data(
        cfg_fine["batch_size"], 
        cfg_gen["num_workers"], 
        cfg_gen["random_state"]
    )
    # 옵티마이저 재정의 (더 작은 LR)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_fine["lr"])

    for epoch in range(cfg_fine["epochs"]):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device,
                                   f"[FINE {epoch+1}/{cfg_fine['epochs']}]", scaler)
        if cfg_wandb["use"]:
            try:
                import wandb
                wandb.log({"fine_loss": avg_loss, "fine_epoch": epoch+1,
                           "total_epoch": cfg_coarse["epochs"] + epoch + 1}) # WandB 로깅
            except ImportError:
                pass

    torch.save(model.state_dict(), cfg_fine["save_path"])
    print(f"Final robust model saved to {cfg_fine['save_path']}")

    if cfg_wandb["use"]:
        try:
            import wandb
            wandb.finish() # WandB 종료
        except ImportError:
            pass


if __name__ == "__main__":
    main()
