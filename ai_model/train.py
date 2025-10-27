import torch
import os
import random
import numpy as np
from tqdm import tqdm  # 학습 진행률 표시
from torch.utils.data import DataLoader
from datetime import datetime
import copy

# 우리가 만든 커스텀 모듈 임포트
from ai_model.model import create_temporal_model, create_model
from ai_model.dataset import create_dataloader
from ai_model.config import get_config, get_device, get_optimizer, get_scheduler, validate_config

# --- 설정 검증 및 초기화 ---
validate_config()

# 랜덤 시드 설정 (YoloLSTM 스타일)
random_state = get_config("general", "random_state")
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------

def train_one_epoch(model, dataloader, optimizer, device, epoch_desc, loss_function=None):
    """표준 PyTorch 1-epoch 학습 루프 (YoloLSTM 스타일로 개선)"""
    model.train() # 모델을 학습 모드로 설정
    total_loss = 0.0
    
    # tqdm으로 진행률 표시
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_desc}")
    
    for images, targets in pbar:
        # 데이터를 디바이스로 이동
        images = images.to(device)
        targets = targets.to(device)
        
        # 그래디언트 초기화
        optimizer.zero_grad()
        
        # 모델 순전파
        if hasattr(model, 'forward') and 'targets' in model.forward.__code__.co_varnames:
            # YOLOv8 스타일 모델 (targets를 받는 경우)
            loss, _ = model(images, targets=targets)
        else:
            # 일반적인 모델 (output만 반환)
            outputs = model(images)
            if loss_function is not None:
                loss = loss_function(outputs, targets)
            else:
                # 기본 손실 함수 (MSE for regression)
                loss = torch.nn.functional.mse_loss(outputs, targets)
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)


def validate_model(model, dataloader, device, loss_function=None):
    """모델 검증 함수 (YoloLSTM 스타일)"""
    model.eval()
    total_loss = 0.0
    pred_list = []
    target_list = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            targets = targets.to(device)
            
            # 모델 순전파
            if hasattr(model, 'forward') and 'targets' in model.forward.__code__.co_varnames:
                loss, outputs = model(images, targets=targets)
            else:
                outputs = model(images)
                if loss_function is not None:
                    loss = loss_function(outputs, targets)
                else:
                    loss = torch.nn.functional.mse_loss(outputs, targets)
            
            total_loss += loss.item()
            pred_list.append(outputs)
            target_list.append(targets)
    
    # 전체 예측과 타겟을 합침
    if pred_list:
        pred_list = torch.cat(pred_list)
        target_list = torch.cat(target_list)
    
    return total_loss / len(dataloader), pred_list, target_list


def main():
    """메인 학습 함수 (YoloLSTM 스타일로 개선)"""
    # 현재 시간으로 출력 디렉토리 생성
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f"outputs/{now}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 디바이스 설정
    device = get_device()
    print(f"Using device: {device}")
    
    # 설정 가져오기
    general_config = get_config("general")
    data_config = get_config("data")
    image_config = get_config("image")
    model_config = get_config("model")
    training_config = get_config("training")
    
    # --- 1단계: Coarse Training (Hot-Start) ---
    print(f"--- 1. Starting Coarse Training (Image Size: {image_config['img_size_coarse']}) ---")
    
    # 1-1. 모델 생성
    model_coarse = create_temporal_model(
        num_frames=image_config['num_frames'], 
        ckpt_path=model_config['yolo_checkpoint']
    )
    model_coarse.to(device)
    
    # 1-2. 데이터 로더 생성
    train_loader_coarse = create_dataloader(
        img_dir=data_config['train_img_dir'],
        label_dir=data_config['train_label_dir'],
        batch_size=general_config['batch_size_coarse'],
        num_frames=image_config['num_frames'],
        img_size=image_config['img_size_coarse'],
        num_workers=general_config['num_workers'],
        shuffle=True
    )
    
    val_loader_coarse = create_dataloader(
        img_dir=data_config['val_img_dir'],
        label_dir=data_config['val_label_dir'],
        batch_size=general_config['batch_size_coarse'],
        num_frames=image_config['num_frames'],
        img_size=image_config['img_size_coarse'],
        num_workers=general_config['num_workers'],
        shuffle=False
    )
    
    # 1-3. 옵티마이저와 스케줄러 설정
    optimizer_coarse = get_optimizer(model_coarse, stage="coarse")
    scheduler_coarse = get_scheduler(optimizer_coarse)
    
    # 1-4. 학습 실행
    best_score = 0.0
    best_model_state = None
    
    for epoch in range(general_config['num_epochs_coarse']):
        # 학습
        train_loss = train_one_epoch(
            model_coarse, train_loader_coarse, optimizer_coarse, device,
            f"[COARSE {epoch+1}/{general_config['num_epochs_coarse']}]"
        )
        
        # 검증
        val_loss, pred_list, target_list = validate_model(model_coarse, val_loader_coarse, device)
        
        # 스케줄러 업데이트
        if scheduler_coarse:
            scheduler_coarse.step()
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 최고 성능 모델 저장
        if val_loss < best_score or best_score == 0:
            best_score = val_loss
            best_model_state = copy.deepcopy(model_coarse.state_dict())
    
    # 1-5. 최고 성능 모델 저장
    os.makedirs('models', exist_ok=True)
    torch.save(best_model_state, model_config['coarse_model_save_path'])
    print(f"Coarse model saved to {model_config['coarse_model_save_path']}")

    # 메모리 정리
    del model_coarse, train_loader_coarse, val_loader_coarse, optimizer_coarse
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    # --- 2단계: Fine-Tuning ---
    print(f"\n--- 2. Starting Fine-Tuning (Image Size: {image_config['img_size_fine']}) ---")
    
    # 2-1. 모델 생성 및 가중치 로드
    model_fine = create_temporal_model(
        num_frames=image_config['num_frames'], 
        ckpt_path=model_config['yolo_checkpoint']
    )
    model_fine.load_state_dict(torch.load(model_config['coarse_model_save_path']))
    model_fine.to(device)
    
    # 2-2. 고해상도 데이터 로더 생성
    train_loader_fine = create_dataloader(
        img_dir=data_config['train_img_dir'],
        label_dir=data_config['train_label_dir'],
        batch_size=general_config['batch_size_fine'],
        num_frames=image_config['num_frames'],
        img_size=image_config['img_size_fine'],
        num_workers=general_config['num_workers'],
        shuffle=True
    )
    
    val_loader_fine = create_dataloader(
        img_dir=data_config['val_img_dir'],
        label_dir=data_config['val_label_dir'],
        batch_size=general_config['batch_size_fine'],
        num_frames=image_config['num_frames'],
        img_size=image_config['img_size_fine'],
        num_workers=general_config['num_workers'],
        shuffle=False
    )
    
    # 2-3. 옵티마이저 설정 (더 작은 학습률)
    optimizer_fine = get_optimizer(model_fine, stage="fine")
    scheduler_fine = get_scheduler(optimizer_fine)

    # 2-4. 미세 조정 실행
    best_score_fine = 0.0
    best_model_state_fine = None
    
    for epoch in range(general_config['num_epochs_fine']):
        # 학습
        train_loss = train_one_epoch(
            model_fine, train_loader_fine, optimizer_fine, device,
            f"[FINE {epoch+1}/{general_config['num_epochs_fine']}]"
        )
        
        # 검증
        val_loss, pred_list, target_list = validate_model(model_fine, val_loader_fine, device)
        
        # 스케줄러 업데이트
        if scheduler_fine:
            scheduler_fine.step()
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 최고 성능 모델 저장
        if val_loss < best_score_fine or best_score_fine == 0:
            best_score_fine = val_loss
            best_model_state_fine = copy.deepcopy(model_fine.state_dict())
    
    # 2-5. 최종 모델 저장
    torch.save(best_model_state_fine, model_config['final_model_save_path'])
    torch.save(best_model_state_fine, model_config['best_model_save_path'])
    print(f"Final robust model saved to {model_config['final_model_save_path']}")
    print(f"Best model saved to {model_config['best_model_save_path']}")


def predict():
    """예측 함수 (YoloLSTM 스타일)"""
    device = get_device()
    model_config = get_config("model")
    data_config = get_config("data")
    image_config = get_config("image")
    general_config = get_config("general")
    
    # 모델 로드
    model = create_temporal_model(
        num_frames=image_config['num_frames'],
        ckpt_path=model_config['yolo_checkpoint']
    )
    model.load_state_dict(torch.load(model_config['best_model_save_path']))
    model.to(device)
    model.eval()
    
    # 테스트 데이터 로더 생성
    test_loader = create_dataloader(
        img_dir=data_config['test_img_dir'],
        label_dir=data_config['test_label_dir'],
        batch_size=general_config['batch_size_fine'],
        num_frames=image_config['num_frames'],
        img_size=image_config['img_size_fine'],
        num_workers=general_config['num_workers'],
        shuffle=False
    )
    
    # 예측 실행
    test_loss, pred_list, target_list = validate_model(model, test_loader, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Predictions shape: {pred_list.shape}")
    print(f"Targets shape: {target_list.shape}")
    
    return pred_list, target_list


if __name__ == "__main__":
    # Student 1이 데이터를 수집/분류한 후 이 스크립트를 실행합니다.
    # (실행 전: config.py의 데이터 경로를 꼭 확인하세요!)
    main()
    # predict()  # 예측을 원하면 주석 해제