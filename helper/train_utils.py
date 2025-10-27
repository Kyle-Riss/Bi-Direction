"""
학습 관련 유틸리티 함수들
Mixed Precision 지원과 효율적인 학습 구조로 개선
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import copy


def train_one_epoch(model, dataloader, optimizer, device, epoch_desc, scaler=None):
    """표준 PyTorch 1-epoch 학습 루프 (Mixed Precision 지원 추가)"""
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_desc}")

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        # Mixed Precision 비활성화 (MPS에서는 제한적 지원)
        # 모델 순전파
        outputs = model(images)
        
        # 간단한 MSE 손실 (임시)
        if targets.shape[0] > 0:
            # targets에서 배치 인덱스 제거하고 좌표만 사용
            target_coords = targets[:, 2:4]  # x, y 좌표만 추출
            loss = nn.MSELoss()(outputs, target_coords)
        else:
            # 빈 배치인 경우 더미 손실
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        if scaler: # Mixed Precision 사용 시
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: # 일반 학습 시
            loss.backward()
            optimizer.step()

        optimizer.zero_grad() # 그래디언트 초기화 중요!

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def validate_model(model, dataloader, device, scaler=None):
    """모델 검증 함수 (Mixed Precision 지원)"""
    model.eval()
    total_loss = 0.0
    pred_list = []
    target_list = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            targets = targets.to(device)
            
            # Mixed Precision 지원
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                loss, outputs = model(images, targets=targets)
            
            total_loss += loss.item()
            pred_list.append(outputs)
            target_list.append(targets)
    
    # 전체 예측과 타겟을 합침
    if pred_list:
        pred_list = torch.cat(pred_list)
        target_list = torch.cat(target_list)
    
    return total_loss / len(dataloader), pred_list, target_list


def train_coarse_stage(model, train_loader, val_loader, optimizer, device, num_epochs, scheduler=None, scaler=None):
    """Coarse 학습 단계 실행 (Mixed Precision 지원)"""
    best_score = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 학습
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            f"[COARSE {epoch+1}/{num_epochs}]", scaler
        )
        
        # 검증
        val_loss, pred_list, target_list = validate_model(model, val_loader, device, scaler)
        
        # 스케줄러 업데이트
        if scheduler:
            scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 최고 성능 모델 저장
        if val_loss < best_score:
            best_score = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
    
    return best_model_state, best_score


def train_fine_stage(model, train_loader, val_loader, optimizer, device, num_epochs, scheduler=None, scaler=None):
    """Fine 학습 단계 실행 (Mixed Precision 지원)"""
    best_score = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 학습
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            f"[FINE {epoch+1}/{num_epochs}]", scaler
        )
        
        # 검증
        val_loss, pred_list, target_list = validate_model(model, val_loader, device, scaler)
        
        # 스케줄러 업데이트
        if scheduler:
            scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 최고 성능 모델 저장
        if val_loss < best_score:
            best_score = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
    
    return best_model_state, best_score


def predict_model(model, test_loader, device, scaler=None):
    """모델 예측 함수 (Mixed Precision 지원)"""
    model.eval()
    test_loss = 0.0
    pred_list = []
    target_list = []
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            targets = targets.to(device)
            
            # Mixed Precision 지원
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                loss, outputs = model(images, targets=targets)
            
            test_loss += loss.item()
            pred_list.append(outputs)
            target_list.append(targets)
    
    test_loss = test_loss / len(test_loader)
    
    if pred_list:
        pred_list = torch.cat(pred_list)
        target_list = torch.cat(target_list)
    
    return test_loss, pred_list, target_list
