"""
Bi-Direction 프로젝트 설정 파일
간단하고 명확한 구조로 개선
"""

# --- 일반 설정 ---
GENERAL = {
    "num_frames": 3,         # 입력 프레임 수
    "yolo_ckpt": 'yolov8n.pt', # 사용할 YOLO 사전학습 모델
    "device": "mps",         # 학습에 사용할 디바이스 ('cuda', 'mps', 'cpu')
    "num_workers": 2,        # 데이터 로딩 워커 수 (맥북 최적화)
    "random_state": 42,      # 랜덤 시드
}

# --- 데이터 경로 ---
DATA = {
    "train_img_dir": 'data/train/images',   # BDD100K 비디오에서 추출된 프레임들
    "train_label_dir": 'data/train/labels', # YOLO 라벨 파일들 (비어있음 - 테스트용)
    "val_img_dir": 'data/val/images',       # 검증용 프레임들
    "val_label_dir": 'data/val/labels',     # 검증용 라벨 파일들 (비어있음 - 테스트용)
    "test_img_dir": 'data/test/images',     # 테스트용 프레임들
    "test_label_dir": 'data/test/labels',   # 테스트용 라벨 파일들 (비어있음 - 테스트용)
}

# --- Multiscale 학습 설정 (맥북 최적화) ---
MULTISCALE = {
    "coarse": {
        "img_size": 128,        # 160 -> 128 (메모리 절약)
        "batch_size": 8,        # 32 -> 8 (메모리 절약)
        "epochs": 30,           # 50 -> 30 (빠른 테스트)
        "lr": 1e-4,
        "save_path": 'models/coarse_hotstart.pt'
    },
    "fine": {
        "img_size": 320,        # 640 -> 320 (메모리 절약)
        "batch_size": 2,        # 4 -> 2 (메모리 절약)
        "epochs": 5,            # 10 -> 5 (빠른 테스트)
        "lr": 1e-5,
        "save_path": 'models/robust_model.pt'
    }
}

# --- (선택) WandB 로깅 설정 ---
WANDB = {
    "use": True,           # Weights & Biases 사용 여부
    "project": "Bi-Direction-YOLO",
    "run_name": None       # None이면 자동 생성
}

# --- 모델 설정 ---
MODELS = {
    "YoloLSTM": {
        "state": True,
        "name": "YoloLSTM",
        "param": {
            "num_frames": 3,
            "hidden_size": 256,
            "num_layers": 2
        }
    },
    "TemporalYOLO": {
        "state": True,
        "name": "TemporalYOLO",
        "param": {
            "num_frames": 3,
            "ckpt_path": "yolov8n.pt"
        }
    }
}

# 설정 값을 가져오는 함수 (sakusaku3939 참고)
def get_config(*keys):
    config = {
        "general": GENERAL,
        "data": DATA,
        "multiscale": MULTISCALE,
        "wandb": WANDB,
        "models": MODELS
    }
    for key in keys:
        config = config[key]
    return config

def get_device():
    """디바이스 설정을 자동으로 결정하는 함수"""
    import torch
    device_setting = GENERAL["device"]
    
    if device_setting == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_setting)

def get_optimizer(model, stage="coarse"):
    """옵티마이저를 생성하는 함수"""
    import torch.optim as optim
    
    if stage == "coarse":
        lr = MULTISCALE["coarse"]["lr"]
    else:
        lr = MULTISCALE["fine"]["lr"]
    
    return optim.Adam(model.parameters(), lr=lr)

def get_scheduler(optimizer, stage="coarse"):
    """학습률 스케줄러를 생성하는 함수"""
    import torch.optim as optim
    
    if stage == "coarse":
        epochs = MULTISCALE["coarse"]["epochs"]
    else:
        epochs = MULTISCALE["fine"]["epochs"]
    
    return optim.lr_scheduler.StepLR(optimizer, step_size=epochs//2, gamma=0.1)

# 설정 검증 함수
def validate_config():
    """설정 값들이 유효한지 검증하는 함수"""
    assert MULTISCALE["coarse"]["epochs"] > 0, "coarse epochs must be positive"
    assert MULTISCALE["fine"]["epochs"] > 0, "fine epochs must be positive"
    assert MULTISCALE["coarse"]["batch_size"] > 0, "coarse batch_size must be positive"
    assert MULTISCALE["fine"]["batch_size"] > 0, "fine batch_size must be positive"
    assert MULTISCALE["coarse"]["img_size"] > 0, "coarse img_size must be positive"
    assert MULTISCALE["fine"]["img_size"] > 0, "fine img_size must be positive"
    assert GENERAL["num_frames"] > 0, "num_frames must be positive"
    assert MULTISCALE["coarse"]["lr"] > 0, "coarse lr must be positive"
    assert MULTISCALE["fine"]["lr"] > 0, "fine lr must be positive"
    
    print("✅ Configuration validation passed!")

if __name__ == "__main__":
    validate_config()
    print("Configuration loaded successfully!")
