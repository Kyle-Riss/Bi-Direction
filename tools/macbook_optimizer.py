"""
맥북 최적화 설정
Apple Silicon MPS와 메모리 효율성을 위한 설정
"""
import torch
import os

def get_optimized_config():
    """맥북에 최적화된 설정 반환"""
    return {
        # 메모리 효율성을 위한 설정
        "batch_size_coarse": 8,      # 기존 32 -> 8로 감소
        "batch_size_fine": 2,        # 기존 4 -> 2로 감소
        "num_workers": 2,           # 기존 4 -> 2로 감소 (메모리 절약)
        
        # MPS 최적화
        "pin_memory": False,         # MPS에서는 지원 안됨
        "persistent_workers": False, # 메모리 절약
        
        # Mixed Precision 대안
        "use_amp": False,           # MPS에서는 제한적 지원
        "gradient_checkpointing": True,  # 메모리 절약
        
        # 이미지 크기 조정
        "img_size_coarse": 128,     # 기존 160 -> 128로 감소
        "img_size_fine": 320,       # 기존 640 -> 320로 감소
        
        # 학습 설정
        "epochs_coarse": 30,        # 기존 50 -> 30으로 감소
        "epochs_fine": 5,           # 기존 10 -> 5로 감소
    }

def optimize_for_macbook():
    """맥북 최적화 설정 적용"""
    # PyTorch 스레드 수 조정
    torch.set_num_threads(4)  # CPU 코어 수에 맞춤
    
    # MPS 최적화
    if torch.backends.mps.is_available():
        print("🍎 Using Apple Silicon MPS acceleration")
        # MPS 메모리 관리 최적화
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    # 메모리 효율성 설정
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    print("✅ MacBook optimization applied!")

def get_memory_efficient_dataloader_kwargs():
    """메모리 효율적인 DataLoader 설정"""
    return {
        'batch_size': 4,           # 작은 배치 크기
        'num_workers': 2,           # 적은 워커 수
        'pin_memory': False,        # MPS에서는 비활성화
        'persistent_workers': False, # 메모리 절약
        'prefetch_factor': 2,       # 프리페치 감소
    }

if __name__ == "__main__":
    config = get_optimized_config()
    print("🍎 MacBook Optimized Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    optimize_for_macbook()

