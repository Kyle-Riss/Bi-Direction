"""
Apple Silicon MPS 최적화 도구
MPS 백엔드의 성능을 최대한 활용하는 설정
"""
import torch
import os
import gc

def optimize_mps_settings():
    """MPS 백엔드 최적화 설정"""
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available")
        return False
    
    print("🍎 Apple Silicon MPS Optimization")
    print("=" * 40)
    
    # 1. MPS 메모리 관리 최적화
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # 메모리 사용량 최소화
    os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'
    
    # 2. MPS 캐시 최적화
    os.environ['PYTORCH_MPS_ALLOCATOR'] = 'native'  # 네이티브 할당자 사용
    
    # 3. 메모리 압박 상황에서의 최적화
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'  # 작은 청크로 분할
    
    # 4. MPS 백엔드 설정 확인
    print(f"✅ MPS Available: {torch.backends.mps.is_available()}")
    print(f"✅ MPS Built: {torch.backends.mps.is_built()}")
    
    # 5. 메모리 정리
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    print("✅ MPS optimization applied!")
    return True

def get_mps_optimized_dataloader_kwargs():
    """MPS에 최적화된 DataLoader 설정"""
    return {
        'batch_size': 4,           # 작은 배치 크기 (메모리 절약)
        'num_workers': 0,           # MPS에서는 싱글 스레드가 더 안정적
        'pin_memory': False,        # MPS에서는 지원 안됨
        'persistent_workers': False, # 메모리 절약
        'prefetch_factor': 1,       # 프리페치 최소화
    }

def mps_memory_cleanup():
    """MPS 메모리 정리"""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("🧹 MPS memory cleaned")

def get_mps_device():
    """MPS 디바이스 반환"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def create_mps_optimized_model(model_class, **kwargs):
    """MPS에 최적화된 모델 생성"""
    device = get_mps_device()
    model = model_class(**kwargs)
    model = model.to(device)
    
    # MPS 메모리 정리
    mps_memory_cleanup()
    
    return model, device

def mps_training_step(model, images, targets, optimizer, device):
    """MPS에 최적화된 학습 스텝"""
    # 메모리 정리
    mps_memory_cleanup()
    
    # 데이터를 MPS로 이동
    images = images.to(device, non_blocking=False)  # MPS는 non_blocking 지원 안됨
    targets = targets.to(device, non_blocking=False)
    
    # 순전파
    outputs = model(images)
    
    # 손실 계산
    if targets.shape[0] > 0:
        target_coords = targets[:, 2:4]
        loss = torch.nn.MSELoss()(outputs, target_coords)
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # 역전파
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # 메모리 정리
    mps_memory_cleanup()
    
    return loss.item()

def check_mps_performance():
    """MPS 성능 테스트"""
    if not torch.backends.mps.is_available():
        print("❌ MPS not available for testing")
        return
    
    device = torch.device("mps")
    
    # 간단한 성능 테스트
    print("🧪 Testing MPS performance...")
    
    # 테스트 텐서 생성
    x = torch.randn(100, 1000, device=device)
    y = torch.randn(1000, 100, device=device)
    
    # 행렬 곱셈 테스트
    import time
    start_time = time.time()
    
    for _ in range(100):
        z = torch.mm(x, y)
    
    end_time = time.time()
    
    print(f"✅ MPS Matrix multiplication: {end_time - start_time:.4f}s")
    print(f"✅ Memory usage: {torch.mps.current_allocated_memory() / 1024**2:.1f} MB")

if __name__ == "__main__":
    optimize_mps_settings()
    check_mps_performance()
    
    print("\n📋 MPS Optimized DataLoader Settings:")
    kwargs = get_mps_optimized_dataloader_kwargs()
    for key, value in kwargs.items():
        print(f"  {key}: {value}")

