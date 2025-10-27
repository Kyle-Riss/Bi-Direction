"""
극도로 가벼운 모델 구조
메모리 부족 상황에서도 실행 가능한 최소한의 모델
"""
import torch
import torch.nn as nn

class UltraLightYoloLSTM(nn.Module):
    """
    극도로 가벼운 YoloLSTM 모델
    메모리 사용량을 최소화한 구조
    """
    def __init__(self, num_frames=3, hidden_size=32, num_layers=1):
        super(UltraLightYoloLSTM, self).__init__()
        
        self.num_frames = num_frames
        self.hidden_size = hidden_size
        
        # 매우 작은 CNN 백본
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3 * num_frames, 8, kernel_size=3, padding=1),   # 9 -> 8 channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 -> 32
            
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 8 -> 16 channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 16 -> 32 channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
        )
        
        # 작은 LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=32 * 8 * 8,  # 32 * 8 * 8 = 2048
            hidden_size=hidden_size,   # 32
            num_layers=num_layers,     # 1
            batch_first=True,
            bidirectional=False,       # bidirectional 비활성화로 메모리 절약
            dropout=0.0                # dropout 비활성화
        )
        
        # 간단한 출력 레이어
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),  # 32 -> 16
            nn.ReLU(),
            nn.Linear(16, 2)  # x, y 좌표
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN 백본 통과
        cnn_out = self.cnn_backbone(x)  # (batch_size, 32, 8, 8)
        
        # LSTM을 위한 형태로 변환
        cnn_out = cnn_out.view(batch_size, -1)  # (batch_size, 32*8*8)
        cnn_out = cnn_out.unsqueeze(1)  # (batch_size, 1, 32*8*8)
        
        # LSTM 통과
        lstm_out, (h_n, c_n) = self.lstm(cnn_out)
        
        # 마지막 출력 사용
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # 최종 출력
        output = self.fc(lstm_out)
        
        return output

def get_ultra_light_config():
    """극도로 가벼운 설정"""
    return {
        "img_size_coarse": 64,      # 128 -> 64 (메모리 1/4 절약)
        "img_size_fine": 128,       # 320 -> 128 (메모리 1/4 절약)
        "batch_size_coarse": 2,     # 8 -> 2 (메모리 1/4 절약)
        "batch_size_fine": 1,       # 2 -> 1 (메모리 1/2 절약)
        "epochs_coarse": 10,        # 30 -> 10 (빠른 테스트)
        "epochs_fine": 3,           # 5 -> 3 (빠른 테스트)
        "num_workers": 0,           # 멀티프로세싱 비활성화
        "hidden_size": 32,          # 더 작은 LSTM
        "num_layers": 1,            # 1층 LSTM만 사용
    }

def estimate_memory_usage():
    """메모리 사용량 추정"""
    config = get_ultra_light_config()
    
    # 이미지 크기별 메모리 사용량 계산
    coarse_memory = config["batch_size_coarse"] * config["img_size_coarse"] ** 2 * 3 * 3  # batch * H * W * C * frames
    fine_memory = config["batch_size_fine"] * config["img_size_fine"] ** 2 * 3 * 3
    
    print("💾 Memory Usage Estimation:")
    print(f"  Coarse stage: {coarse_memory / (1024**2):.1f} MB")
    print(f"  Fine stage: {fine_memory / (1024**2):.1f} MB")
    print(f"  Total estimated: {(coarse_memory + fine_memory) / (1024**2):.1f} MB")

if __name__ == "__main__":
    estimate_memory_usage()
    
    # 모델 크기 확인
    model = UltraLightYoloLSTM()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Parameters: {total_params:,} ({total_params/1000:.1f}K)")
    
    # 테스트 입력으로 메모리 사용량 확인
    test_input = torch.randn(1, 9, 64, 64)  # batch=1, channels=9, H=64, W=64
    output = model(test_input)
    print(f"✅ Model test successful: {output.shape}")

