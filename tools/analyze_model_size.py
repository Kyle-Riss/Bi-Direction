"""
모델 크기 및 파라미터 분석 스크립트

현재 사용 중인 모델들의 파라미터 수, 메모리 사용량을 분석하고
경량화 옵션을 제안합니다.
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from models.YoloLSTM import YoloLSTM
from models.ultra_light_model import UltraLightYoloLSTM
from models.model import TemporalYoloLSTM, create_temporal_model


def count_parameters(model):
    """모델의 파라미터 수 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def estimate_memory(model, batch_size=1, img_size=128, num_frames=3):
    """모델의 메모리 사용량 추정 (MB)"""
    # 입력 크기
    input_size = batch_size * 3 * num_frames * img_size * img_size * 4  # float32 = 4 bytes
    
    # 모델 파라미터 크기
    param_size = sum(p.numel() * 4 for p in model.parameters())  # float32 = 4 bytes
    
    # 순전파 메모리 (대략적으로 파라미터 크기의 2배 가정)
    forward_size = param_size * 2
    
    # 총 메모리 (MB)
    total_mb = (input_size + param_size + forward_size) / (1024 ** 2)
    
    return total_mb, param_size / (1024 ** 2)


def analyze_model(model_name, model, batch_size=1, img_size=128, num_frames=3):
    """모델 분석"""
    total_params, trainable_params = count_parameters(model)
    
    # 메모리 추정
    total_mb, param_mb = estimate_memory(model, batch_size, img_size, num_frames)
    
    print(f"\n{'='*60}")
    print(f"📊 {model_name}")
    print(f"{'='*60}")
    print(f"총 파라미터: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"학습 가능 파라미터: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"파라미터 크기: {param_mb:.2f} MB")
    print(f"예상 메모리 사용량 (batch={batch_size}, img={img_size}): {total_mb:.2f} MB")
    
    # 모델 구조 정보
    if hasattr(model, 'hidden_size'):
        print(f"LSTM hidden_size: {model.hidden_size}")
    if hasattr(model, 'num_frames'):
        print(f"입력 프레임 수: {model.num_frames}")
    
    return {
        'name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_mb': param_mb,
        'total_mb': total_mb
    }


def compare_all_models():
    """모든 모델 비교"""
    print("🔍 모델 경량화 분석")
    print("="*60)
    
    results = []
    
    # 1. 현재 YoloLSTM (기본)
    print("\n1️⃣  YoloLSTM (현재 기본 설정)")
    model1 = YoloLSTM(param={
        'num_frames': 3,
        'hidden_size': 256,
        'num_layers': 2
    })
    r1 = analyze_model("YoloLSTM (hidden=256, layers=2, bidirectional)", model1, 
                       batch_size=8, img_size=128)
    results.append(r1)
    
    # 2. 경량화된 YoloLSTM
    print("\n2️⃣  YoloLSTM (경량화 버전)")
    model2 = YoloLSTM(param={
        'num_frames': 3,
        'hidden_size': 128,  # 256 -> 128
        'num_layers': 1       # 2 -> 1
    })
    r2 = analyze_model("YoloLSTM (hidden=128, layers=1, bidirectional)", model2,
                       batch_size=8, img_size=128)
    results.append(r2)
    
    # 3. UltraLightYoloLSTM
    print("\n3️⃣  UltraLightYoloLSTM")
    model3 = UltraLightYoloLSTM(num_frames=3, hidden_size=32, num_layers=1)
    r3 = analyze_model("UltraLightYoloLSTM (hidden=32, layers=1)", model3,
                       batch_size=2, img_size=64)
    results.append(r3)
    
    # 4. YOLOv8n 기반 TemporalYoloLSTM
    print("\n4️⃣  TemporalYoloLSTM (YOLOv8n 기반)")
    try:
        model4 = TemporalYoloLSTM(num_frames=3, yolo_checkpoint='yolov8n.pt')
        r4 = analyze_model("TemporalYoloLSTM (YOLOv8n + LSTM)", model4,
                           batch_size=8, img_size=128)
        results.append(r4)
    except Exception as e:
        print(f"   ⚠️  모델 로드 실패: {e}")
    
    # 비교 요약
    print(f"\n{'='*60}")
    print("📊 모델 비교 요약")
    print(f"{'='*60}")
    print(f"{'모델':<40} {'파라미터':<15} {'메모리(MB)':<15}")
    print("-"*70)
    for r in results:
        print(f"{r['name']:<40} {r['total_params']/1e6:>7.2f}M{'':<5} {r['total_mb']:>7.2f}")
    
    # 경량화 추천
    print(f"\n💡 경량화 추천:")
    results_sorted = sorted(results, key=lambda x: x['total_params'])
    lightest = results_sorted[0]
    print(f"   가장 경량: {lightest['name']} ({lightest['total_params']/1e6:.2f}M 파라미터)")
    
    reduction = (r1['total_params'] - lightest['total_params']) / r1['total_params'] * 100
    print(f"   현재 대비 {reduction:.1f}% 파라미터 감소")


def create_lightweight_models():
    """경량화된 모델 변형 생성"""
    print("\n" + "="*60)
    print("🛠️  경량화 모델 옵션 생성")
    print("="*60)
    
    # GRU 기반 모델 (LSTM 대비 더 경량)
    class YoloGRU(nn.Module):
        """YoloLSTM의 GRU 버전 (LSTM 대비 파라미터 약 25% 감소)"""
        def __init__(self, num_frames=3, hidden_size=256, num_layers=2):
            super(YoloGRU, self).__init__()
            self.num_frames = num_frames
            self.hidden_size = hidden_size
            
            # CNN 백본 (YoloLSTM과 동일)
            self.cnn_backbone = nn.Sequential(
                nn.Conv2d(3 * num_frames, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            
            # GRU 레이어 (LSTM 대신)
            self.gru = nn.GRU(
                input_size=128 * 8 * 8,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.2
            )
            
            # 출력 레이어
            self.fc = nn.Sequential(
                nn.Linear(hidden_size * 2, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 2)
            )
        
        def forward(self, x):
            batch_size = x.size(0)
            cnn_out = self.cnn_backbone(x)
            cnn_out = cnn_out.view(batch_size, -1).unsqueeze(1)
            gru_out, _ = self.gru(cnn_out)
            gru_out = gru_out[:, -1, :]
            output = self.fc(gru_out)
            return output
    
    # GRU 모델 분석
    model_gru = YoloGRU(hidden_size=256, num_layers=2)
    r_gru = analyze_model("YoloGRU (hidden=256, layers=2, bidirectional)", model_gru,
                          batch_size=8, img_size=128)
    
    # LSTM과 비교
    model_lstm = YoloLSTM(param={'num_frames': 3, 'hidden_size': 256, 'num_layers': 2})
    r_lstm = analyze_model("YoloLSTM (hidden=256, layers=2, bidirectional)", model_lstm,
                           batch_size=8, img_size=128)
    
    reduction = (r_lstm['total_params'] - r_gru['total_params']) / r_lstm['total_params'] * 100
    print(f"\n💡 LSTM → GRU 전환 시 파라미터 {reduction:.1f}% 감소")


if __name__ == '__main__':
    # 모든 모델 비교
    compare_all_models()
    
    # 경량화 모델 생성 및 분석
    create_lightweight_models()
    
    print("\n" + "="*60)
    print("✅ 분석 완료!")
    print("="*60)








