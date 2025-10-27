import torch
import torch.nn as nn
from ultralytics import YOLO
from ai_model.config import get_config

def create_temporal_model(num_frames=3, ckpt_path='yolov8n.pt'):
    """
    YOLOv8 모델을 로드하고 첫 번째 레이어를 수정하여
    여러 프레임을 동시에 입력받도록(Temporal Context) 개조합니다.
    
    이 방식은 'OpenCV Robust.pdf' 논문의 Early Fusion 전략을 따릅니다.
    YoloLSTM의 LSTM 사용법을 참고하여 개선
    
    Args:
        num_frames (int): 한 번에 입력으로 사용할 프레임 수 (예: 3)
        ckpt_path (str): 사용할 YOLOv8 체크포인트 (예: 'yolov8n.pt')
        
    Returns:
        model (nn.Module): 첫 번째 레이어가 수정된 PyTorch 모델
    """
    
    # 1. YOLOv8 모델을 로드하되, PyTorch nn.Module 자체를 가져옵니다.
    model = YOLO(ckpt_path).model.model
    
    print(f"Original model loaded. First layer input channels: {model[0].conv.in_channels}")

    # 2. 첫 번째 컨볼루션 레이어(conv)의 기존 파라미터를 백업합니다.
    try:
        first_conv = model[0].conv  # 첫 번째 CBS(Conv-BN-SiLU) 블록의 conv
        original_weights = first_conv.weight.data
        out_channels = first_conv.out_channels
        kernel_size = first_conv.kernel_size
        stride = first_conv.stride
        padding = first_conv.padding
        bias = first_conv.bias
    except Exception as e:
        print(f"Error accessing first conv layer: {e}")
        print("YOLOv8 모델 구조가 변경되었을 수 있습니다. model[0].conv를 확인하세요.")
        return None

    # 3. 새 입력 채널 수에 맞춰 새 컨볼루션 레이어를 정의합니다.
    # (예: 3 프레임 * 3 채널(RGB) = 9 채널)
    new_in_channels = 3 * num_frames
    
    new_conv = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=(bias is not None)
    )

    # 4. 새 레이어의 가중치를 'Early Fusion' 방식으로 초기화합니다.
    #
    # 기존 3채널 가중치를 n번 반복하고, 1/n로 스케일링합니다.
    with torch.no_grad():
        # 3채널(RGB) 가중치를 num_frames 만큼 복사
        new_weights = original_weights.repeat(1, num_frames, 1, 1)
        # 스케일링 (활성화 값의 크기를 유지하기 위함)
        new_weights = new_weights * (1.0 / num_frames)
        
        new_conv.weight.data = new_weights
        
        if bias is not None:
            new_conv.bias.data = bias.data

    # 5. 모델의 첫 번째 레이어를 우리가 만든 새 레이어로 교체합니다!
    model[0].conv = new_conv
    
    # 6. (중요) 모델의 내부 설정(yaml)에도 입력 채널이 바뀌었다고 알려줍니다.
    model.yaml['ch'] = new_in_channels
    
    print(f"Successfully modified model! New first layer input channels: {new_conv.in_channels}")
    
    return model


class YoloLSTM(nn.Module):
    """
    YoloLSTM 프로젝트의 YoloLSTM 모델 구조를 참고한 클래스
    CNN + LSTM 조합으로 temporal context를 활용
    """
    def __init__(self, num_frames=3, hidden_size=256, num_layers=2):
        super(YoloLSTM, self).__init__()
        
        self.num_frames = num_frames
        self.hidden_size = hidden_size
        
        # CNN 백본 (YOLOv8의 일부 레이어 사용)
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
        
        # LSTM 레이어 (YoloLSTM 스타일)
        self.lstm = nn.LSTM(
            input_size=128 * 8 * 8,  # CNN 출력 크기
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # 출력 레이어
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),  # bidirectional이므로 *2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # x, y 좌표
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels * num_frames, height, width)
        Returns:
            output: (batch_size, 2) - x, y 좌표
        """
        batch_size = x.size(0)
        
        # CNN 백본 통과
        cnn_out = self.cnn_backbone(x)  # (batch_size, 128, 8, 8)
        
        # LSTM을 위한 형태로 변환
        cnn_out = cnn_out.view(batch_size, -1)  # (batch_size, 128*8*8)
        cnn_out = cnn_out.unsqueeze(1)  # (batch_size, 1, 128*8*8)
        
        # LSTM 통과
        lstm_out, (h_n, c_n) = self.lstm(cnn_out)
        
        # 마지막 출력 사용
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size*2)
        
        # 최종 출력
        output = self.fc(lstm_out)
        
        return output


class TemporalYoloLSTM(nn.Module):
    """
    YOLOv8 + LSTM 조합 모델
    YOLOv8의 detection 능력과 LSTM의 temporal modeling을 결합
    """
    def __init__(self, num_frames=3, yolo_checkpoint='yolov8n.pt'):
        super(TemporalYoloLSTM, self).__init__()
        
        self.num_frames = num_frames
        
        # YOLOv8 백본 (수정된 버전)
        self.yolo_backbone = create_temporal_model(num_frames, yolo_checkpoint)
        
        # LSTM 레이어 추가
        self.lstm = nn.LSTM(
            input_size=1024,  # YOLOv8의 feature 크기에 맞춤
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # 출력 레이어
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels * num_frames, height, width)
        Returns:
            output: (batch_size, 2) - x, y 좌표
        """
        # YOLOv8 백본 통과 (detection features 추출)
        yolo_features = self.yolo_backbone(x)
        
        # LSTM을 위한 형태로 변환
        batch_size = yolo_features.size(0)
        yolo_features = yolo_features.view(batch_size, -1).unsqueeze(1)
        
        # LSTM 통과
        lstm_out, _ = self.lstm(yolo_features)
        lstm_out = lstm_out[:, -1, :]  # 마지막 출력
        
        # 최종 출력
        output = self.fc(lstm_out)
        
        return output


def create_model(model_type="temporal_yolo", **kwargs):
    """
    모델 타입에 따라 적절한 모델을 생성하는 팩토리 함수
    YoloLSTM의 모델 생성 패턴을 참고
    """
    if model_type == "temporal_yolo":
        return create_temporal_model(**kwargs)
    elif model_type == "yolo_lstm":
        return YoloLSTM(**kwargs)
    elif model_type == "temporal_yolo_lstm":
        return TemporalYoloLSTM(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")