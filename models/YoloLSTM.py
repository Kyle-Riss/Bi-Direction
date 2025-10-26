"""
YoloLSTM 모델 구현
YoloLSTM 프로젝트의 YoloLSTM.py 구조를 참고하여 작성
"""
import torch
import torch.nn as nn


class YoloLSTM(nn.Module):
    """
    YoloLSTM 모델
    CNN + LSTM 조합으로 temporal context를 활용한 객체 탐지 모델
    """
    def __init__(self, param=None):
        super(YoloLSTM, self).__init__()
        
        self.param = param or {}
        num_frames = self.param.get('num_frames', 3)
        hidden_size = self.param.get('hidden_size', 256)
        num_layers = self.param.get('num_layers', 2)
        
        # CNN 백본
        self.cnn = nn.Sequential(
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
        
        # LSTM 레이어
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
        cnn_out = self.cnn(x)  # (batch_size, 128, 8, 8)
        
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
