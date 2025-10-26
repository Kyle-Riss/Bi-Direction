"""
Temporal YOLO 모델 구현
YOLOv8을 기반으로 한 temporal context 모델
"""
import torch
import torch.nn as nn
from ultralytics import YOLO
# from config import get_config  # 필요시 주석 해제


class TemporalYOLO(nn.Module):
    """
    Temporal YOLO 모델
    YOLOv8을 기반으로 여러 프레임을 동시에 처리하는 모델
    """
    def __init__(self, param=None):
        super(TemporalYOLO, self).__init__()
        
        self.param = param or {}
        num_frames = self.param.get('num_frames', 3)
        ckpt_path = self.param.get('ckpt_path', 'yolov8n.pt')
        
        # YOLOv8 백본 로드
        self.yolo_model = YOLO(ckpt_path).model.model
        
        # 첫 번째 레이어 수정 (temporal context를 위해)
        self._modify_first_layer(num_frames)
        
    def _modify_first_layer(self, num_frames):
        """첫 번째 레이어를 temporal context에 맞게 수정"""
        try:
            first_conv = self.yolo_model[0].conv
            original_weights = first_conv.weight.data
            out_channels = first_conv.out_channels
            kernel_size = first_conv.kernel_size
            stride = first_conv.stride
            padding = first_conv.padding
            bias = first_conv.bias
            
            # 새 입력 채널 수에 맞춰 새 컨볼루션 레이어 정의
            new_in_channels = 3 * num_frames
            
            new_conv = nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=(bias is not None)
            )
            
            # 가중치 초기화 (Early Fusion 방식)
            with torch.no_grad():
                new_weights = original_weights.repeat(1, num_frames, 1, 1)
                new_weights = new_weights * (1.0 / num_frames)
                new_conv.weight.data = new_weights
                
                if bias is not None:
                    new_conv.bias.data = bias.data
            
            # 첫 번째 레이어 교체
            self.yolo_model[0].conv = new_conv
            self.yolo_model.yaml['ch'] = new_in_channels
            
        except Exception as e:
            print(f"Error modifying first layer: {e}")
            raise
    
    def forward(self, x, targets=None):
        """
        Args:
            x: (batch_size, channels * num_frames, height, width)
            targets: (optional) YOLO targets
        Returns:
            loss, outputs or just outputs
        """
        if targets is not None:
            # YOLOv8 스타일 (targets가 있는 경우)
            return self.yolo_model(x, targets=targets)
        else:
            # 일반적인 순전파
            return self.yolo_model(x)
