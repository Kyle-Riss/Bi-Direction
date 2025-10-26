"""
모델 유틸리티 함수들
YoloLSTM 프로젝트의 helper 구조를 참고하여 작성
"""
import torch.nn as nn
import torch.optim as optim
from models.YoloLSTM import YoloLSTM
from models.TemporalYOLO import TemporalYOLO
from datasets.dataset_utils import load_temporal_yolo_data, load_temporal_yolo_test_data, load_yolo_lstm_data, load_yolo_lstm_test_data
from config import get_config


def get_models():
    """
    사용 가능한 모델들을 반환하는 함수
    YoloLSTM 프로젝트의 get_models 함수 구조를 참고
    """
    models_config = get_config("models")
    models = []
    
    for model_name, config in models_config.items():
        if not config.get("state", False):
            continue
            
        # 모델 생성
        if model_name == "YoloLSTM":
            model = YoloLSTM(config.get("param", {}))
        elif model_name == "TemporalYOLO":
            model = TemporalYOLO(config.get("param", {}))
        else:
            continue
        
        # 학습 설정 추가
        train_settings = {
            "data_loader_function": (load_yolo_lstm_data, load_yolo_lstm_test_data),
            "loss_function": nn.MSELoss(),
            "optimizer": optim.Adam,
            "eval_function": None,  # 필요시 추가
        }
        
        config["train_settings"] = train_settings
        models.append((model, config))
    
    return models


def get_model_by_name(model_name):
    """특정 이름의 모델을 반환하는 함수"""
    models = get_models()
    for model, config in models:
        if config["name"] == model_name:
            return model, config
    return None, None


def save_model(model, path, epoch=None, optimizer=None, loss=None):
    """모델을 저장하는 함수"""
    save_dict = {
        'model_state_dict': model.state_dict(),
    }
    
    if epoch is not None:
        save_dict['epoch'] = epoch
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    if loss is not None:
        save_dict['loss'] = loss
    
    torch.save(save_dict, path)


def load_model(model, path, optimizer=None):
    """모델을 로드하는 함수"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('loss', None)
