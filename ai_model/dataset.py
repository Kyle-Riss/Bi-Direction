import torch
import cv2
import os
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms.v2 as T
from ai_model.config import get_config

# Multiscale-CNN.pdf 논문에 따른 Coarsening(축소) 변환
#
def get_coarsening_transform(img_size=160, normalize=True):
    """지정된 크기로 이미지를 축소하는 'Coarsening' 변환을 반환합니다."""
    transforms_list = [
        T.ToPILImage(), # OpenCV/Numpy -> PIL
        T.Resize((img_size, img_size)),
        T.ToTensor(), # PIL -> Tensor (0-1 정규화 포함)
    ]
    
    if normalize:
        # ImageNet 정규화 (YoloLSTM 스타일)
        mean = get_config("image", "normalize_mean")
        std = get_config("image", "normalize_std")
        transforms_list.append(T.Normalize(mean=mean, std=std))
    
    return T.Compose(transforms_list)

class TemporalYOLODataset(Dataset):
    """
    연속된 프레임을 묶어(Frame Stacking) YOLO 모델에 전달하는 커스텀 데이터셋.
    'OpenCV Robust.pdf'의 Early Fusion 전략을 위한 데이터 로더입니다.
   
    """
    def __init__(self, img_dir, label_dir, num_frames=3, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.num_frames = num_frames
        self.transform = transform
        
        # 이미지 파일 목록을 스캔하고 정렬합니다.
        self.img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        
        # 연속된 프레임이 존재하지 않는 파일은 학습에서 제외합니다.
        # (예: 0, 1, 2번 프레임이 있어야 2번 인덱스를 학습에 사용 가능)
        self.valid_indices = self._find_valid_indices()
        
    def _find_valid_indices(self):
        valid_indices = []
        for i in range(len(self.img_files)):
            if i < self.num_frames - 1:
                continue # 이전 프레임이 부족하므로 스킵
            
            # TODO: 프레임이 실제로 연속적인지 확인하는 로직 (예: 'scene01_003.jpg', 'scene01_002.jpg'...)
            # 지금은 단순히 인덱스만으로 연속적이라고 가정합니다.
            
            valid_indices.append(i)
        
        print(f"Found {len(self.img_files)} images, {len(valid_indices)} valid target frames.")
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 1. 대상 인덱스(가장 최신 프레임)를 가져옵니다.
        target_file_index = self.valid_indices[idx]
        
        frame_stack = []
        
        # 2. num_frames 만큼 이전 프레임부터 순서대로 로드합니다.
        for i in range(self.num_frames):
            frame_index = target_file_index - (self.num_frames - 1) + i
            img_path = self.img_files[frame_index]
            
            # OpenCV로 이미지 로드 (PIL보다 빠름)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
            
            if self.transform:
                img = self.transform(img) # Coarsening (Resize + ToTensor)
            
            frame_stack.append(img)
            
        # 3. 프레임들을 하나의 텐서로 결합합니다 (C, H, W) -> (C*num_frames, H, W)
        # 예: (3, 160, 160) 3개를 -> (9, 160, 160) 1개로
        stacked_images = torch.cat(frame_stack, dim=0)

        # 4. 대상 프레임(가장 최신 프레임)의 라벨을 로드합니다.
        target_img_path = self.img_files[target_file_index]
        label_name = os.path.basename(target_img_path).replace('.jpg', '.txt')
        label_path = os.path.join(self.label_dir, label_name)
        
        # 라벨 로드 (YOLO 형식: [class_id, x_center, y_center, width, height])
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    labels.append(parts)
        
        # 라벨을 텐서로 변환
        # (나중에 collate_fn에서 배치 인덱스를 추가해야 함)
        targets = torch.tensor(labels, dtype=torch.float32)

        return stacked_images, targets

def create_dataloader(img_dir, label_dir, batch_size=16, num_frames=3, img_size=160, num_workers=4, shuffle=True, normalize=True):
    """
    TemporalYOLODataset을 위한 PyTorch DataLoader를 생성합니다.
    YoloLSTM의 데이터 로딩 구조를 참고하여 개선
    """
    transform = get_coarsening_transform(img_size=img_size, normalize=normalize)
    
    dataset = TemporalYOLODataset(
        img_dir=img_dir,
        label_dir=label_dir,
        num_frames=num_frames,
        transform=transform
    )
    
    # YoloLSTM 스타일의 DataLoader 생성
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=yolo_collate_fn,  # YOLO 라벨 처리를 위한 collate_fn
        drop_last=True,  # 마지막 불완전한 배치 제거
    )
    
    return data_loader

def yolo_collate_fn(batch):
    """
    YOLO 라벨(크기가 다름)을 배치로 묶고, 배치 인덱스를 추가합니다.
    YoloLSTM의 collate_fn 구조를 참고하여 개선
    """
    # 빈 배치 필터링 (YoloLSTM 스타일)
    filtered_batch = [data for data in batch if data[0] is not None and data[1] is not None]
    
    if len(filtered_batch) == 0:
        # 빈 배치인 경우
        return torch.zeros((0, 3, 160, 160)), torch.zeros((0, 6))
    
    stacked_images = []
    targets = []
    
    for i, (img, target) in enumerate(filtered_batch):
        stacked_images.append(img)
        
        if target.shape[0] > 0:
            # target 텐서의 맨 앞에 배치 인덱스(i)를 추가
            batch_index = torch.full((target.shape[0], 1), i, dtype=torch.float32)
            target_with_index = torch.cat([batch_index, target], dim=1)
            targets.append(target_with_index)

    # 이미지들을 하나의 배치 텐서로 묶음
    batch_images = torch.stack(stacked_images)
    
    # 라벨들을 하나의 배치 텐서로 묶음
    if len(targets) > 0:
        batch_targets = torch.cat(targets, dim=0)
    else:
        # 이 배치에 라벨이 하나도 없는 경우
        batch_targets = torch.zeros((0, 6), dtype=torch.float32) # (batch_idx, cls, x, y, w, h)

    return batch_images, batch_targets