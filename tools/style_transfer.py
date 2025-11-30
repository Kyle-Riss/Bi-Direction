"""
스타일 변환 유틸리티

CycleGAN 등을 통한 photorealism 증가
시뮬레이션 이미지를 실제 영상처럼 변환
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import glob


class CycleGAN_Generator(nn.Module):
    """
    CycleGAN Generator (간소화 버전)
    
    실제 구현은 별도 CycleGAN 모델이 필요하지만,
    여기서는 placeholder와 통합 방법 제공
    """
    def __init__(self):
        super(CycleGAN_Generator, self).__init__()
        # 실제 CycleGAN 구조는 복잡하므로 간소화
        # 실제로는 pretrained CycleGAN 모델 사용 권장
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Residual blocks (간소화)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Upsampling
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 3, kernel_size=7, padding=3),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Input: (batch, 3, H, W) in [-1, 1]
        # Output: (batch, 3, H, W) in [-1, 1]
        return self.conv_layers(x)


def apply_photorealism_transform(image_path, output_path, generator_path=None):
    """
    이미지에 photorealism 변환 적용
    
    Args:
        image_path: 입력 이미지 경로
        output_path: 출력 이미지 경로
        generator_path: CycleGAN generator 체크포인트 경로 (선택)
    """
    # 이미지 로드
    img = Image.open(image_path).convert('RGB')
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    
    # Generator (선택적)
    if generator_path and os.path.exists(generator_path):
        generator = CycleGAN_Generator()
        generator.load_state_dict(torch.load(generator_path, map_location='cpu'))
        generator.eval()
        
        with torch.no_grad():
            transformed = generator(img_tensor)
    else:
        # Generator 없으면 원본 반환 (또는 다른 변환 적용)
        transformed = img_tensor
    
    # 이미지로 변환
    transformed = (transformed.squeeze(0).cpu() + 1) / 2.0  # [0, 1]
    transformed = transforms.ToPILImage()(transformed)
    
    # 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    transformed.save(output_path, quality=95)
    
    return output_path


def batch_photorealism_transform(input_dir, output_dir, generator_path=None):
    """
    디렉토리 내 모든 이미지에 photorealism 변환 적용
    
    Args:
        input_dir: 입력 이미지 디렉토리
        output_dir: 출력 이미지 디렉토리
        generator_path: Generator 체크포인트 경로
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = glob.glob(os.path.join(input_dir, '*.jpg')) + \
                  glob.glob(os.path.join(input_dir, '*.png'))
    
    print(f"🔄 총 {len(image_files)}개 이미지 변환 시작...")
    
    for i, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, img_name)
        
        try:
            apply_photorealism_transform(img_path, output_path, generator_path)
            
            if (i + 1) % 100 == 0:
                print(f"   진행: {i+1}/{len(image_files)} ({i+1/len(image_files)*100:.1f}%)")
        
        except Exception as e:
            print(f"⚠️  오류 ({img_name}): {e}")
            continue
    
    print(f"\n✅ 변환 완료! 출력: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Apply photorealism transformation')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input image directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output image directory')
    parser.add_argument('--generator', type=str, default=None,
                       help='CycleGAN generator checkpoint path (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"❌ 입력 디렉토리 없음: {args.input_dir}")
        return
    
    batch_photorealism_transform(args.input_dir, args.output_dir, args.generator)


if __name__ == '__main__':
    main()








