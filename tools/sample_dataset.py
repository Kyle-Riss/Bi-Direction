"""
데이터셋 샘플링 스크립트 - 학습 속도 개선을 위해 데이터셋 크기 축소

사용법:
    python tools/sample_dataset.py \
        --input-dir carla_datasetv2/realB_split \
        --output-dir carla_datasetv2/realB_split_small \
        --ratio 0.3 \
        --mode symlink
"""
import argparse
import os
import random
import shutil
from pathlib import Path


def sample_dataset(
    input_dir: str,
    output_dir: str,
    ratio: float = 0.3,
    mode: str = "symlink",
    seed: int = 42,
):
    """
    데이터셋을 샘플링하여 더 작은 서브셋 생성
    
    Args:
        input_dir: 원본 데이터셋 디렉토리 (train/val/test 구조)
        output_dir: 출력 디렉토리
        ratio: 샘플링 비율 (0.0 ~ 1.0)
        mode: 'symlink' (심볼릭 링크) 또는 'copy' (복사)
        seed: 랜덤 시드
    """
    random.seed(seed)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"입력 디렉토리를 찾을 수 없습니다: {input_dir}")
    
    print(f"📦 데이터셋 샘플링 시작")
    print(f"   입력: {input_dir}")
    print(f"   출력: {output_dir}")
    print(f"   샘플링 비율: {ratio*100:.1f}%")
    print(f"   모드: {mode}")
    print()
    
    # train/val/test 각각 처리
    for split in ["train", "val", "test"]:
        split_input = input_path / split
        if not split_input.exists():
            print(f"⚠️  {split} 디렉토리가 없습니다. 스킵합니다.")
            continue
        
        split_output = output_path / split
        images_input = split_input / "images"
        labels_input = split_input / "labels"
        images_output = split_output / "images"
        labels_output = split_output / "labels"
        
        if not images_input.exists():
            print(f"⚠️  {split}/images 디렉토리가 없습니다. 스킵합니다.")
            continue
        
        # 이미지 파일 목록 가져오기
        image_files = sorted(list(images_input.glob("*.png")) + list(images_input.glob("*.jpg")))
        if len(image_files) == 0:
            print(f"⚠️  {split}/images에 이미지가 없습니다. 스킵합니다.")
            continue
        
        # 샘플링
        num_samples = max(1, int(len(image_files) * ratio))
        sampled_files = random.sample(image_files, num_samples)
        
        print(f"   {split}: {len(image_files)}개 → {num_samples}개 샘플링")
        
        # 출력 디렉토리 생성
        images_output.mkdir(parents=True, exist_ok=True)
        if labels_input.exists():
            labels_output.mkdir(parents=True, exist_ok=True)
        
        # 파일 복사 또는 심볼릭 링크 생성
        copied = 0
        for img_file in sampled_files:
            img_name = img_file.name
            img_stem = img_file.stem
            
            # 이미지 처리
            img_dst = images_output / img_name
            if mode == "symlink":
                if img_dst.exists() or img_dst.is_symlink():
                    img_dst.unlink()
                img_dst.symlink_to(img_file.resolve())
            else:  # copy
                shutil.copy2(img_file, img_dst)
            
            # 레이블 처리
            if labels_input.exists():
                label_file = labels_input / f"{img_stem}.txt"
                if label_file.exists():
                    label_dst = labels_output / f"{img_stem}.txt"
                    if mode == "symlink":
                        if label_dst.exists() or label_dst.is_symlink():
                            label_dst.unlink()
                        label_dst.symlink_to(label_file.resolve())
                    else:  # copy
                        shutil.copy2(label_file, label_dst)
            
            copied += 1
        
        print(f"      ✅ {copied}개 파일 처리 완료")
    
    print()
    print(f"✅ 샘플링 완료: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="데이터셋 샘플링으로 학습 속도 개선"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="원본 데이터셋 디렉토리 (train/val/test 구조)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="샘플링된 데이터셋 출력 디렉토리",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.3,
        help="샘플링 비율 (0.0 ~ 1.0, 기본값: 0.3)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="symlink",
        choices=["symlink", "copy"],
        help="파일 생성 모드: symlink (기본값) 또는 copy",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (기본값: 42)",
    )
    
    args = parser.parse_args()
    
    if args.ratio <= 0 or args.ratio > 1:
        raise ValueError("--ratio는 0.0과 1.0 사이의 값이어야 합니다.")
    
    sample_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ratio=args.ratio,
        mode=args.mode,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()



