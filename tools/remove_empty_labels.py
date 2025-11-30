"""
빈 레이블 파일을 가진 이미지를 train 디렉토리에서 제거하는 스크립트

사용법:
    python tools/remove_empty_labels.py --split-dir carla_datasetv2/realA_split
"""
import argparse
import os
from pathlib import Path


def remove_empty_label_images(split_dir: str, dry_run: bool = False):
    """
    빈 레이블 파일을 가진 이미지를 train에서 제거
    
    Args:
        split_dir: split 디렉토리 경로 (예: carla_datasetv2/realA_split)
        dry_run: 실제로 제거하지 않고 확인만
    """
    split_path = Path(split_dir)
    train_images_dir = split_path / "train" / "images"
    train_labels_dir = split_path / "train" / "labels"
    
    if not train_images_dir.exists():
        print(f"❌ Train images 디렉토리를 찾을 수 없습니다: {train_images_dir}")
        return
    
    if not train_labels_dir.exists():
        print(f"❌ Train labels 디렉토리를 찾을 수 없습니다: {train_labels_dir}")
        return
    
    # 이미지 파일 찾기
    image_files = list(train_images_dir.glob("*.png")) + list(train_images_dir.glob("*.jpg"))
    empty_count = 0
    removed_count = 0
    
    for img_file in image_files:
        label_file = train_labels_dir / f"{img_file.stem}.txt"
        
        # 레이블 파일이 없거나 빈 경우
        if not label_file.exists() or label_file.stat().st_size == 0:
            empty_count += 1
            if not dry_run:
                # 심볼릭 링크인 경우
                if img_file.is_symlink():
                    img_file.unlink()
                else:
                    img_file.unlink()
                
                if label_file.exists():
                    if label_file.is_symlink():
                        label_file.unlink()
                    else:
                        label_file.unlink()
                
                removed_count += 1
            else:
                print(f"[DRY RUN] 제거 예정: {img_file.name}")
    
    print(f"\n=== 결과 ===")
    print(f"빈 레이블 이미지: {empty_count}개")
    if not dry_run:
        print(f"제거된 이미지: {removed_count}개")
        print(f"남은 이미지: {len(image_files) - removed_count}개")
    else:
        print(f"[DRY RUN] 실제로 제거하려면 --dry-run 플래그를 제거하세요")


def main():
    parser = argparse.ArgumentParser(description="빈 레이블 이미지를 train에서 제거")
    parser.add_argument(
        "--split-dir",
        type=str,
        required=True,
        help="Split 디렉토리 경로 (예: carla_datasetv2/realA_split)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제로 제거하지 않고 확인만",
    )
    
    args = parser.parse_args()
    
    remove_empty_label_images(args.split_dir, args.dry_run)


if __name__ == "__main__":
    main()



