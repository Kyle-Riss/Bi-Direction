"""
Ultralytics YOLOv8 FastCut fine-tuning script.

이 스크립트는 carla_datasetv2/fake_B (FastCut 변환) 이미지를 사용하여
YOLOv8 모델을 간단히 파인튜닝 합니다. 기존 temporal/GRU 모델 대신
순수 YOLO 파이프라인으로 빠르게 도메인 적응을 돌리고 싶은 경우 사용하세요.
"""
import argparse
import os

import torch
from ultralytics import YOLO


def resolve_device(requested: str) -> str:
    """Return a device string that Ultralytics YOLO가 이해할 수 있는 형태."""
    requested = requested.lower()
    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        print("⚠️  MPS를 사용할 수 없어 CPU로 대체합니다.")
        requested = "cpu"
    if requested in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return "0"  # Ultralytics는 GPU 인덱스를 문자열로 받음
        print("⚠️  CUDA를 사용할 수 없어 CPU로 대체합니다.")
        requested = "cpu"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="FastCut 데이터로 Ultralytics YOLOv8 도메인 적응 파인튜닝"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="carla_fastcut.yaml",
        help="YOLO dataset yaml 경로",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="기존 YOLO 가중치 경로 (사전학습 체크포인트)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="학습 epoch 수",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="배치 크기",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="입력 이미지 크기 (정사각형)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="초기 학습률",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="훈련 디바이스 (mps/cuda/cpu)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/fastcut",
        help="Ultralytics 결과 저장 루트",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolov8_fastcut",
        help="실험 이름 (project 내 하위 폴더)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="조기 종료 patience (Ultralytics 기본값 50)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"데이터셋 YAML을 찾을 수 없습니다: {args.data}\n"
            "carla_datasetv2/fastcut.yaml 혹은 원하는 경로를 지정하세요."
        )

    # 체크포인트 경로 확인
    checkpoint_dir = os.path.join(args.project, args.name, "weights")
    last_pt = os.path.join(checkpoint_dir, "last.pt")
    resume_checkpoint = last_pt if os.path.exists(last_pt) else None

    print("============================================================")
    print("🚗 FastCut → YOLOv8 도메인 적응 파인튜닝 시작")
    print("============================================================")
    print(f"• 데이터셋: {args.data}")
    if resume_checkpoint:
        print(f"• 체크포인트에서 재개: {resume_checkpoint}")
    else:
        print(f"• 초기 가중치: {args.weights}")
    print(f"• Epochs / Batch / Img: {args.epochs} / {args.batch} / {args.imgsz}")
    print(f"• Device: {device}")
    print(f"• 결과 저장: {os.path.join(args.project, args.name)}")
    print("============================================================\n")

    # 체크포인트가 있으면 그걸 사용, 없으면 초기 가중치 사용
    weights_path = resume_checkpoint if resume_checkpoint else args.weights
    model = YOLO(weights_path)
    
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "lr0": args.lr,
        "device": device,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "pretrained": not resume_checkpoint,  # 체크포인트에서 재개하면 pretrained=False
        "exist_ok": True,
        "workers": 0,  # macOS multiprocessing 이슈 방지 및 메모리 절약
        "cache": False,  # 캐시 비활성화로 메모리 절약
        "amp": True,  # Mixed precision으로 속도 향상 (MPS 지원 시)
        "val": True,  # Validation 활성화
        "plots": False,  # 플롯 생성 비활성화로 메모리 절약
        "save": True,  # 모델 저장
        "max_det": 100,  # 최대 detection 수 감소로 메모리 절약 및 속도 향상
        "conf": 0.25,  # Confidence threshold
        "iou": 0.7,  # IoU threshold
        "mosaic": 0.0,  # Mosaic augmentation 비활성화 (빈 배치 에러 방지)
        "copy_paste": 0.0,  # Copy-paste augmentation 비활성화
        "mixup": 0.0,  # Mixup augmentation 비활성화
    }
    
    # 체크포인트에서 재개하는 경우 resume 파라미터 추가
    if resume_checkpoint:
        train_kwargs["resume"] = True
    
    model.train(**train_kwargs)

    print("\n✅ FastCut 기반 YOLOv8 파인튜닝 완료!")
    print(f"• 최종 체크포인트: {os.path.join(args.project, args.name, 'weights/best.pt')}")
    print("• Ultralytics 결과 요약은 runs 디렉토리를 확인하세요.")


if __name__ == "__main__":
    main()



