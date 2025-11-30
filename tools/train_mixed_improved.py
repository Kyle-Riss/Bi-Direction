"""
Mixed 모델 성능 개선을 위한 학습 스크립트

Phase 1 개선사항:
- 이미지 크기: 192 -> 320
- 에폭: 15 -> 30
- 배치: 4 -> 2 (이미지 크기 증가로 인한 메모리 고려)
- Confidence: 0.25 -> 0.20 (추론 시)
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
            return "0"
        print("⚠️  CUDA를 사용할 수 없어 CPU로 대체합니다.")
        requested = "cpu"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mixed 모델 성능 개선 학습 (Phase 1)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="carla_mixed.yaml",
        help="YOLO dataset yaml 경로",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/fastcut/mixed_full_e15/weights/best.pt",
        help="기존 모델 가중치 (best.pt 또는 last.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="학습 epoch 수 (기본: 30)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help="배치 크기 (기본: 2, 이미지 크기 증가로 인해 감소)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="입력 이미지 크기 (기본: 320, 이전: 192)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="초기 학습률 (기본: 1e-4)",
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
        default="mixed_improved_phase1",
        help="실험 이름",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="조기 종료 patience (기본: 15)",
    )
    parser.add_argument(
        "--mosaic",
        type=float,
        default=0.0,
        help="Mosaic augmentation 확률 (기본: 0.0, 안전을 위해 비활성화)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Confidence threshold (기본: 0.20, 이전: 0.25)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"데이터셋 YAML을 찾을 수 없습니다: {args.data}"
        )

    # 가중치 파일 확인
    if not os.path.exists(args.weights):
        print(f"⚠️  가중치 파일을 찾을 수 없습니다: {args.weights}")
        print("   기본 yolov8n.pt를 사용합니다.")
        args.weights = "yolov8n.pt"

    print("=" * 70)
    print("🚀 Mixed 모델 성능 개선 학습 (Phase 1)")
    print("=" * 70)
    print(f"• 데이터셋: {args.data}")
    print(f"• 초기 가중치: {args.weights}")
    print(f"• Epochs / Batch / Img: {args.epochs} / {args.batch} / {args.imgsz}")
    print(f"• 학습률: {args.lr}")
    print(f"• Confidence threshold: {args.conf}")
    print(f"• Mosaic augmentation: {args.mosaic}")
    print(f"• Device: {device}")
    print(f"• 결과 저장: {os.path.join(args.project, args.name)}")
    print("=" * 70)
    print("\n📊 개선 사항:")
    print("   ✓ 이미지 크기: 192 → 320 (해상도 2.8배 증가)")
    print("   ✓ 에폭: 15 → 30 (2배 증가)")
    print("   ✓ Confidence: 0.25 → 0.20 (Recall 향상)")
    print("   ⚠️  배치: 4 → 2 (메모리 고려)")
    print("=" * 70 + "\n")

    model = YOLO(args.weights)
    
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
        "pretrained": False,  # 이미 fine-tuned 모델 사용
        "exist_ok": True,
        "workers": 0,
        "cache": False,
        "amp": True,
        "val": True,
        "plots": False,
        "save": True,
        "max_det": 100,
        "conf": args.conf,
        "iou": 0.7,
        "mosaic": args.mosaic,
        "copy_paste": 0.0,
        "mixup": 0.0,
        # 추가 증강 설정
        "degrees": 5.0,      # 작은 회전 추가
        "translate": 0.1,    # 이미 활성화
        "scale": 0.5,        # 이미 활성화
        "fliplr": 0.5,      # 이미 활성화
    }
    
    model.train(**train_kwargs)

    print("\n" + "=" * 70)
    print("✅ Mixed 모델 개선 학습 완료!")
    print("=" * 70)
    print(f"• 최종 체크포인트: {os.path.join(args.project, args.name, 'weights/best.pt')}")
    print(f"• 결과 디렉토리: {os.path.join(args.project, args.name)}")
    print("\n💡 다음 단계:")
    print("   1. results.csv에서 mAP50과 Recall 확인")
    print("   2. 동영상에서 추론 테스트")
    print("   3. 성능 향상 확인 후 Phase 2 고려")
    print("=" * 70)


if __name__ == "__main__":
    main()

