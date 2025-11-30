"""
Mixed 모델에 표지판/신호등 클래스 추가 학습

기존 모델 (vehicle, pedestrian)에 traffic_sign, traffic_light 추가
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
        description="Mixed 모델에 표지판/신호등 클래스 추가 학습"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="carla_mixed.yaml",
        help="YOLO dataset yaml 경로 (4개 클래스 포함)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/fastcut/mixed_full_e15/weights/best.pt",
        help="기존 모델 가중치 (2개 클래스 모델)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="추가 학습 epoch 수 (기본: 20)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="배치 크기 (기본: 4)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=192,
        help="입력 이미지 크기 (기본: 192, 기존과 동일)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="초기 학습률 (기본: 5e-5, 추가 학습이므로 낮게)",
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
        default="mixed_with_traffic",
        help="실험 이름",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="조기 종료 patience (기본: 10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"데이터셋 YAML을 찾을 수 없습니다: {args.data}"
        )

    if not os.path.exists(args.weights):
        print(f"⚠️  가중치 파일을 찾을 수 없습니다: {args.weights}")
        print("   기본 yolov8n.pt를 사용합니다.")
        args.weights = "yolov8n.pt"

    print("=" * 70)
    print("🚦 Mixed 모델에 표지판/신호등 클래스 추가 학습")
    print("=" * 70)
    print(f"• 데이터셋: {args.data}")
    print(f"• 기존 모델: {args.weights} (2개 클래스: vehicle, pedestrian)")
    print(f"• 추가 클래스: traffic_sign, traffic_light")
    print(f"• 최종 클래스: 4개 (vehicle, pedestrian, traffic_sign, traffic_light)")
    print(f"• Epochs / Batch / Img: {args.epochs} / {args.batch} / {args.imgsz}")
    print(f"• 학습률: {args.lr} (추가 학습용 낮은 학습률)")
    print(f"• Device: {device}")
    print(f"• 결과 저장: {os.path.join(args.project, args.name)}")
    print("=" * 70)
    print("\n⚠️  주의사항:")
    print("   - 클래스 수가 2개 → 4개로 변경됩니다")
    print("   - YOLOv8이 자동으로 마지막 레이어를 재초기화합니다")
    print("   - 기존 vehicle/pedestrian 가중치는 유지됩니다")
    print("   - 새로운 traffic_sign/traffic_light는 처음부터 학습됩니다")
    print("=" * 70 + "\n")

    # 모델 로드
    model = YOLO(args.weights)
    
    # 데이터셋 정보 확인
    import yaml
    with open(args.data, 'r') as f:
        data_info = yaml.safe_load(f)
    num_classes = data_info.get('nc', 2)
    class_names = data_info.get('names', {})
    
    print(f"📋 데이터셋 클래스 정보:")
    print(f"   총 클래스 수: {num_classes}")
    for cls_id, cls_name in class_names.items():
        print(f"   {cls_id}: {cls_name}")
    print()
    
    if num_classes != 4:
        print("⚠️  경고: 데이터셋에 4개 클래스가 정의되어 있지 않습니다!")
        print("   carla_mixed.yaml을 확인하세요.")
        response = input("계속 진행하시겠습니까? (y/n): ")
        if response.lower() != 'y':
            return
    
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
        "conf": 0.25,
        "iou": 0.7,
        "mosaic": 0.0,
        "copy_paste": 0.0,
        "mixup": 0.0,
    }
    
    model.train(**train_kwargs)

    print("\n" + "=" * 70)
    print("✅ 표지판/신호등 클래스 추가 학습 완료!")
    print("=" * 70)
    print(f"• 최종 체크포인트: {os.path.join(args.project, args.name, 'weights/best.pt')}")
    print(f"• 결과 디렉토리: {os.path.join(args.project, args.name)}")
    print("\n💡 다음 단계:")
    print("   1. results.csv에서 4개 클래스별 성능 확인")
    print("   2. 동영상에서 추론 테스트 (표지판/신호등 탐지 확인)")
    print("=" * 70)


if __name__ == "__main__":
    main()

