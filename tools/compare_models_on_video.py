"""
두 모델을 동영상에서 비교하는 스크립트
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import os


def run_inference_on_video(model_path, video_path, output_dir, model_name, conf=0.25):
    """
    동영상에서 추론 실행하고 결과 저장
    
    Args:
        model_path: 모델 파일 경로
        video_path: 동영상 파일 경로
        output_dir: 출력 디렉토리
        model_name: 모델 이름 (출력 파일명에 사용)
        conf: 신뢰도 임계값
    """
    print(f"\n{'='*60}")
    print(f"🔍 Running inference with {model_name}")
    print(f"   Model: {model_path}")
    print(f"   Video: {video_path}")
    print(f"   Confidence: {conf}")
    print(f"{'='*60}\n")
    
    # 모델 로드
    model = YOLO(model_path)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 동영상 이름 추출
    video_name = Path(video_path).stem
    output_video = os.path.join(output_dir, f"{model_name}_{video_name}.mp4")
    
    # 추론 실행
    results = model.predict(
        source=video_path,
        conf=conf,
        save=True,
        project=output_dir,
        name=model_name,
        exist_ok=True,
        verbose=True,
        device='mps'  # Apple Silicon GPU 사용
    )
    
    print(f"\n✅ Inference completed!")
    print(f"   Output saved to: {output_dir}/{model_name}/")
    
    # 통계 정보 출력
    if results and len(results) > 0:
        # 첫 번째 프레임 결과로 통계 확인
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            num_detections = len(result.boxes)
            avg_conf = result.boxes.conf.mean().item() if result.boxes.conf is not None else 0
            print(f"   Detections in first frame: {num_detections}")
            print(f"   Average confidence: {avg_conf:.3f}")
        else:
            print(f"   No detections in first frame")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare two YOLO models on video')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--model1', type=str, required=True,
                       help='Path to first model (baseline)')
    parser.add_argument('--model2', type=str, required=True,
                       help='Path to second model (mixed)')
    parser.add_argument('--output_dir', type=str, default='runs/video_comparison',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--name1', type=str, default='baseline',
                       help='Name for first model (default: baseline)')
    parser.add_argument('--name2', type=str, default='mixed',
                       help='Name for second model (default: mixed)')
    
    args = parser.parse_args()
    
    # 동영상 파일 확인
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        return
    
    # 모델 파일 확인
    if not os.path.exists(args.model1):
        print(f"❌ Error: Model 1 not found: {args.model1}")
        return
    
    if not os.path.exists(args.model2):
        print(f"❌ Error: Model 2 not found: {args.model2}")
        return
    
    print("🎬 Video Model Comparison")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Model 1: {args.model1} ({args.name1})")
    print(f"Model 2: {args.model2} ({args.name2})")
    print(f"Confidence threshold: {args.conf}")
    print("=" * 60)
    
    # 첫 번째 모델로 추론
    results1 = run_inference_on_video(
        args.model1,
        args.video,
        args.output_dir,
        args.name1,
        args.conf
    )
    
    # 두 번째 모델로 추론
    results2 = run_inference_on_video(
        args.model2,
        args.video,
        args.output_dir,
        args.name2,
        args.conf
    )
    
    print("\n" + "=" * 60)
    print("✅ Comparison completed!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)
    print("\n📁 Output structure:")
    print(f"   {args.output_dir}/{args.name1}/ - {args.name1} model results")
    print(f"   {args.output_dir}/{args.name2}/ - {args.name2} model results")
    print("\n💡 Tip: Check the output videos to compare bounding box quality")


if __name__ == "__main__":
    main()

