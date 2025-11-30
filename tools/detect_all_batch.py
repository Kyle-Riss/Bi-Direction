"""
전체 이미지 배치 처리 스크립트

YOLO로 전체 이미지를 배치 단위로 처리하여 안정성 향상
"""
import os
import glob
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import argparse


def map_coco_to_custom_class(coco_class_id):
    """COCO 클래스를 커스텀 클래스로 매핑"""
    mapping = {
        0: 0,   # person -> pedestrian
        1: 3,   # bicycle -> bicycle_motorcycle
        2: 1,   # car -> car
        3: 3,   # motorcycle -> bicycle_motorcycle
        5: 2,   # bus -> truck_bus
        7: 2,   # truck -> truck_bus
    }
    return mapping.get(coco_class_id, None)


def convert_bbox_to_yolo_format(bbox, img_width, img_height):
    """바운딩 박스를 YOLO 형식으로 변환"""
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # 범위 검증
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0.001, min(1, width))
    height = max(0.001, min(1, height))
    
    return x_center, y_center, width, height


def draw_bbox_on_image(image_path, detections, output_path, class_names):
    """이미지에 바운딩 박스 그리기"""
    img = Image.open(image_path)
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)
    
    colors = {
        0: (255, 0, 0),      # pedestrian
        1: (0, 255, 0),      # car
        2: (0, 0, 255),      # truck_bus
        3: (255, 255, 0),    # bicycle_motorcycle
        4: (255, 0, 255),    # traffic_sign
    }
    
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 20)
    except:
        font = ImageFont.load_default()
    
    for detection in detections:
        if len(detection) == 5:
            class_id, x_center_norm, y_center_norm, width_norm, height_norm = detection
            conf = 1.0
        else:
            class_id, x_center_norm, y_center_norm, width_norm, height_norm, conf = detection
        
        x_center = x_center_norm * img_width
        y_center = y_center_norm * img_height
        width = width_norm * img_width
        height = height_norm * img_height
        
        x1 = max(0, x_center - width / 2)
        y1 = max(0, y_center - height / 2)
        x2 = min(img_width, x_center + width / 2)
        y2 = min(img_height, y_center + height / 2)
        
        color = colors.get(class_id, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
        text = f'{class_name} {conf:.2f}'
        
        try:
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            text_bg = [text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2]
            draw.rectangle(text_bg, fill=color)
            draw.text((x1, y1), text, fill=(0, 0, 0), font=font)
        except:
            pass
    
    img.save(output_path, quality=95)


def process_batch(model, image_files, image_dir, label_dir, vis_dir, 
                  class_names, batch_size=10, conf_threshold=0.25):
    """이미지 배치 처리"""
    total = len(image_files)
    processed = 0
    total_detections = 0
    errors = 0
    
    print(f"📦 총 {total}개 이미지 처리 시작...")
    print(f"   배치 크기: {batch_size}")
    print(f"   신뢰도 임계값: {conf_threshold}\n")
    
    # 배치 단위로 처리
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_files = image_files[batch_start:batch_end]
        
        print(f"📸 배치 {batch_start//batch_size + 1}: {batch_start+1}-{batch_end}/{total}")
        
        for img_path in batch_files:
            try:
                img_basename = os.path.basename(img_path)
                base_name = os.path.splitext(img_basename)[0]
                
                # 이미지 크기 확인
                img = Image.open(img_path)
                img_width, img_height = img.size
                
                # YOLO 추론
                results = model.predict(
                    source=img_path,
                    conf=conf_threshold,
                    iou=0.45,
                    verbose=False,
                    device='cpu'
                )
                
                result = results[0]
                
                # 라벨 파일 생성
                label_file = os.path.join(label_dir, base_name + '.txt')
                detections_list = []
                
                with open(label_file, 'w') as f:
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            coco_class_id = int(box.cls[0])
                            custom_class_id = map_coco_to_custom_class(coco_class_id)
                            
                            if custom_class_id is None:
                                continue
                            
                            bbox = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            
                            x_center, y_center, width, height = convert_bbox_to_yolo_format(
                                bbox, img_width, img_height
                            )
                            
                            f.write(f"{custom_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            detections_list.append((custom_class_id, x_center, y_center, width, height, conf))
                            total_detections += 1
                
                # 시각화 이미지 생성
                if len(detections_list) > 0:
                    vis_path = os.path.join(vis_dir, base_name + '_detected.jpg')
                    draw_bbox_on_image(img_path, detections_list, vis_path, class_names)
                
                processed += 1
                
                if processed % 50 == 0:
                    print(f"   진행: {processed}/{total} ({processed/total*100:.1f}%)")
            
            except Exception as e:
                print(f"   ⚠️  오류 ({os.path.basename(img_path)}): {e}")
                errors += 1
                continue
        
        # 배치 완료 후 잠시 대기 (메모리 정리)
        time.sleep(0.1)
    
    return processed, total_detections, errors


def main():
    parser = argparse.ArgumentParser(description='Batch process all images with YOLO')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model path')
    parser.add_argument('--image_dir', type=str, default='test',
                       help='Image directory')
    parser.add_argument('--label_dir', type=str, default='test/labels',
                       help='Label output directory')
    parser.add_argument('--vis_dir', type=str, default='test/visualized',
                       help='Visualization output directory')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    # 디렉토리 생성
    os.makedirs(args.label_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    
    # 모델 로드
    print(f"📦 모델 로드 중: {args.model}")
    try:
        model = YOLO(args.model)
        print("✅ 모델 로드 완료!\n")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 이미지 파일 찾기
    image_files = sorted(glob.glob(os.path.join(args.image_dir, '*.jpg')) +
                         glob.glob(os.path.join(args.image_dir, '*.png')) +
                         glob.glob(os.path.join(args.image_dir, '*.jpeg')))
    
    if len(image_files) == 0:
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {args.image_dir}")
        return
    
    class_names = ['pedestrian', 'car', 'truck_bus', 'bicycle_motorcycle', 'traffic_sign']
    
    # 배치 처리
    start_time = time.time()
    processed, total_detections, errors = process_batch(
        model, image_files, args.image_dir, args.label_dir, args.vis_dir,
        class_names, args.batch_size, args.conf
    )
    elapsed_time = time.time() - start_time
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"✅ 처리 완료!")
    print(f"{'='*60}")
    print(f"총 이미지: {len(image_files)}개")
    print(f"처리 성공: {processed}개")
    print(f"오류: {errors}개")
    print(f"총 검출 객체: {total_detections}개")
    print(f"평균 검출/이미지: {total_detections/processed if processed > 0 else 0:.2f}개")
    print(f"소요 시간: {elapsed_time:.1f}초 ({elapsed_time/60:.1f}분)")
    print(f"\n📁 결과:")
    print(f"   라벨 파일: {args.label_dir}")
    print(f"   시각화 이미지: {args.vis_dir}")


if __name__ == '__main__':
    main()








