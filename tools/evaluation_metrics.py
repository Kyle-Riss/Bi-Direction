"""
평가 메트릭 구현

YOLO 모델의 성능 평가를 위한 메트릭:
- mAP (mean Average Precision)
- F1 Score
- Precision, Recall
- IoU (Intersection over Union)
"""
import torch
import numpy as np
from collections import defaultdict


def compute_iou(box1, box2):
    """
    두 바운딩 박스의 IoU (Intersection over Union) 계산
    
    Args:
        box1: (x1, y1, x2, y2) 또는 (x_center, y_center, width, height)
        box2: (x1, y1, x2, y2) 또는 (x_center, y_center, width, height)
    
    Returns:
        iou: IoU 값 (0-1)
    """
    # YOLO 형식 (x_center, y_center, width, height)을 xyxy로 변환
    if len(box1) == 4 and len(box2) == 4:
        # YOLO 형식인지 확인 (값이 0-1 범위인지)
        if all(0 <= val <= 1 for val in box1) and all(0 <= val <= 1 for val in box2):
            # YOLO 형식으로 가정
            x1_c, y1_c, w1, h1 = box1
            x2_c, y2_c, w2, h2 = box2
            
            # xyxy로 변환
            x1_1 = x1_c - w1 / 2
            y1_1 = y1_c - h1 / 2
            x1_2 = x1_c + w1 / 2
            y1_2 = y1_c + h1 / 2
            
            x2_1 = x2_c - w2 / 2
            y2_1 = y2_c - h2 / 2
            x2_2 = x2_c + w2 / 2
            y2_2 = y2_c + h2 / 2
        else:
            # 이미 xyxy 형식
            x1_1, y1_1, x1_2, y1_2 = box1
            x2_1, y2_1, x2_2, y2_2 = box2
    else:
        return 0.0
    
    # Intersection 계산
    x1_inter = max(x1_1, x2_1)
    y1_inter = max(y1_1, y2_1)
    x2_inter = min(x1_2, x2_2)
    y2_inter = min(y1_2, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Union 계산
    box1_area = (x1_2 - x1_1) * (y1_2 - y1_1)
    box2_area = (x2_2 - x2_1) * (y2_2 - y2_1)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou


def compute_precision_recall(predictions, ground_truths, iou_threshold=0.5, num_classes=5):
    """
    Precision과 Recall 계산
    
    Args:
        predictions: [(class_id, x_center, y_center, width, height, conf), ...]
        ground_truths: [(class_id, x_center, y_center, width, height), ...]
        iou_threshold: IoU 임계값 (기본: 0.5)
        num_classes: 클래스 수
    
    Returns:
        precision: 클래스별 precision
        recall: 클래스별 recall
        f1: 클래스별 F1 score
    """
    class_precision = {}
    class_recall = {}
    class_f1 = {}
    
    for class_id in range(num_classes):
        pred_boxes = [(bbox, conf) for cls, *bbox, conf in predictions if cls == class_id]
        gt_boxes = [bbox for cls, *bbox in ground_truths if cls == class_id]
        
        # 신뢰도 기준으로 정렬
        pred_boxes.sort(key=lambda x: x[1], reverse=True)
        
        # True Positive, False Positive 계산
        tp = 0
        fp = 0
        
        matched_gt = set()
        
        for pred_box, conf in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            
            # 가장 높은 IoU를 가진 ground truth 찾기
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                
                iou = compute_iou(pred_box[:4], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            # IoU가 임계값 이상이면 TP, 아니면 FP
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        # Precision, Recall 계산
        total_pred = len(pred_boxes)
        total_gt = len(gt_boxes)
        
        precision = tp / total_pred if total_pred > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_precision[class_id] = precision
        class_recall[class_id] = recall
        class_f1[class_id] = f1
    
    return class_precision, class_recall, class_f1


def compute_ap(precision, recall):
    """
    Average Precision (AP) 계산
    
    Precision-Recall 곡선 아래의 면적
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    
    return ap


def compute_map(predictions_dict, ground_truths_dict, iou_threshold=0.5, num_classes=5):
    """
    mAP (mean Average Precision) 계산
    
    Args:
        predictions_dict: {image_name: [(class_id, x, y, w, h, conf), ...]}
        ground_truths_dict: {image_name: [(class_id, x, y, w, h), ...]}
        iou_threshold: IoU 임계값
        num_classes: 클래스 수
    
    Returns:
        map: mAP 값 (0-1)
        class_ap: 클래스별 AP
    """
    # 클래스별로 모든 예측과 ground truth 수집
    class_predictions = defaultdict(list)
    class_ground_truths = defaultdict(list)
    
    for img_name in predictions_dict.keys():
        preds = predictions_dict.get(img_name, [])
        gts = ground_truths_dict.get(img_name, [])
        
        for class_id in range(num_classes):
            # 예측
            for pred in preds:
                if pred[0] == class_id:
                    class_predictions[class_id].append(pred + (img_name,))
            
            # Ground truth
            for gt in gts:
                if gt[0] == class_id:
                    class_ground_truths[class_id].append(gt + (img_name,))
    
    # 클래스별 AP 계산
    class_ap = {}
    aps = []
    
    for class_id in range(num_classes):
        preds = class_predictions[class_id]
        gts = class_ground_truths[class_id]
        
        if len(preds) == 0 and len(gts) == 0:
            ap = 1.0  # 둘 다 없으면 완벽한 성능
        elif len(preds) == 0:
            ap = 0.0  # 예측 없고 GT만 있으면 0
        elif len(gts) == 0:
            ap = 0.0  # GT 없고 예측만 있으면 0
        else:
            # 신뢰도 기준 정렬
            preds.sort(key=lambda x: x[5], reverse=True)  # conf가 5번째 인덱스
            
            # Precision-Recall 계산
            tp = 0
            fp = 0
            matched_gt = set()
            
            precisions = []
            recalls = []
            
            for pred in preds:
                img_name = pred[6]
                pred_box = pred[1:5]  # x, y, w, h
                pred_conf = pred[5]
                
                # 같은 이미지의 GT만 고려
                img_gts = [gt for gt in gts if gt[5] == img_name]  # img_name이 5번째 인덱스
                
                best_iou = 0.0
                best_gt_idx = -1
                
                for i, gt in enumerate(img_gts):
                    if (img_name, i) in matched_gt:
                        continue
                    
                    gt_box = gt[1:5]
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou >= iou_threshold:
                    tp += 1
                    matched_gt.add((img_name, best_gt_idx))
                else:
                    fp += 1
                
                # Precision, Recall 계산
                total_pred = tp + fp
                precision = tp / total_pred if total_pred > 0 else 0.0
                recall = tp / len(gts) if len(gts) > 0 else 0.0
                
                precisions.append(precision)
                recalls.append(recall)
            
            # AP 계산 (11-point interpolation)
            precisions = np.array(precisions)
            recalls = np.array(recalls)
            ap = compute_ap(precisions, recalls)
        
        class_ap[class_id] = ap
        aps.append(ap)
    
    # mAP 계산 (모든 클래스의 평균)
    map_value = np.mean(aps) if len(aps) > 0 else 0.0
    
    return map_value, class_ap


def evaluate_yolo_predictions(predictions_dir, ground_truths_dir, num_classes=5, iou_threshold=0.5):
    """
    YOLO 형식 라벨 파일들을 평가
    
    Args:
        predictions_dir: 예측 라벨 파일 디렉토리
        ground_truths_dir: 실제 라벨 파일 디렉토리
        num_classes: 클래스 수
        iou_threshold: IoU 임계값
    
    Returns:
        metrics: 평가 메트릭 딕셔너리
    """
    import os
    import glob
    
    # 라벨 파일 읽기
    pred_dict = {}
    gt_dict = {}
    
    pred_files = glob.glob(os.path.join(predictions_dir, '*.txt'))
    
    for pred_file in pred_files:
        img_name = os.path.basename(pred_file).replace('.txt', '')
        gt_file = os.path.join(ground_truths_dir, img_name + '.txt')
        
        # 예측 읽기
        preds = []
        if os.path.exists(pred_file):
            with open(pred_file, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) == 5:
                        class_id, x, y, w, h = parts
                        # 신뢰도는 없으므로 1.0으로 가정
                        preds.append((int(class_id), x, y, w, h, 1.0))
        pred_dict[img_name] = preds
        
        # Ground truth 읽기
        gts = []
        if os.path.exists(gt_file):
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) == 5:
                        class_id, x, y, w, h = parts
                        gts.append((int(class_id), x, y, w, h))
        gt_dict[img_name] = gts
    
    # mAP 계산
    map_value, class_ap = compute_map(pred_dict, gt_dict, iou_threshold, num_classes)
    
    # Precision, Recall, F1 계산
    all_preds = []
    all_gts = []
    for img_name in pred_dict.keys():
        all_preds.extend(pred_dict[img_name])
        all_gts.extend(gt_dict.get(img_name, []))
    
    class_precision, class_recall, class_f1 = compute_precision_recall(
        all_preds, all_gts, iou_threshold, num_classes
    )
    
    # 평균 계산
    avg_precision = np.mean(list(class_precision.values()))
    avg_recall = np.mean(list(class_recall.values()))
    avg_f1 = np.mean(list(class_f1.values()))
    
    metrics = {
        'mAP': map_value,
        'class_AP': class_ap,
        'precision': avg_precision,
        'recall': avg_recall,
        'F1': avg_f1,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1
    }
    
    return metrics


def print_evaluation_results(metrics, class_names=None):
    """평가 결과 출력"""
    if class_names is None:
        class_names = ['pedestrian', 'car', 'truck_bus', 'bicycle_motorcycle', 'traffic_sign']
    
    print("=" * 60)
    print("📊 평가 결과")
    print("=" * 60)
    print(f"mAP@0.5: {metrics['mAP']:.4f}")
    print(f"평균 Precision: {metrics['precision']:.4f}")
    print(f"평균 Recall: {metrics['recall']:.4f}")
    print(f"평균 F1 Score: {metrics['F1']:.4f}")
    
    print("\n클래스별 AP:")
    for class_id, ap in metrics['class_AP'].items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        print(f"  {class_name} (class {class_id}): {ap:.4f}")
    
    print("\n클래스별 Precision/Recall/F1:")
    for class_id in sorted(metrics['class_precision'].keys()):
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        prec = metrics['class_precision'][class_id]
        rec = metrics['class_recall'][class_id]
        f1 = metrics['class_f1'][class_id]
        print(f"  {class_name}: P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate YOLO predictions')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Predictions label directory')
    parser.add_argument('--ground_truths', type=str, required=True,
                       help='Ground truth label directory')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of classes')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU threshold')
    
    args = parser.parse_args()
    
    metrics = evaluate_yolo_predictions(
        args.predictions,
        args.ground_truths,
        args.num_classes,
        args.iou_threshold
    )
    
    print_evaluation_results(metrics)








