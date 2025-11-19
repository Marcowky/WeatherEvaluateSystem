from typing import Any, Dict, List
from scipy.optimize import linear_sum_assignment
from evaluation.util import geo_list_to_stationid

def set_iou(pred_set: set, label_set: set) -> float:
    """计算两个集合的set iou"""
    if not pred_set and not label_set:
        return 1.0
    intersection = len(pred_set.intersection(label_set))
    union = len(pred_set.union(label_set))
    if union == 0:
        return 0.0
    return intersection / union


def geo_list_iou(pred_geo_list: List[str], label_geo_list: List[str]) -> float:
    """计算两个地理位置列表的 IOU"""
    pred_station_ids = geo_list_to_stationid(pred_geo_list)
    label_station_ids = geo_list_to_stationid(label_geo_list)
    iou = set_iou(set(pred_station_ids), set(label_station_ids))
    return iou


def number_precise_scoring(predicted: float, actual: float) -> float:
    """根据预测值和实际值的差距，返回精度评分"""
    diff = abs(actual - predicted)

    if diff < 0.09:
        return 1.0
    elif diff < 0.5:
        return 0.5
    elif diff < 1.0:
        return 0.1
    else:
        return 0.0
    

def geo_list_match_and_iou(pred_geo_list_list: List[List[str]], label_geo_list_list: List[List[str]]) -> Dict[str, Any]:
    """使用匈牙利算法为 geo_list_list 配对并统计 IoU"""
    pred_geo_list_list = pred_geo_list_list or []
    label_geo_list_list = label_geo_list_list or []
    pred_count = len(pred_geo_list_list)
    label_count = len(label_geo_list_list)
    if pred_count == 0 and label_count == 0:
        return {
            'matched_pairs': [],
            'unmatched_label_indices': [],
            'avg_iou': 1.0,
        }

    # 构建预测/标注之间的 IoU 矩阵，后续供匈牙利算法挑选最佳配对
    iou_matrix: List[List[float]] = []
    for pred_geo_list in pred_geo_list_list:
        row = []
        for label_geo_list in label_geo_list_list:
            row.append(geo_list_iou(pred_geo_list, label_geo_list))
        iou_matrix.append(row)

    # 线性和分配默认求最小值，取 cost=1-IoU 即可将最大 IoU 转化为最小 cost
    cost_matrix = [[1.0 - iou for iou in row] for row in iou_matrix]
    if cost_matrix and label_count > 0:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignment_map = {r: c for r, c in zip(row_ind, col_ind)}
    else:
        assignment_map = {}

    matched_pairs = []
    matched_label_indices = set()
    # 遍历所有预测项，记录其最佳配对及 IoU
    for pred_idx in range(pred_count):
        label_idx = assignment_map.get(pred_idx)
        if label_idx is None or label_idx < 0 or label_idx >= label_count:
            # matched_pairs.append({
            #     'prediction_index': pred_idx,
            #     'label_index': None,
            #     'iou': 0.0,
            # })
            continue
        matched_label_indices.add(label_idx)
        matched_pairs.append({
            'prediction_index': pred_idx,
            'label_index': int(label_idx),
            'iou': iou_matrix[pred_idx][label_idx],
        })

    total_pairs = max(pred_count, label_count)
    total_iou = sum(pair['iou'] for pair in matched_pairs)
    # 将 IoU 归一到预测和标注数量较大的那一侧，避免数量差异导致得分虚高
    avg_iou = total_iou / total_pairs if total_pairs else 1.0

    return {
        'matched_pairs': matched_pairs,
        'unmatched_prediction_indices': [idx for idx in range(pred_count) if idx not in assignment_map],
        'unmatched_label_indices': [idx for idx in range(label_count) if idx not in matched_label_indices],
        'avg_iou': avg_iou,
    }