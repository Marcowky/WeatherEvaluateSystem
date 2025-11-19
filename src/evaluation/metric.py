from typing import List
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