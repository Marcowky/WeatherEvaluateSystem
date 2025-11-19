"""Task4 阶段二：为抽取结果计算地理与温度评分"""

import os
import sys
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

# 允许脚本在直接运行时也能加载 src 下的模块
sys.path.append('src')

from evaluation.metric import (geo_list_iou, geo_list_match_and_iou,
                               number_precise_scoring, number_range_scoring,
                               set_iou)
from evaluation.util import geo_list_to_stationid, get_station_id_set
from util.data_process import load_json, save_json

# ------------------------- 默认路径配置 -------------------------
CSV_FOLDER = "/home/kaiyu/Project/WeatherEvaluateSystem/data/task4/2024/tmax"
LABEL_JSON_PATH = "/home/kaiyu/Project/WeatherEvaluateSystem/data/newspaper/task4/qa_data_info_extract_geo_standardize.json"
DEFAULT_INPUT_PATH = "/home/kaiyu/Project/WeatherEvaluateSystem/result/evaluation/task4/Qwen2.5-VL-7B-Instruct/task4_info_extract_geo_standardize.json"
DEFAULT_OUTPUT_PATH = "/home/kaiyu/Project/WeatherEvaluateSystem/result/evaluation/task4/Qwen2.5-VL-7B-Instruct/task4_scoring.json"
SUMMARY_OUTPUT_PATH = DEFAULT_OUTPUT_PATH.replace('.json', '_summary.json')


def get_label_dict(label_path: str) -> Dict[str, Any]:
    """加载标准答案，构建 {qid: 标准答案条目} 的查询字典。"""
    label_data = load_json(label_path)
    label_dict: Dict[str, Any] = {}
    for item in label_data:
        label_dict[item['qid']] = item
    return label_dict


LABEL_DICT = get_label_dict(LABEL_JSON_PATH)


def accuracy_scoring(model_result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    批量执行评分：
    1. 深拷贝输入，避免覆盖上游数据；
    2. 为每条样本添加 accuracy_score 字段；
    3. 返回带评分的新列表。
    """
    scored_result = deepcopy(model_result)
    for single_result in tqdm(scored_result, desc="Scoring accuracy"):
        # 每个样本单独计算得分，保证评分的可追溯性
        single_result['accuracy_score'] = accuracy_scoring_single(single_result)
    return scored_result


def summary(model_result: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总所有样本的平均得分，便于整体评估模型表现。"""
    def _append_value(container: List[float], value: Optional[float]) -> None:
        if value is None:
            return
        container.append(float(value))

    def _average(values: List[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    geo_metrics = {
        'max_temp_geo_iou': [],
        'other_regions_geo_iou': [],
        'specific_regions_geo_avg_iou': [],
    }
    temp_metrics = {
        'max_temp_score': [],
        'other_regions_range_score': [],
        'specific_regions_range_score': [],
    }

    for single_result in model_result:
        accuracy_score = single_result.get('accuracy_score') or {}
        geo_accuracy = accuracy_score.get('geo_accuracy') or {}
        temp_accuracy = accuracy_score.get('temp_accuracy') or {}

        _append_value(geo_metrics['max_temp_geo_iou'], geo_accuracy.get('max_temp_geo_iou'))
        _append_value(geo_metrics['other_regions_geo_iou'], geo_accuracy.get('other_regions_geo_iou'))
        specific_geo_score = geo_accuracy.get('specific_regions_geo_iou') or {}
        _append_value(geo_metrics['specific_regions_geo_avg_iou'], specific_geo_score.get('avg_iou'))

        _append_value(temp_metrics['max_temp_score'], temp_accuracy.get('max_temp_score'))
        other_temp_score = temp_accuracy.get('other_regions_temp_score') or {}
        _append_value(temp_metrics['other_regions_range_score'], other_temp_score.get('range_score'))
        for region_score in temp_accuracy.get('specific_regions_temp_scores') or []:
            _append_value(temp_metrics['specific_regions_range_score'], (region_score or {}).get('range_score'))

    return {
        'total_samples': len(model_result),
        'geo_accuracy': {
            'max_temp_geo_iou': _average(geo_metrics['max_temp_geo_iou']),
            'other_regions_geo_iou': _average(geo_metrics['other_regions_geo_iou']),
            'specific_regions_geo_avg_iou': _average(geo_metrics['specific_regions_geo_avg_iou']),
        },
        'temp_accuracy': {
            'max_temp_score': _average(temp_metrics['max_temp_score']),
            'other_regions_range_score': _average(temp_metrics['other_regions_range_score']),
            'specific_regions_range_score': _average(temp_metrics['specific_regions_range_score']),
        }
    }


def accuracy_scoring_single(single_result: Dict[str, Any]) -> Dict[str, Any]:
    """对单个抽取结果进行准确率评分"""
    geo_accuracy_score = geo_accuracy_scoring(single_result)
    temp_accuracy_score = temp_accuracy_scoring(single_result)
    accuracy_score = {
        'geo_accuracy': geo_accuracy_score,
        'temp_accuracy': temp_accuracy_score,
    }
    return accuracy_score


def geo_accuracy_scoring(single_result: Dict[str, Any]) -> Dict[str, float]:
    """计算地理维度的三个子项 IoU 分数。"""
    single_result_extracted_info = single_result['extracted_info']
    # 根据 qid 获取标准答案的抽取结果
    label_extracted_info = LABEL_DICT[single_result['qid']]['extracted_info']
    # 计算 max_temp 部分的地理准确率
    single_max_temp = single_result_extracted_info.get('max_temp') or {}
    label_max_temp = label_extracted_info.get('max_temp') or {}
    max_temp_geo_iou = geo_list_iou(single_max_temp.get('std_geo', []), label_max_temp.get('std_geo', []))
    # 依靠 station id 的集合计算 other_temp 的 IoU
    label_other_station_id = get_other_station_id(label_extracted_info)
    single_result_other_station_id = get_other_station_id(single_result_extracted_info)
    other_regions_geo_iou = set_iou(label_other_station_id, single_result_other_station_id)
    # 计算 specific_regions 部分的地理准确率
    single_result_geo_list_list = get_std_geo_list(single_result_extracted_info)
    label_geo_list_list = get_std_geo_list(label_extracted_info)
    specific_regions_geo_iou = geo_list_match_and_iou(
        single_result_geo_list_list,
        label_geo_list_list,
    )
    return {
        'max_temp_geo_iou': max_temp_geo_iou,
        'other_regions_geo_iou': other_regions_geo_iou,
        'specific_regions_geo_iou': specific_regions_geo_iou,
    }


def get_std_geo_list(extracted_info: Dict[str, Any]) -> List[List[str]]:
    """获取 specific_regions 中每条数据的标准化地理列表。"""
    geo_list_list = []
    for region in extracted_info.get('specific_regions', []):
        geo_list_list.append(region.get('std_geo', []))
    return geo_list_list


def get_other_station_id(extracted_info: Dict[str, Any]) -> Set[str]:
    """获取除 specific_regions / max_temp 外其余站点对应的 station id 集合。"""
    stationid_without_other: Set[str] = set()
    # 收集 specific_regions 占用的 station id
    for region in extracted_info.get('specific_regions', []):
        stationid_without_other.update(geo_list_to_stationid(region.get('std_geo', [])))
    # 以及 max_temp 所覆盖的站点
    max_temp_geo = (extracted_info.get('max_temp') or {}).get('std_geo', [])
    stationid_without_other.update(geo_list_to_stationid(max_temp_geo))
    # 所有站点减去已出现的站点即为 other station
    other_station_id = get_station_id_set() - stationid_without_other
    return other_station_id


def temp_accuracy_scoring(single_result: Dict[str, Any]) -> Dict[str, Any]:
    """计算温度维度的单值与区间得分。"""
    qid = single_result['qid']
    single_result_extracted_info = single_result['extracted_info']
    # 根据 qid 获取标准答案的抽取结果
    label_extracted_info = LABEL_DICT[qid]['extracted_info']
    # 计算 max_temp 部分的温度准确率
    single_max_temp = (single_result_extracted_info.get('max_temp') or {}).get('tmax', None)
    label_max_temp = (label_extracted_info.get('max_temp') or {}).get('tmax', None)
    max_temp_score = number_precise_scoring(single_max_temp, label_max_temp)
    # 计算 other_temp 部分的温度准确率
    other_regions_temp_score = get_range_score_for_temp_by_station_id_list(
        get_other_station_id(single_result_extracted_info), # 获取 other 部分的站点 id
        single_result_extracted_info.get('other_regions', {}) if single_result_extracted_info.get('other_regions') else {},
        qid
    )
    # 计算 specific_regions 部分的温度准确率
    specific_regions_temp_scores = []
    for region in single_result_extracted_info.get('specific_regions', []):
        region_temp_score = get_range_score_for_temp_by_station_id_list(
            geo_list_to_stationid(region.get('std_geo', [])),
            region,
            qid
        )
        specific_regions_temp_scores.append(region_temp_score)
    
    return {
        'max_temp_score': max_temp_score,
        'other_regions_temp_score': other_regions_temp_score,
        'specific_regions_temp_scores': specific_regions_temp_scores

    }


def get_range_score_for_temp_by_station_id_list(
    station_id_list: Iterable[str],
    region_info: Dict[str, Any],
    qid: str,
) -> Dict[str, Optional[float]]:
    """根据站点集合计算实际温度区间，并与模型区间作 number_range_scoring。"""
    actual_temp_lower, actual_temp_upper = get_actual_temp_lower_upper(station_id_list, qid)
    pred_temp_lower = region_info.get('tmax_min', None)
    pred_temp_upper = region_info.get('tmax_max', None)
    range_score = number_range_scoring(
        (pred_temp_lower, pred_temp_upper),
        (actual_temp_lower, actual_temp_upper)
    )
    return {
        'actual_tmax_min': actual_temp_lower,
        'actual_tmax_max': actual_temp_upper,
        'range_score': range_score
    }


def get_actual_temp_lower_upper(
    station_id_list: Iterable[str],
    qid: str,
) -> Tuple[Optional[float], Optional[float]]:
    """从站点列表获取实际温度的上下界（去除离群值后）。"""
    temp_list = get_actual_temp_list(station_id_list, qid)
    temp_list_no_outliers = remove_outliers(temp_list)
    if len(temp_list_no_outliers) == 0:
        return None, None
    return min(temp_list_no_outliers), max(temp_list_no_outliers)


def get_actual_temp_list(station_id_list: Iterable[str], qid: str) -> List[float]:
    """读取对应 csv，提取 stationid 的 tmax 序列。"""
    # 加载 csv 数据
    temp_csv_path = LABEL_DICT[qid]['input']['csv_data_path']
    csv_path = os.path.join(CSV_FOLDER, temp_csv_path)
    df = pd.read_csv(csv_path)
    # df 的 stationid 列转为 str
    df['stationid'] = df['stationid'].astype(str)
    # 按照 stationid 筛选
    temp_list = []
    for station_id in station_id_list:
        temp_value = df.loc[df['stationid'] == station_id, 'tmax'].values
        if len(temp_value) > 0:
            temp_list.append(float(temp_value[0]))
        else:
            print(f"Station ID {station_id} not found in CSV for QID {qid}.")
    return temp_list


def remove_outliers(temp_list: List[float]) -> List[float]:
    """通过均值 ± 2σ 的方式去除温度离群值。"""
    if len(temp_list) == 0:
        return temp_list
    mean_temp = sum(temp_list) / len(temp_list)
    variance = sum((x - mean_temp) ** 2 for x in temp_list) / len(temp_list)
    std_dev = variance ** 0.5
    # 设定阈值，通常为 2 或 3 个标准差
    threshold = 2 * std_dev
    # 过滤掉离群值
    filtered_temps = [x for x in temp_list if abs(x - mean_temp) <= threshold]
    return filtered_temps


############# 主逻辑

def main(input_path: str = DEFAULT_INPUT_PATH, output_path: str = DEFAULT_OUTPUT_PATH, summary_output_path: str = SUMMARY_OUTPUT_PATH) -> None:
    """命令行入口：读取默认输入，执行评分并写入结果。"""
    # 加载 json 数据
    model_result = load_json(input_path)

    # 逐条打分并附在原始结果中
    model_result_with_accuracy_score = accuracy_scoring(model_result)
    summary_result = summary(model_result_with_accuracy_score)

    # 保存 json 数据
    save_json(model_result_with_accuracy_score, output_path)
    
    # 保存 summary 文件
    print("Summary scores:", summary_result)
    save_json(summary_result, summary_output_path)

if __name__ == '__main__':
    main()
