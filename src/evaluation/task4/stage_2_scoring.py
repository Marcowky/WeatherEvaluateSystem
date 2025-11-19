import sys

sys.path.append('src')

from util.data_process import load_json, save_json
from typing import Any, Dict, List
from evaluation.metric import geo_list_iou
from evaluation.util import geo_list_to_stationid, get_station_id_set
from evaluation.metric import set_iou

def get_label_dict():
    """加载标准答案，构建字典以便快速查询"""
    label_data = load_json('/home/kaiyu/Project/WeatherEvaluateSystem/data/newspaper/task4/qa_data_info_extract_geo_standardize.json')
    label_dict = {}
    for item in label_data:
        label_dict[item['qid']] = item
    return label_dict

label_dict = get_label_dict()


def accuracy_scoring(model_result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """对抽取结果进行准确率评分"""
    for i, single_result in enumerate(model_result):
        model_result[i]['accuracy_score'] = accuracy_scoring_single(single_result)
    return model_result

def accuracy_scoring_single(single_result: Dict[str, Any]) -> Dict[str, Any]:
    """对单个抽取结果进行准确率评分"""
    geo_accuracy_score = geo_accuracy_scoring(single_result)
    temp_accuracy_score = temp_accuracy_scoring(single_result)
    accuracy_score = {
        'geo_accuracy': geo_accuracy_score,
        'temp_accuracy': temp_accuracy_score,
    }
    return accuracy_score


def geo_accuracy_scoring(single_result: Dict[str, Any]):
    """对单个抽取结果进行准确率评分"""
    single_result_extracted_info = single_result['extracted_info']
    # 匹配 station id
    label_extracted_info = label_dict[single_result['qid']]['extracted_info']
    # 计算 max_temp 部分的地理准确率
    max_temp_geo_iou = geo_list_iou(single_result_extracted_info['max_temp']['std_geo'], label_extracted_info['max_temp']['std_geo'])
    # 计算 other_temp 部分的地理准确率
    label_other_station_id = get_other_station_id(label_extracted_info)
    single_result_other_station_id = get_other_station_id(single_result_extracted_info)
    other_station_id_iou = set_iou(label_other_station_id, single_result_other_station_id)
    # 计算 specific_regions 部分的地理准确率
    # TODO
    return {
        'max_temp_geo_iou': max_temp_geo_iou,
        'other_station_id_iou': other_station_id_iou,
        # 'specific_regions_geo_iou': specific_regions_geo_iou,
    }


def get_other_station_id(extracted_info: Dict[str, Any]) -> set:
    """获取 other 部分的 station id 集合"""
    stationid_without_other = set()
    for region in extracted_info['specific_regions']:
        stationid_without_other.update(geo_list_to_stationid(region['std_geo']))
    stationid_without_other.update(geo_list_to_stationid(extracted_info['max_temp']['std_geo']))
    other_station_id = get_station_id_set() - stationid_without_other
    return other_station_id


def temp_accuracy_scoring(single_result: Dict[str, Any]):
    """对单个抽取结果进行准确率评分"""
    return 0.0


############# 主逻辑

def main():

    # 加载 json 数据
    model_result = load_json('/home/kaiyu/Project/WeatherEvaluateSystem/result/evaluation/task4_1119_test/temp_20251119103446.json')

    # 进行评分

    model_result_with_accuracy_score = accuracy_scoring(model_result)

    # 保存 json 数据

    save_json(model_result_with_accuracy_score, '/home/kaiyu/Project/WeatherEvaluateSystem/result/evaluation/task4_1119_test/temp_scoring.json')

if __name__ == '__main__':
    main()