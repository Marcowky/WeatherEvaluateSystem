import sys

sys.path.append('src')

from evaluation.task4.stage_2_scoring import temp_accuracy_scoring, get_range_score_for_temp_by_station_id_list
from util.data_process import load_json
from evaluation.util import geo_list_to_stationid

# print(geo_list_iou(["惠州北部", "东莞市"], ["惠州北部"]))

model_result = load_json('/home/kaiyu/Project/WeatherEvaluateSystem/result/evaluation/task4/Qwen2.5-VL-7B-Instruct/task4_info_extract_geo_standardize.json')


single_result = model_result[26]
qid = single_result['qid']

res = temp_accuracy_scoring(single_result)

print(res)

# 计算 specific_regions 部分的温度准确率
specific_regions_temp_scores = []
for region in single_result['extracted_info'].get('specific_regions', []):
    print("-" * 20)
    region_temp_score = get_range_score_for_temp_by_station_id_list(
        geo_list_to_stationid(region.get('std_geo', [])),
        region,
        qid
    )
    specific_regions_temp_scores.append(region_temp_score)

print(specific_regions_temp_scores)

print(single_result)