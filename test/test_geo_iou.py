import sys

sys.path.append('src')

from evaluation.task4.stage_3_scoring import geo_list_iou, geo_accuracy_scoring
from util.data_process import load_json

# print(geo_list_iou(["惠州北部", "东莞市"], ["惠州北部"]))

model_result = load_json('/home/kaiyu/Project/WeatherEvaluateSystem/result/evaluation/task4/Qwen2.5-VL-7B-Instruct/task4_info_extract_geo_standardize.json')


res = geo_accuracy_scoring(model_result[0])

print(res)