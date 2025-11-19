import sys

sys.path.append('src')

from evaluation.task4.stage_2_scoring import geo_list_iou, geo_accuracy_scoring, get_label_dict
from util.data_process import load_json
from evaluation.util import geo_list_to_stationid

# print(geo_list_iou(["惠州北部", "东莞市"], ["惠州北部"]))

model_result = load_json('/home/kaiyu/Project/WeatherEvaluateSystem/result/evaluation/task4/Qwen2.5-VL-7B-Instruct/task4_info_extract_geo_standardize.json')


single_result = model_result[26]

res = geo_accuracy_scoring(single_result)

print(res)

label_dict = get_label_dict()


single_result_extracted_info = single_result['extracted_info']

label_extracted_info = label_dict[single_result['qid']]['extracted_info']
print(single_result_extracted_info)
print(label_extracted_info)

print('=='*20)

for region in single_result_extracted_info['specific_regions']:
    print(region['std_geo'])
    print(geo_list_to_stationid(region['std_geo']))

print('=='*20)

for region in label_extracted_info['specific_regions']:
    print(region['std_geo'])
    print(geo_list_to_stationid(region['std_geo']))