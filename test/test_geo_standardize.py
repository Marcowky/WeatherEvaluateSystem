import sys

sys.path.append('src')

from evaluation.task4.scorer import Task4Scorer
from util.data_process import load_json, save_json

scorer = Task4Scorer(result_folder='result/evaluation/task4/temp')

model_result = load_json('/home/kaiyu/Project/WeatherEvaluateSystem/result/evaluation/task4_1119_test/task4_info_extract_by_llm_20251119101123.json')

res = scorer.geo_standardize(model_result)

save_json(res, 'result/evaluation/task4/temp/temp.json')