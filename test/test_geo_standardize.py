import sys

sys.path.append('src')

from evaluation.task4.scorer import Task4Scorer
from util.data_process import load_json, save_json

scorer = Task4Scorer(result_folder='result/evaluation/task4/temp')

model_result = load_json('/home/kaiyu/Project/WeatherEvaluateSystem/result/evaluation/task4/temp/task4_extracted_info_20251118230235.json')

res = scorer.geo_standardize(model_result)

save_json(res, 'result/evaluation/task4/temp/temp.json')