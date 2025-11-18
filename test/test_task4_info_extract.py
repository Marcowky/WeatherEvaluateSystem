import sys

sys.path.append('src')

from evaluation.task4.scorer import Task4Scorer
from util.data_process import load_json

scorer = Task4Scorer(result_folder='result/evaluation/task4/temp')

model_result = load_json('/home/kaiyu/Project/WeatherEvaluateSystem/data/task4/2024/Qwen2.5-VL-7B-Instruct.json')

scorer.info_extract(model_result)