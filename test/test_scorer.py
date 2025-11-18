import sys

sys.path.append('src')

from evaluation.task4.scorer import Task4Scorer

scorer = Task4Scorer()

print(scorer.score('/home/kaiyu/Project/WeatherEvaluateSystem/data/task4/2024/Qwen2.5-VL-7B-Instruct.json'))