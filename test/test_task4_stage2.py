import sys

sys.path.append('src')

from evaluation.util import get_geo_stationid_map, get_geo_list

# geo_list = get_geo_list()

# geo_set = set()

# for geo in geo_list:
#     if geo in geo_set:
#         print(geo)
#     geo_set.add(geo)


# print(get_geo_stationid_map())

from evaluation.task4.scorer import Task4Scorer
from util.data_process import load_json, save_json

scorer = Task4Scorer(result_folder='result/evaluation/task4_1119_test')

model_result = load_json('/home/kaiyu/Project/WeatherEvaluateSystem/result/evaluation/task4_1119_test/temp_20251119103446.json')

scorer.accuracy_scoring(model_result)

save_json(model_result, '/home/kaiyu/Project/WeatherEvaluateSystem/result/evaluation/task4_1119_test/temp_scoring.json')