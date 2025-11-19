"""Task4 阶段一-2：对抽取结果的地理描述进行标准化"""

import sys

# 使脚本在 CLI 下运行时也能加载项目内模块
sys.path.append('src')

from copy import deepcopy
from typing import Any, Dict, List

from evaluation.util import geo_standardize
from util.data_process import load_json, save_json
from util.multi_thread import run_in_threads

# ------------------------- 默认路径配置 -------------------------
DEFAULT_INPUT_PATH = '/home/kaiyu/Project/WeatherEvaluateSystem/result/evaluation/task4_1119_test/task4_info_extract_by_llm.json'
DEFAULT_OUTPUT_PATH = '/home/kaiyu/Project/WeatherEvaluateSystem/result/evaluation/task4_1119_test/task4_info_extract_geo_standardize.json'


def geo_standardize_batch(old_model_result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    批量为 extracted_info 中的地理字段添加 std_geo：
    1. 深拷贝输入，避免污染上游数据；
    2. 使用线程池加速对每条样本的处理；
    3. 将 geo_standardize_single 的输出重新组合成列表。
    """
    model_result = deepcopy(old_model_result)
    args_list_dict = {"single_result": model_result}
    results = run_in_threads(geo_standardize_single, args_list_dict, max_workers=5)
    return results


def geo_standardize_single(single_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    单条样本的地理标准化：
    - 收集 specific_regions 与 max_temp 中所有地理描述，去重后调用 geo_standardize；
    - 根据原始 -> 标准化映射补充 std_geo 字段；
    - 返回带有 std_geo 的 single_result。
    """
    single_info = single_result['extracted_info']

    geo_set = set()
    # 收集所有出现过的地理描述，减少重复调用标准化接口
    for region in single_info['specific_regions']:
        geo_set.update(region.get('geo', []))
    if single_info['max_temp'] is not None:
        geo_set.update(single_info['max_temp'].get('geo', []))

    geo_list = list(geo_set)
    std_geo_list = geo_standardize(geo_list)
    ori_std_geo_map = {ori_geo: std_geo for ori_geo, std_geo in zip(geo_list, std_geo_list)}

    # specific_regions 中每个 geo 列表补充 std_geo
    for i, region in enumerate(single_info['specific_regions']):
        single_info['specific_regions'][i]['std_geo'] = [ori_std_geo_map[g] for g in region.get('geo', [])]
    # max_temp 若存在也补齐 std_geo
    if single_info['max_temp'] is not None:
        single_info['max_temp']['std_geo'] = [ori_std_geo_map[g] for g in single_info['max_temp'].get('geo', [])]

    single_result['extracted_info'] = single_info
    return single_result


def main():
    """命令行入口：读取默认输入，执行地理标准化并写入结果"""
    model_result = load_json(DEFAULT_INPUT_PATH)
    geo_standardized_result = geo_standardize_batch(model_result)
    save_json(geo_standardized_result, DEFAULT_OUTPUT_PATH)


if __name__ == '__main__':
    main()
