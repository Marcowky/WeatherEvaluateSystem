"""Task4 阶段一-1：调用大模型抽取结构化气象信息"""

import sys

# 允许脚本以 CLI 方式运行时能够导入 src 目录下的模块
sys.path.append('src')

from copy import deepcopy
from typing import Any, Dict, List

from model.call_api import call_llm_for_data_cleaning_or_analysis
from model.client import ModelClient
from prompt.evaluation_prompt import TASK4_PROMPT
from util.data_process import load_json, save_json, str_to_json
from util.multi_thread import run_in_threads

# ------------------------- 全局配置 -------------------------
MAX_EXTRACTION_ATTEMPTS = 5  # 单条样本最大重试次数，避免无限循环
# 默认输入/输出文件，便于直接运行脚本做快速调试
DEFAULT_INPUT_PATH = '/home/kaiyu/Project/WeatherEvaluateSystem/result/evaluation/task4_1119_test/temp_20251119103446.json'
DEFAULT_OUTPUT_PATH = '/home/kaiyu/Project/WeatherEvaluateSystem/result/evaluation/task4_1119_test/task4_info_extract_by_llm.json'


def info_extract_by_llm(old_model_result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    批量执行信息抽取：
    1. 深拷贝输入，避免覆盖上游输出；
    2. 使用线程池并发处理，加速大批量请求；
    3. 返回带有 extracted_info 字段的新列表。
    """
    model_result = deepcopy(old_model_result)
    # run_in_threads 会自动为每个元素调用 info_extract_by_llm_single
    args_list_dict = {"single_result": model_result}
    results = run_in_threads(info_extract_by_llm_single, args_list_dict, max_workers=5)
    return results


def info_extract_by_llm_single(single_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    单条样本的信息抽取流程：
    - 构造 Prompt 调用 DeepSeek-V3；
    - 将字符串输出解析为 JSON；
    - 检查结构合法性，不符合就重试；
    - 重试 MAX_EXTRACTION_ATTEMPTS 次后仍失败，则记录 error 信息返回。
    """
    client = ModelClient()  # 每个线程持有独立 client，避免共享状态

    attempts = 0
    invalid_payload = None  # 保存最后一次失败的内容，方便排查
    while attempts < MAX_EXTRACTION_ATTEMPTS:
        attempts += 1
        if attempts > 1:
            print(f"Retrying extraction for attempt {attempts}...")

        # 构造抽取提示词，将原文插入 Prompt
        prompt = TASK4_PROMPT.EXTRACT_INFO.format(original_text=single_result['model_output'])
        try:
            extracted_info = call_llm_for_data_cleaning_or_analysis(
                client=client,
                model="deepseek-ai/DeepSeek-V3",
                prompt=prompt,
            )
        except Exception as e:
            # 接口调用失败，记录异常并继续下一次重试
            print(f"Error calling LLM: {e}")
            invalid_payload = str(e)
            continue

        try:
            # 将模型输出转换为 JSON，并校验字段格式
            json_res = str_to_json(extracted_info)
            if validate_extracted_info(json_res):
                single_result['extracted_info'] = json_res
                return single_result
            print(f"Invalid extracted_info format: {json_res}")
            invalid_payload = json_res
        except Exception as e:
            # JSON 解析报错，直接记录原始字符串
            print(f"Error parsing extracted_info: {e}")
            invalid_payload = extracted_info

    # 所有重试均失败，写入错误信息以便后续人工处理
    single_result['extracted_info'] = {"error_res": invalid_payload}
    return single_result


def validate_extracted_info(extracted_info: Dict[str, Any]) -> bool:
    """
    校验大模型输出是否满足评分需求：
    - specific_regions / other_regions / max_temp 三个一级字段必须存在；
    - specific_regions 中每个元素需要 geo 列表 + tmax_min/max；
    - other_regions、max_temp 的数值字段必须为数值；
    - geo 列表元素均为字符串。
    """
    required_fields = ['specific_regions', 'other_regions', 'max_temp']
    for field in required_fields:
        if field not in extracted_info:
            return False

    str_list: List[Any] = []
    num_list: List[Any] = []

    # specific_regions 校验
    for region in extracted_info['specific_regions']:
        if 'geo' not in region or 'tmax_min' not in region or 'tmax_max' not in region:
            return False
        if not isinstance(region['geo'], list):
            return False
        str_list.extend(region['geo'])
        num_list.extend([region['tmax_min'], region['tmax_max']])

    # other_regions 可为 None，但若存在则需数值字段
    other_regions = extracted_info['other_regions']
    if other_regions is not None:
        if 'tmax_min' not in other_regions or 'tmax_max' not in other_regions:
            return False
        num_list.extend([other_regions['tmax_min'], other_regions['tmax_max']])

    # max_temp 同理，可为空；存在时必须包含 geo 列表与 tmax
    max_temp = extracted_info['max_temp']
    if max_temp is not None:
        if 'geo' not in max_temp or 'tmax' not in max_temp:
            return False
        if not isinstance(max_temp['geo'], list):
            return False
        str_list.extend(max_temp['geo'])
        num_list.append(max_temp['tmax'])

    # 核验 geo 列表均为字符串
    for item in str_list:
        if not isinstance(item, str):
            return False
    # 数值字段必须为数字或 None（允许缺失）
    for item in num_list:
        if not isinstance(item, (int, float)) and item is not None:
            return False

    return True


def main():
    """命令行入口：读取默认输入，执行抽取并写入结果"""
    model_result = load_json(DEFAULT_INPUT_PATH)
    model_result_with_extract_info = info_extract_by_llm(model_result)
    save_json(model_result_with_extract_info, DEFAULT_OUTPUT_PATH)


if __name__ == '__main__':
    main()
