from copy import deepcopy
from typing import Any, Dict, List

from evaluation.scorer_two_stage import TwoStageScorer
from evaluation.util import geo_standardize
from model.client import ModelClient
from model.call_api import call_llm_for_data_cleaning_or_analysis
from prompt.evaluation_prompt import TASK4_PROMPT
from util.data_process import save_json, str_to_json
from util.multi_thread import run_in_threads

MAX_EXTRACTION_ATTEMPTS = 5


class Task4Scorer(TwoStageScorer):
    """Task4 评分器：负责信息抽取+两阶段评分的调度"""

    def __init__(self, result_folder: str = 'result/evaluation') -> None:
        super().__init__(result_folder=result_folder)
        self.name = 'task4'


    def info_extract(self, model_result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        信息抽取接口实现：调用 LLM 提取结构化信息，再做地理位置规范化
        """

        model_result_with_extract_info = self.info_extract_by_llm(model_result)

        save_json(model_result_with_extract_info, f'{self.result_folder}/task4_info_extract_by_llm.json')

        model_result_with_extract_info_geo_standardize = self.geo_standardize(model_result_with_extract_info)

        save_json(model_result_with_extract_info_geo_standardize, f'{self.result_folder}/task4_info_extract_geo_standardize.json')
        
        return model_result_with_extract_info_geo_standardize


    def info_scoring(self, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Task4 评分逻辑留空，由上层根据具体评估策略实现"""
        return {}


    def geo_standardize(self, old_model_result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对地理区域字段做标准化，确保后续评分数据可比"""
        model_result = deepcopy(old_model_result)
        args_list_dict = {
            "single_result": model_result,
        }

        # 多线程向模型发送请求，提升批量效率
        results = run_in_threads(self.geo_standardize_single, args_list_dict, max_workers=5)

        return results
    

    def geo_standardize_single(self, single_result: Dict[str, Any]) -> Dict[str, Any]:
        """单条结果的地理信息标准化"""
        single_info = single_result['extracted_info']
        geo_set = set()
        # 收集所有出现过的地理描述，避免重复调用标准化接口
        for region in single_info['specific_regions']:
            geo_set.update(region['geo'])
        if single_info['max_temp'] is not None:
            geo_set.update(single_info['max_temp']['geo'])

        geo_list = list(geo_set)
        std_geo_list = geo_standardize(geo_list)
        ori_std_geo_map = {ori_geo: std_geo for ori_geo, std_geo in zip(geo_list, std_geo_list)}

        for i, region in enumerate(single_info['specific_regions']):
            single_info['specific_regions'][i]['std_geo'] = [ori_std_geo_map[g] for g in region['geo']]
        if single_info['max_temp'] is not None:
            single_info['max_temp']['std_geo'] = [ori_std_geo_map[g] for g in single_info['max_temp']['geo']]

        single_result['extracted_info'] = single_info

        return single_result

    def info_extract_by_llm(self, old_model_result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用 LLM 对模型输出进行信息抽取，返回带有抽取结果的列表"""
        model_result = deepcopy(old_model_result)
        args_list_dict = {
            "single_result": model_result,
        }

        results = run_in_threads(self.info_extract_by_llm_single, args_list_dict, max_workers=5)
        return results


    def info_extract_by_llm_single(self, single_result: Dict[str, Any]) -> Dict[str, Any]:
        """单条结果的信息抽取，多次重试直至成功或达到上限"""
        client = ModelClient()

        attempts = 0
        invalid_payload = None
        while attempts < MAX_EXTRACTION_ATTEMPTS:
            attempts += 1
            if attempts > 1:
                print(f"Retrying extraction for attempt {attempts}...")
            prompt = TASK4_PROMPT.EXTRACT_INFO.format(original_text=single_result['model_output'])
            try:
                extracted_info = call_llm_for_data_cleaning_or_analysis(
                    client=client,
                    model="deepseek-ai/DeepSeek-V3",
                    prompt=prompt,
                )
            except Exception as e:
                print(f"Error calling LLM: {e}")
                invalid_payload = str(e)
                continue

            json_res = None
            try:
                json_res = str_to_json(extracted_info)
                is_valid = self.validate_extracted_info(json_res)
                if is_valid:
                    single_result['extracted_info'] = json_res
                    return single_result
                print(f"Invalid extracted_info format: {json_res}")
                invalid_payload = json_res
            except Exception as e:
                print(f"Error parsing extracted_info: {e}")
                invalid_payload = extracted_info

        single_result['extracted_info'] = {"error_res": invalid_payload}
        return single_result


    def validate_extracted_info(self, extracted_info: Dict[str, Any]) -> bool:
        """校验抽取结果的格式是否符合预期"""
        required_fields = ['specific_regions', 'other_regions', 'max_temp']
        for field in required_fields:
            if field not in extracted_info:
                return False
            
        str_list = []  # 收集所有应为字符串的字段便于统一校验
        num_list = []  # 收集所有应为数值的字段便于统一校验

        # 判断 specific_regions 格式是否正确
        for region in extracted_info['specific_regions']:
            # 判断字段是否齐全
            if 'geo' not in region or 'tmax_min' not in region or 'tmax_max' not in region:
                return False
            # 判断 geo 是否为列表
            if not isinstance(region['geo'], list):
                return False
            str_list.extend(region['geo'])
            num_list.extend([region['tmax_min'], region['tmax_max']])

        # 判断 other_regions 格式是否正确
        other_regions = extracted_info['other_regions']
        if other_regions is not None:
            if 'tmax_min' not in other_regions or 'tmax_max' not in other_regions:
                return False
            num_list.extend([other_regions['tmax_min'], other_regions['tmax_max']])
            
        # 判断 max_temp 格式是否正确
        max_temp = extracted_info['max_temp']
        if max_temp is not None:
            if 'geo' not in max_temp or 'tmax' not in max_temp:
                return False
            if not isinstance(max_temp['geo'], list):
                return False
            str_list.extend(max_temp['geo'])
            num_list.append(max_temp['tmax'])

        # 判断 geo 列表内元素均为字符串
        for item in str_list:
            if not isinstance(item, str):
                return False
        # 判断数值列表内元素均为数字
        for item in num_list:
            if not isinstance(item, (int, float)) and item is not None:
                return False
            
        return True
