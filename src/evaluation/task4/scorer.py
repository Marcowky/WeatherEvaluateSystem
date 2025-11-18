from evaluation.scorer_two_stage import TwoStageScorer
from model.client import ModelClient
from model.call_api import call_llm_for_data_cleaning_or_analysis
from prompt.evaluation_prompt import task4_prompt
from util.data_process import save_json, str_to_json
from copy import deepcopy

class Task4Scorer(TwoStageScorer):
    """Task4 评分器：负责信息抽取+两阶段评分的调度"""
    def __init__(self, result_folder: str = 'result/evaluation') -> None:
        super().__init__(result_folder=result_folder)
        self.name = 'task4'


    def info_extract(self, model_result):
        """信息抽取接口实现，首先调用大模型提取结构化信息，然后对地理位置进行规范化"""
        # 处理 model_result，提取所需信息
        self.client = ModelClient()  # 初始化客户端，避免在多线程中重复创建

        model_result = self.info_extract_by_llm(model_result)
        
        return model_result


    def info_scoring(self, extracted_info):
        # Implement the scoring logic specific to Task 4
        scoring_result = {}
        # ... scoring code ...
        return scoring_result


    def info_extract_by_llm(self, old_model_result):
        """使用 LLM 对模型输出进行信息抽取，返回带有抽取结果的列表"""
        model_result = deepcopy(old_model_result)
        pending_indices = list(range(len(model_result)))
        invalid_payloads = {}
        max_attempts = 5
        attempts = 0

        while pending_indices and attempts < max_attempts:
            print(f"Info extraction attempt {attempts + 1}, pending items: {len(pending_indices)}")
            attempts += 1
            extracted_infos = self.natural_language_to_json_format(
                [model_result[idx]['model_output'] for idx in pending_indices]
            )

            next_pending = []
            for idx, extracted_info in zip(pending_indices, extracted_infos):
                json_res = None
                is_valid = False

                try:
                    json_res = str_to_json(extracted_info)
                    is_valid = self.validate_extracted_info(json_res)
                    if not is_valid:
                        print(f"Invalid extracted_info format for index {idx}: {json_res}")
                    else:
                        model_result[idx]['extracted_info'] = json_res
                        if idx in invalid_payloads:
                            invalid_payloads.pop(idx)
                except Exception as e:
                    print(f"Error parsing extracted_info for index {idx}: {e}")

                if not is_valid:
                    next_pending.append(idx)
                    invalid_payloads[idx] = json_res if json_res is not None else extracted_info

            pending_indices = next_pending

        # 经过多次尝试仍无法通过校验的结果使用 error_res 标记
        for idx in pending_indices:
            model_result[idx]['extracted_info'] = {"error_res": invalid_payloads.get(idx)}

        # 将带有抽取结果的列表落盘，便于复查
        save_json(model_result, f'{self.result_folder}/task4_extracted_info.json')
        
        return model_result


    def natural_language_to_json_format(self, nl_texts: list) -> dict:
        """将自然语言描述转化为结构化 JSON"""
        prompt = task4_prompt.EXTRACT_INFO  # 固定提示词模板
        orgnized_prompts = [prompt.format(original_text=nl_text) for nl_text in nl_texts]

        # 多线程向模型发送请求，提升批量抽取效率
        extracted_infos = call_llm_for_data_cleaning_or_analysis(
            client=self.client,
            model="deepseek-ai/DeepSeek-V3",
            prompts=orgnized_prompts,
            max_workers=20
        )
        return extracted_infos


    def validate_extracted_info(self, extracted_info: dict) -> bool:
        """校验抽取结果的格式是否符合预期"""
        # 判断字段是否齐全
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