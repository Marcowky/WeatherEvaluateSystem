from evaluation.scorer_two_stage import TwoStageScorer
from util.multi_thread import run_in_threads
from model.client import ModelClient
from prompt.evaluation_prompt import task4_prompt
from util.data_process import save_json, str_to_json

class Task4Scorer(TwoStageScorer):
    """Task4 评分器：负责信息抽取+两阶段评分的调度"""
    def __init__(self, result_folder: str = 'result/evaluation') -> None:
        super().__init__(result_folder=result_folder)
        self.name = 'task4'

    def info_extract(self, model_result):
        # 处理 model_result，提取所需信息
        self.client = ModelClient()  # 初始化客户端，避免在多线程中重复创建

        extracted_infos = self.natural_language_to_json_format(
            [res['model_output'] for res in model_result]
        )
        
        # 针对每条模型输出进行解析与校验
        for i, extracted_info in enumerate(extracted_infos):
            try:
                json_res = str_to_json(extracted_info)
                if not self.validate_extracted_info(json_res):
                    print(f"Invalid extracted_info format for index {i}: {json_res}")
                model_result[i]['extracted_info'] = json_res
            except Exception as e:
                print(f"Error parsing extracted_info for index {i}: {e}")
        
        # 将带有抽取结果的列表落盘，便于复查
        save_json(model_result, f'{self.result_folder}/task4_extracted_info.json')
        
        return model_result

    def info_scoring(self, extracted_info):
        # Implement the scoring logic specific to Task 4
        scoring_result = {}
        # ... scoring code ...
        return scoring_result
    
    def natural_language_to_json_format(self, nl_texts: list) -> dict:
        # 将自然语言描述转化为结构化 JSON
        prompt = task4_prompt.EXTRACT_INFO  # 固定提示词模板
        orgnized_prompts = [prompt.format(original_text=nl_text) for nl_text in nl_texts]

        args_list_dict = {
            "model": "deepseek-ai/DeepSeek-V3",
            "prompt": orgnized_prompts,
            "max_tokens": 2048,
            "temperature": 0.1,
            "response_format": {'type': 'json_object'},
        }

        # 多线程向模型发送请求，提升批量抽取效率
        extracted_infos = run_in_threads(self.client.chat_with_prompt_return_text, args_list_dict, max_workers=5)
        return extracted_infos
    
    def validate_extracted_info(self, extracted_info: dict) -> bool:
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