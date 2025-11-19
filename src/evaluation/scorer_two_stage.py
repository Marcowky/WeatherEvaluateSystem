from .scorer_base import BaseScorer
from util.data_process import load_json, save_json
from util.file_timestamp import get_timestamp

class TwoStageScorer(BaseScorer):
    def __init__(self, result_folder: str = 'result/evaluation') -> None:
        super().__init__(name='two_stage', result_folder=result_folder)

    def info_extract(self, model_result):
        raise NotImplementedError
        
    def info_scoring(self, model_result):
        raise NotImplementedError

    def score(self, path) -> str:
        model_result = load_json(path)
        extracted_info = self.info_extract(model_result)
        scoring_result = self.info_scoring(extracted_info)
        save_path = f"{self.result_folder}/{self.name}_scoring_result_{get_timestamp()}.json"
        return save_json(scoring_result, save_path)