from ..scorer_two_stage import TwoStageScorer

class Task4Scorer(TwoStageScorer):
    def __init__(self, result_folder: str = 'result/evaluation') -> None:
        super().__init__(result_folder=result_folder)
        self.name = 'task4'

    def info_extract(self, model_result):
        # Implement the extraction logic specific to Task 4
        extracted_info = {}
        # ... extraction code ...
        return extracted_info

    def info_scoring(self, extracted_info):
        # Implement the scoring logic specific to Task 4
        scoring_result = {}
        # ... scoring code ...
        return scoring_result