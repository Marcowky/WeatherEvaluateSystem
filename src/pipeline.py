from model.client import ModelClient
from task import TaskRunSummary
from evaluation import EvaluationSummary


class PipelineConfig:
    """Configuration for the evaluation pipeline."""
    def __init__(self, model: str, task: str) -> None:
        self.model = model
        self.task = task


class EvaluationPipeline:
    """Coordinates model invocation and task evaluation."""

    def __init__(self, config: PipelineConfig, model_client: ModelClient = None) -> None:
        self.config = config
        self._model_client = model_client
        self._task_run_summary = TaskRunSummary(config.task)
        self._evaluation_summary = EvaluationSummary(config.task)

    def get_model_response(self) -> str:
        self._task_run_result_path = self._task_run_summary.start(self._model_client, self.config.model)
        return self._task_run_result_path
    
    def evaluate_model_response(self, task_run_result_path: str = None) -> str:
        if task_run_result_path is not None:
            self._task_run_result_path = task_run_result_path
        self._evaluation_result_path = self._evaluation_summary.score(self._task_run_result_path)
        return self._evaluation_result_path
    
# main 函数
def main():
    model = "gpt-4"
    task = "task4"
    pipeline_config = PipelineConfig(model, task)

    # client = ModelClient()
    pipeline = EvaluationPipeline(pipeline_config)

    evaluation_result_path = pipeline.evaluate_model_response('path/to/task_run_result.json')

    print(f"Evaluation results saved to: {evaluation_result_path}")

if __name__ == "__main__":
    main()