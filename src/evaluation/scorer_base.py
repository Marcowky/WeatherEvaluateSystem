class BaseScorer:
    """Apply a bundle of metrics to evaluate structured predictions."""

    def __init__(self, name: str = 'base', result_folder: str = 'result/evaluation') -> None:
        self.name = name
        self.result_folder = result_folder

    def score(self, path) -> str:
        raise NotImplementedError
    
    def summary(self, path) -> str:
        raise NotImplementedError