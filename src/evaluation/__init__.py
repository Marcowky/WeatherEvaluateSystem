from .task4.scorer import Task4Scorer

registered_evaluations = {
    "task4": Task4Scorer,
}

def EvaluationSummary(task: str):
    return registered_evaluations[task]()