import sys

sys.path.append('src')

from evaluation.task4.scorer import Task4Scorer

scorer = Task4Scorer()

scorer.score('aa')