from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .metric import Metric, MetricResult


class BaseScorer:
	"""Apply a bundle of metrics to evaluate structured predictions."""

	def __init__(self) -> None:
		pass

	def score(self, path) -> str:
		print(f"Scoring using {self.name} on data at: {path}")
		return path