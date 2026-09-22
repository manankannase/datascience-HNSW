"""Query-dependent cluster budgets for clustered HNSW experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AdaptiveClusterBudget:
    """Choose a cluster-search budget from the nearest-centroid margin.

    A larger margin means the nearest centroid is much closer than the second
    nearest centroid, so the query is more likely to need fewer clusters.
    ``thresholds`` maps increasingly difficult queries to larger budgets.
    """

    budgets: tuple[int, ...] = (1, 2, 4, 8)
    thresholds: tuple[float, ...] = (0.05, 0.20, 0.50)
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if not self.budgets or any(value < 1 for value in self.budgets):
            raise ValueError("budgets must contain positive integers")
        if tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("budgets must be unique and sorted")
        if len(self.thresholds) != len(self.budgets) - 1:
            raise ValueError("thresholds must have len(budgets) - 1 values")
        if tuple(sorted(self.thresholds)) != self.thresholds:
            raise ValueError("thresholds must be sorted")

    def margin(self, centroid_distances: Sequence[float]) -> float:
        """Return normalized gap between the two closest centroids."""
        distances = np.asarray(centroid_distances, dtype=float)
        if distances.ndim != 1 or len(distances) < 2:
            raise ValueError("at least two centroid distances are required")
        if not np.isfinite(distances).all() or (distances < 0).any():
            raise ValueError("centroid distances must be finite and non-negative")
        nearest, second = np.partition(distances, 1)[:2]
        return float((second - nearest) / max(nearest, self.epsilon))

    def choose(self, centroid_distances: Sequence[float]) -> int:
        """Return the number of clusters to search for this query."""
        value = self.margin(centroid_distances)
        for threshold, budget in zip(reversed(self.thresholds), self.budgets):
            if value >= threshold:
                return budget
        return self.budgets[-1]
