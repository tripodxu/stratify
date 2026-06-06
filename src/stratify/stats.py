"""Statistics and snapshot utilities for Stratify simulation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stratify.agent import Agent


@dataclass(frozen=True)
class Snapshot:
    """A single-tick statistical snapshot of the simulation state."""

    tick: int
    population: int
    alive_count: int
    total_value: float
    gini: float
    class_counts: list[int]
    class_mean_value: list[float]


def gini_coefficient(values: list[float]) -> float:
    """Compute the Gini coefficient of a list of non-negative values.

    Returns 0.0 for empty or single-element lists.
    """
    if len(values) <= 1:
        return 0.0
    arr = np.array(values, dtype=float)
    if arr.sum() == 0:
        return 0.0
    arr_sorted = np.sort(arr)
    n = len(arr_sorted)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr_sorted) / (n * np.sum(arr_sorted))) - (n + 1) / n)


def class_distribution(agents: list[Agent], num_classes: int) -> list[int]:
    """Count the number of agents in each class."""
    counts = [0] * num_classes
    for a in agents:
        if 0 <= a.cls < num_classes:
            counts[a.cls] += 1
    return counts


def class_mean_values(agents: list[Agent], num_classes: int) -> list[float]:
    """Compute the mean value of agents in each class."""
    sums = [0.0] * num_classes
    counts = [0] * num_classes
    for a in agents:
        if 0 <= a.cls < num_classes:
            sums[a.cls] += a.value
            counts[a.cls] += 1
    return [
        sums[i] / counts[i] if counts[i] > 0 else 0.0
        for i in range(num_classes)
    ]


def compute_snapshot(
    agents: list[Agent], tick: int, num_classes: int
) -> Snapshot:
    """Compute a full statistical snapshot for the given tick."""
    alive = [a for a in agents if a.alive]
    all_values = [a.value for a in agents]
    return Snapshot(
        tick=tick,
        population=len(agents),
        alive_count=len(alive),
        total_value=sum(a.value for a in agents),
        gini=gini_coefficient(all_values),
        class_counts=class_distribution(agents, num_classes),
        class_mean_value=class_mean_values(agents, num_classes),
    )