"""Common trajectory and simulation quality metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .solvers import ODESolution


def final_state_error(solution: ODESolution, expected: Sequence[float]) -> float:
    """Return Euclidean error between the final state and an expected state."""
    if not solution.states:
        raise ValueError("solution must contain at least one state")
    actual = solution.states[-1]
    if len(actual) != len(expected):
        raise ValueError("expected state has the wrong dimension")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(actual, expected)))


def root_mean_square(values: Sequence[float]) -> float:
    """Return the RMS magnitude of a non-empty sequence."""
    if not values:
        raise ValueError("values must not be empty")
    return math.sqrt(sum(float(value) ** 2 for value in values) / len(values))
