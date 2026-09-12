"""Numerical analysis helpers for evaluating simulation quality."""

from __future__ import annotations

import math
from collections.abc import Callable

from .solvers import ODESolution


StateError = Callable[[tuple[float, ...], tuple[float, ...]], float]


def max_state_error(
    solution: ODESolution,
    exact_state: StateError,
) -> float:
    """Return the maximum error against an exact solution over all samples.

    ``exact_state`` receives each sampled time and returns the corresponding
    exact state. The error is the Euclidean norm of the state difference.
    """
    if not solution.times:
        raise ValueError("solution must contain at least one sample")

    maximum = 0.0
    for time, state in zip(solution.times, solution.states):
        expected = tuple(float(value) for value in exact_state(time, state))
        if len(expected) != len(state):
            raise ValueError("exact state has the wrong dimension")
        error = math.sqrt(sum((actual - target) ** 2 for actual, target in zip(state, expected)))
        maximum = max(maximum, error)
    return maximum
