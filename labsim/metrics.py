"""Common trajectory and simulation quality metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .solvers import ODESolution


def final_state_error(solution: ODESolution, expected: Sequence[float]) -> float:
    """Return Euclidean error between the final state and an expected state."""
    actual = solution.final_state
    if len(actual) != len(expected):
        raise ValueError("expected state has the wrong dimension")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(actual, expected)))


def root_mean_square(values: Sequence[float]) -> float:
    """Return the RMS magnitude of a non-empty sequence."""
    if not values:
        raise ValueError("values must not be empty")
    return math.sqrt(sum(float(value) ** 2 for value in values) / len(values))


def state_range(solution: ODESolution, component: int) -> float:
    """Return max-min for one state component across a trajectory."""
    if not 0 <= component < solution.state_dimension:
        raise IndexError("component is outside the solution state dimension")
    values = [state[component] for state in solution.states]
    return max(values) - min(values)


def time_average(solution: ODESolution, component: int) -> float:
    """Approximate a component's time average using the trapezoidal rule."""
    if not 0 <= component < solution.state_dimension:
        raise IndexError("component is outside the solution state dimension")
    if len(solution) == 1:
        return solution.states[0][component]
    area = 0.0
    for left_time, right_time, left_state, right_state in zip(
        solution.times,
        solution.times[1:],
        solution.states,
        solution.states[1:],
    ):
        area += (right_time - left_time) * (left_state[component] + right_state[component]) / 2
    duration = solution.final_time - solution.times[0]
    if duration <= 0:
        raise ValueError("solution times must span a positive duration")
    return area / duration
