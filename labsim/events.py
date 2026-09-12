"""Event detection helpers for sampled simulation trajectories."""

from __future__ import annotations

from collections.abc import Callable

from .solvers import ODESolution

StatePredicate = Callable[[float, tuple[float, ...]], bool]


def first_event(
    solution: ODESolution,
    predicate: StatePredicate,
) -> tuple[float, tuple[float, ...]] | None:
    """Return the first sampled state satisfying ``predicate``."""
    for time, state in zip(solution.times, solution.states):
        if predicate(time, state):
            return time, state
    return None


def first_crossing(
    solution: ODESolution,
    component: int,
    threshold: float = 0.0,
    *,
    direction: int = 0,
) -> tuple[float, tuple[float, ...]] | None:
    """Find the first sampled crossing of a scalar state component.

    ``direction`` is ``0`` for either direction, ``1`` for upward, and ``-1``
    for downward crossings. The returned point is the first sampled value
    on or beyond the threshold; interpolation is intentionally left to a
    future event-localization API.
    """
    if not 0 <= component < solution.state_dimension:
        raise IndexError("component is outside the solution state dimension")
    if direction not in {-1, 0, 1}:
        raise ValueError("direction must be -1, 0, or 1")

    previous = solution.states[0][component] - threshold
    if previous == 0:
        return solution.times[0], solution.states[0]

    for time, state in zip(solution.times[1:], solution.states[1:]):
        current = state[component] - threshold
        upward = previous < 0 <= current
        downward = previous > 0 >= current
        crossed = upward if direction == 1 else downward if direction == -1 else upward or downward
        if crossed:
            return time, state
        previous = current
    return None
