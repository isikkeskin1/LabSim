"""Event detection and localized crossing helpers for simulation trajectories."""

from __future__ import annotations

from collections.abc import Callable

from .solvers import ODESolution

StatePredicate = Callable[[float, tuple[float, ...]], bool]


def first_event(solution: ODESolution, predicate: StatePredicate) -> tuple[float, tuple[float, ...]] | None:
    """Return the first sampled state satisfying ``predicate``."""
    for time, state in zip(solution.times, solution.states):
        if predicate(time, state):
            return time, state
    return None


def first_crossing(solution: ODESolution, component: int, threshold: float = 0.0, *, direction: int = 0) -> tuple[float, tuple[float, ...]] | None:
    """Find the first sampled threshold crossing."""
    return first_crossing_linear(solution, component, threshold, direction=direction, interpolate=False)


def first_crossing_linear(
    solution: ODESolution,
    component: int,
    threshold: float = 0.0,
    *,
    direction: int = 0,
    interpolate: bool = True,
) -> tuple[float, tuple[float, ...]] | None:
    """Find and optionally linearly localize a threshold crossing."""
    if not 0 <= component < solution.state_dimension:
        raise IndexError("component is outside the solution state dimension")
    if direction not in {-1, 0, 1}:
        raise ValueError("direction must be -1, 0, or 1")

    previous_time = solution.times[0]
    previous_state = solution.states[0]
    previous = previous_state[component] - threshold
    if previous == 0:
        return previous_time, previous_state

    for time, state in zip(solution.times[1:], solution.states[1:]):
        current = state[component] - threshold
        upward = previous < 0 <= current
        downward = previous > 0 >= current
        crossed = upward if direction == 1 else downward if direction == -1 else upward or downward
        if crossed:
            if not interpolate or current == previous:
                return time, state
            fraction = -previous / (current - previous)
            localized_time = previous_time + fraction * (time - previous_time)
            localized_state = tuple(
                before + fraction * (after - before)
                for before, after in zip(previous_state, state)
            )
            return localized_time, localized_state
        previous_time = time
        previous_state = state
        previous = current
    return None
