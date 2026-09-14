"""Event detection and localized crossing helpers for simulation trajectories."""

from __future__ import annotations

from collections.abc import Callable

from .solvers import Derivative, ODESolution

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


def _validate_crossing_request(solution: ODESolution, component: int, direction: int) -> None:
    if not 0 <= component < solution.state_dimension:
        raise IndexError("component is outside the solution state dimension")
    if direction not in {-1, 0, 1}:
        raise ValueError("direction must be -1, 0, or 1")


def _crossed(previous: float, current: float, direction: int) -> bool:
    upward = previous < 0 <= current
    downward = previous > 0 >= current
    return upward if direction == 1 else downward if direction == -1 else upward or downward


def first_crossing_linear(
    solution: ODESolution,
    component: int,
    threshold: float = 0.0,
    *,
    direction: int = 0,
    interpolate: bool = True,
) -> tuple[float, tuple[float, ...]] | None:
    """Find and optionally linearly localize a threshold crossing."""
    _validate_crossing_request(solution, component, direction)

    previous_time = solution.times[0]
    previous_state = solution.states[0]
    previous = previous_state[component] - threshold
    if previous == 0:
        return previous_time, previous_state

    for time, state in zip(solution.times[1:], solution.states[1:]):
        current = state[component] - threshold
        if _crossed(previous, current, direction):
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


def first_crossing_hermite(
    solution: ODESolution,
    derivative: Derivative,
    component: int,
    threshold: float = 0.0,
    *,
    direction: int = 0,
    iterations: int = 40,
) -> tuple[float, tuple[float, ...]] | None:
    """Localize a crossing with cubic Hermite interpolation inside its sample bracket.

    Endpoint states and endpoint derivatives define a cubic interpolant for every
    state component. Once a sampled interval brackets the requested threshold,
    bisection on that interpolant provides a stable dense-output estimate without
    changing or re-integrating the original trajectory.
    """
    _validate_crossing_request(solution, component, direction)
    if iterations < 1:
        raise ValueError("iterations must be positive")

    previous_time = solution.times[0]
    previous_state = solution.states[0]
    previous_value = previous_state[component] - threshold
    if previous_value == 0:
        return previous_time, previous_state

    for time, state in zip(solution.times[1:], solution.states[1:]):
        current_value = state[component] - threshold
        if _crossed(previous_value, current_value, direction):
            width = time - previous_time
            if width <= 0:
                raise ValueError("solution times must be strictly increasing")

            start_derivative = tuple(float(value) for value in derivative(previous_time, previous_state))
            end_derivative = tuple(float(value) for value in derivative(time, state))
            if len(start_derivative) != solution.state_dimension or len(end_derivative) != solution.state_dimension:
                raise ValueError("derivative returned a vector with the wrong dimension")

            def interpolate(fraction: float) -> tuple[float, ...]:
                f2 = fraction * fraction
                f3 = f2 * fraction
                h00 = 2 * f3 - 3 * f2 + 1
                h10 = f3 - 2 * f2 + fraction
                h01 = -2 * f3 + 3 * f2
                h11 = f3 - f2
                return tuple(
                    h00 * before
                    + h10 * width * slope_before
                    + h01 * after
                    + h11 * width * slope_after
                    for before, after, slope_before, slope_after in zip(
                        previous_state, state, start_derivative, end_derivative
                    )
                )

            low = 0.0
            high = 1.0
            low_value = previous_value
            localized_state = previous_state
            fraction = 0.0
            for _ in range(iterations):
                fraction = (low + high) / 2
                localized_state = interpolate(fraction)
                middle_value = localized_state[component] - threshold
                if middle_value == 0:
                    break
                if (low_value < 0 <= middle_value) or (low_value > 0 >= middle_value):
                    high = fraction
                else:
                    low = fraction
                    low_value = middle_value

            localized_time = previous_time + fraction * width
            return localized_time, localized_state

        previous_time = time
        previous_state = state
        previous_value = current_value
    return None
