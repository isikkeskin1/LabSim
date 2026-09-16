"""Dense trajectory reconstruction utilities.

These helpers reconstruct states between accepted solver samples without
re-integrating the system. They are useful for plotting, uniform output grids,
and post-processing adaptive trajectories whose accepted samples are irregular.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterable, Sequence

from .solvers import ODESolution

State = Sequence[float]
Derivative = Callable[[float, State], Sequence[float]]


def _hermite_state(
    left_time: float,
    right_time: float,
    left_state: tuple[float, ...],
    right_state: tuple[float, ...],
    left_derivative: tuple[float, ...],
    right_derivative: tuple[float, ...],
    time: float,
) -> tuple[float, ...]:
    step = right_time - left_time
    if step <= 0:
        raise ValueError("solution times must be strictly increasing")
    if not (
        len(left_state)
        == len(right_state)
        == len(left_derivative)
        == len(right_derivative)
    ):
        raise ValueError("state and derivative dimensions must match")

    theta = (time - left_time) / step
    theta2 = theta * theta
    theta3 = theta2 * theta
    h00 = 2 * theta3 - 3 * theta2 + 1
    h10 = theta3 - 2 * theta2 + theta
    h01 = -2 * theta3 + 3 * theta2
    h11 = theta3 - theta2

    return tuple(
        h00 * y0 + h10 * step * f0 + h01 * y1 + h11 * step * f1
        for y0, y1, f0, f1 in zip(
            left_state, right_state, left_derivative, right_derivative
        )
    )


def sample_hermite(
    solution: ODESolution,
    derivative: Derivative,
    times: Iterable[float],
) -> ODESolution:
    """Sample a completed trajectory at arbitrary in-domain times.

    Cubic Hermite interpolation uses the state and governing derivative at both
    endpoints of each accepted step. Requested times may be irregular but must
    be non-decreasing and lie within the solution's time domain. Exact existing
    samples are returned unchanged.
    """
    requested = tuple(float(time) for time in times)
    if not requested:
        raise ValueError("times must not be empty")
    if any(right < left for left, right in zip(requested, requested[1:])):
        raise ValueError("requested times must be non-decreasing")
    if requested[0] < solution.times[0] or requested[-1] > solution.times[-1]:
        raise ValueError("requested times must lie within the solution domain")

    derivatives: dict[int, tuple[float, ...]] = {}

    def endpoint_derivative(index: int) -> tuple[float, ...]:
        if index not in derivatives:
            values = tuple(
                float(value)
                for value in derivative(solution.times[index], solution.states[index])
            )
            if len(values) != solution.state_dimension:
                raise ValueError("derivative returned a vector with the wrong dimension")
            derivatives[index] = values
        return derivatives[index]

    sampled: list[tuple[float, ...]] = []
    for time in requested:
        index = bisect_right(solution.times, time) - 1
        if solution.times[index] == time or index == len(solution.times) - 1:
            sampled.append(solution.states[index])
            continue

        sampled.append(
            _hermite_state(
                solution.times[index],
                solution.times[index + 1],
                solution.states[index],
                solution.states[index + 1],
                endpoint_derivative(index),
                endpoint_derivative(index + 1),
                time,
            )
        )

    return ODESolution(requested, tuple(sampled))


def resample_uniform(
    solution: ODESolution,
    derivative: Derivative,
    *,
    dt: float,
) -> ODESolution:
    """Reconstruct a trajectory on a uniform grid, including both endpoints."""
    if dt <= 0:
        raise ValueError("dt must be positive")

    start = solution.times[0]
    end = solution.times[-1]
    times = [start]
    time = start
    while time + dt < end:
        time += dt
        times.append(time)
    if times[-1] != end:
        times.append(end)
    return sample_hermite(solution, derivative, times)
