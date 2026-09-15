"""Higher-order adaptive integration methods."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from .solvers import ODESolution

State = Sequence[float]
Derivative = Callable[[float, State], Sequence[float]]


def _combine(state: tuple[float, ...], step: float, *terms: tuple[float, tuple[float, ...]]) -> tuple[float, ...]:
    size = len(state)
    if any(len(vector) != size for _, vector in terms):
        raise ValueError("derivative vectors must have the same dimension as the state")
    return tuple(
        value + step * sum(weight * vector[index] for weight, vector in terms)
        for index, value in enumerate(state)
    )


def integrate_rk23(
    derivative: Derivative,
    initial_state: State,
    *,
    t0: float = 0.0,
    duration: float = 1.0,
    dt: float = 0.01,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    min_dt: float = 1e-10,
    max_dt: float | None = None,
    max_steps: int = 100_000,
) -> ODESolution:
    """Integrate an IVP with the embedded Bogacki-Shampine RK2(3) pair.

    The accepted state uses the third-order formula while the embedded
    second-order estimate controls the next step size. The final accepted
    sample is forced to ``t0 + duration``.
    """
    if duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must be positive")
    if rtol <= 0 or atol <= 0:
        raise ValueError("rtol and atol must be positive")
    if min_dt <= 0 or min_dt > dt:
        raise ValueError("min_dt must be positive and no larger than dt")
    if max_dt is not None and max_dt < min_dt:
        raise ValueError("max_dt must be at least min_dt")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if not initial_state:
        raise ValueError("initial_state must contain at least one value")

    state = tuple(float(value) for value in initial_state)
    time = float(t0)
    end_time = time + float(duration)
    step = min(dt, max_dt) if max_dt is not None else dt
    times = [time]
    states = [state]
    attempts = 0

    while time < end_time:
        if attempts >= max_steps:
            raise RuntimeError("RK23 solver exceeded max_steps")
        attempts += 1
        step = min(step, end_time - time)

        k1 = tuple(float(value) for value in derivative(time, state))
        y2 = _combine(state, step, (0.5, k1))
        k2 = tuple(float(value) for value in derivative(time + 0.5 * step, y2))
        y3_stage = _combine(state, step, (0.0, k1), (0.75, k2))
        k3 = tuple(float(value) for value in derivative(time + 0.75 * step, y3_stage))

        third_order = _combine(
            state, step, (2 / 9, k1), (1 / 3, k2), (4 / 9, k3)
        )
        k4 = tuple(float(value) for value in derivative(time + step, third_order))
        second_order = _combine(
            state,
            step,
            (7 / 24, k1),
            (1 / 4, k2),
            (1 / 3, k3),
            (1 / 8, k4),
        )

        error_ratio = 0.0
        for high, low, previous in zip(third_order, second_order, state):
            scale = atol + rtol * max(abs(previous), abs(high))
            error_ratio = max(error_ratio, abs(high - low) / scale)

        if not math.isfinite(error_ratio):
            raise RuntimeError("RK23 solver encountered a non-finite error estimate")

        if error_ratio <= 1.0:
            time += step
            state = third_order
            times.append(time)
            states.append(state)

        if error_ratio == 0.0:
            factor = 5.0
        else:
            factor = min(5.0, max(0.2, 0.9 * error_ratio ** (-1 / 3)))

        proposed = step * factor
        if max_dt is not None:
            proposed = min(proposed, max_dt)
        if error_ratio > 1.0 and proposed < min_dt:
            if step <= min_dt * (1 + 1e-12):
                raise RuntimeError("RK23 solver reached min_dt before satisfying tolerance")
            proposed = min_dt
        step = max(min_dt, proposed)

    return ODESolution(tuple(times), tuple(states))
