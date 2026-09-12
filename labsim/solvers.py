"""Numerical integration primitives used by LabSim experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

State = Sequence[float]
Derivative = Callable[[float, State], Sequence[float]]


@dataclass(frozen=True)
class ODESolution:
    """Sampled solution of an initial-value ordinary differential equation."""

    times: tuple[float, ...]
    states: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if len(self.times) != len(self.states):
            raise ValueError("times and states must contain the same number of samples")
        if not self.times:
            raise ValueError("solution must contain at least one sample")
        width = len(self.states[0])
        if width == 0 or any(len(state) != width for state in self.states):
            raise ValueError("all states must have the same non-zero dimension")

    def __len__(self) -> int:
        return len(self.times)

    @property
    def final_time(self) -> float:
        return self.times[-1]

    @property
    def final_state(self) -> tuple[float, ...]:
        return self.states[-1]

    @property
    def state_dimension(self) -> int:
        return len(self.states[0])


def _add_scaled(state: State, *terms: tuple[float, State]) -> tuple[float, ...]:
    """Return state + sum(scale * vector) with explicit shape checking."""
    if not terms:
        return tuple(state)
    size = len(state)
    if any(len(vector) != size for _, vector in terms):
        raise ValueError("derivative vectors must have the same dimension as the state")
    return tuple(
        value + sum(scale * vector[i] for scale, vector in terms)
        for i, value in enumerate(state)
    )


def integrate_ode(
    derivative: Derivative,
    initial_state: State,
    *,
    t0: float = 0.0,
    dt: float = 0.01,
    steps: int = 1,
    method: str = "rk4",
) -> ODESolution:
    """Integrate ``y' = f(t, y)`` on a fixed time grid."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not initial_state:
        raise ValueError("initial_state must contain at least one value")
    if method not in {"euler", "rk4"}:
        raise ValueError("method must be 'euler' or 'rk4'")

    state = tuple(float(value) for value in initial_state)
    times = [float(t0)]
    states = [state]
    time = float(t0)

    for _ in range(steps):
        if method == "euler":
            k1 = tuple(float(value) for value in derivative(time, state))
            state = _add_scaled(state, (dt, k1))
        else:
            k1 = tuple(float(value) for value in derivative(time, state))
            k2_state = _add_scaled(state, (dt / 2, k1))
            k2 = tuple(float(value) for value in derivative(time + dt / 2, k2_state))
            k3_state = _add_scaled(state, (dt / 2, k2))
            k3 = tuple(float(value) for value in derivative(time + dt / 2, k3_state))
            k4_state = _add_scaled(state, (dt, k3))
            k4 = tuple(float(value) for value in derivative(time + dt, k4_state))
            state = _add_scaled(
                state,
                (dt / 6, k1),
                (dt / 3, k2),
                (dt / 3, k3),
                (dt / 6, k4),
            )

        if len(state) != len(initial_state):
            raise ValueError("derivative returned a vector with the wrong dimension")
        time += dt
        times.append(time)
        states.append(state)

    return ODESolution(tuple(times), tuple(states))


def integrate_adaptive(
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
    """Integrate with adaptive Heun-Euler error control."""
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
    accepted_steps = 0

    while time < end_time:
        if accepted_steps >= max_steps:
            raise RuntimeError("adaptive solver exceeded max_steps")
        step = min(step, end_time - time)

        k1 = tuple(float(value) for value in derivative(time, state))
        euler = _add_scaled(state, (step, k1))
        k2 = tuple(float(value) for value in derivative(time + step, euler))
        heun = _add_scaled(state, (step / 2, k1), (step / 2, k2))
        if len(k1) != len(state) or len(k2) != len(state):
            raise ValueError("derivative vectors must have the same dimension as the state")

        error_ratio = 0.0
        for actual, estimate, previous in zip(heun, euler, state):
            scale = atol + rtol * max(abs(previous), abs(actual))
            error_ratio = max(error_ratio, abs(actual - estimate) / scale)

        if error_ratio <= 1.0:
            time += step
            state = tuple(heun)
            times.append(time)
            states.append(state)
            accepted_steps += 1
            factor = 5.0 if error_ratio == 0 else min(5.0, max(0.2, 0.9 * error_ratio ** -0.5))
            step = min(step * factor, max_dt) if max_dt is not None else step * factor
        else:
            if step <= min_dt * (1 + 1e-12):
                raise RuntimeError("adaptive solver reached min_dt before satisfying tolerance")
            factor = max(0.2, 0.9 * error_ratio ** -0.5)
            step = max(min_dt, step * factor)

    return ODESolution(tuple(times), tuple(states))
