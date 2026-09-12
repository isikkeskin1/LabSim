"""Numerical integration primitives used by LabSim experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

State = Sequence[float]
Derivative = Callable[[float, State], Sequence[float]]


@dataclass(frozen=True)
class ODESolution:
    """Sampled solution of an initial-value ordinary differential equation."""

    times: tuple[float, ...]
    states: tuple[tuple[float, ...], ...]

    def __len__(self) -> int:
        return len(self.times)


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
    """Integrate ``y' = f(t, y)`` on a fixed time grid.

    ``method`` currently supports ``"euler"`` and classical fourth-order
    Runge-Kutta (``"rk4"``). The initial state is always included in the
    returned samples.
    """
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
