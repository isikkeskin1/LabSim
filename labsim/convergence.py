"""Tools for measuring numerical convergence of ODE solvers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from .analysis import max_state_error
from .solvers import ODESolution, integrate_ode

State = Sequence[float]
Derivative = Callable[[float, State], Sequence[float]]
ExactSolution = Callable[[float], State]


@dataclass(frozen=True)
class ConvergencePoint:
    """Error measured for one time-step size."""

    dt: float
    error: float


@dataclass(frozen=True)
class ConvergenceStudy:
    """Collection of error measurements for a solver configuration."""

    method: str
    points: tuple[ConvergencePoint, ...]

    @property
    def observed_order(self) -> float | None:
        if len(self.points) < 2:
            return None
        first, last = self.points[0], self.points[-1]
        if first.error <= 0 or last.error <= 0 or first.dt == last.dt:
            return None
        return math.log(first.error / last.error) / math.log(first.dt / last.dt)


def run_convergence_study(
    derivative: Derivative,
    initial_state: State,
    exact: ExactSolution,
    *,
    duration: float,
    step_sizes: Sequence[float],
    method: str = "rk4",
) -> ConvergenceStudy:
    """Run the same IVP at several step sizes and estimate error order."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if not step_sizes:
        raise ValueError("step_sizes must not be empty")
    if any(dt <= 0 for dt in step_sizes):
        raise ValueError("step sizes must be positive")

    points: list[ConvergencePoint] = []
    for dt in step_sizes:
        steps = int(round(duration / dt))
        if steps < 1 or not math.isclose(steps * dt, duration, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("duration must be an integer multiple of every step size")
        solution: ODESolution = integrate_ode(
            derivative, initial_state, dt=dt, steps=steps, method=method
        )
        error = max_state_error(
            solution,
            lambda time, _state: exact(time),
        )
        points.append(ConvergencePoint(dt, error))

    return ConvergenceStudy(method, tuple(points))
