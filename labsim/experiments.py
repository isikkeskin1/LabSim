"""Reproducible experiment configuration and execution helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ODEModel
from .solvers import ODESolution, integrate_ode


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete numerical configuration for one deterministic simulation."""

    t0: float = 0.0
    duration: float = 1.0
    dt: float = 0.01
    method: str = "rk4"

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.method not in {"euler", "rk4"}:
            raise ValueError("method must be 'euler' or 'rk4'")

    @property
    def steps(self) -> int:
        steps = round(self.duration / self.dt)
        if steps < 1 or abs(steps * self.dt - self.duration) > 1e-10:
            raise ValueError("duration must be an integer multiple of dt")
        return steps


def run_experiment(model: ODEModel, initial_state: tuple[float, ...], config: ExperimentConfig) -> ODESolution:
    """Run a model using a validated, reusable experiment configuration."""
    if len(initial_state) != model.state_size:
        raise ValueError("initial_state does not match model state_size")
    return integrate_ode(
        model,
        initial_state,
        t0=config.t0,
        dt=config.dt,
        steps=config.steps,
        method=config.method,
    )
