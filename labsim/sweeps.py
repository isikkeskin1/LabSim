"""Parameter sweep utilities for systematic simulation experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .experiments import ExperimentConfig, run_experiment
from .models import ODEModel
from .solvers import ODESolution


@dataclass(frozen=True)
class SweepResult:
    """Simulation result paired with the parameter value that produced it."""

    parameter: float
    solution: ODESolution


def parameter_sweep(
    model_factory: Callable[[float], ODEModel],
    initial_state: tuple[float, ...],
    parameters: Iterable[float],
    config: ExperimentConfig,
) -> tuple[SweepResult, ...]:
    """Run the same experiment for a sequence of model parameters."""
    values = tuple(float(value) for value in parameters)
    if not values:
        raise ValueError("parameters must not be empty")

    results: list[SweepResult] = []
    for parameter in values:
        model = model_factory(parameter)
        solution = run_experiment(model, initial_state, config)
        results.append(SweepResult(parameter, solution))
    return tuple(results)
