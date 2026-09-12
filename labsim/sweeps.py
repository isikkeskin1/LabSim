"""Parameter sweep utilities for systematic simulation experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .experiments import ExperimentConfig, run_experiment
from .metrics import root_mean_square
from .models import ODEModel
from .solvers import ODESolution


@dataclass(frozen=True)
class SweepResult:
    """Simulation result paired with the parameter value that produced it."""

    parameter: float
    solution: ODESolution


@dataclass(frozen=True)
class SweepSummary:
    """Compact scalar summary for each sweep point."""

    parameter: float
    final_state: tuple[float, ...]
    final_norm: float


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


def summarize_sweep(results: Iterable[SweepResult]) -> tuple[SweepSummary, ...]:
    """Reduce sweep trajectories to final-state norms for reporting."""
    values = tuple(results)
    if not values:
        raise ValueError("results must not be empty")
    return tuple(
        SweepSummary(
            parameter=result.parameter,
            final_state=result.solution.final_state,
            final_norm=root_mean_square(result.solution.final_state),
        )
        for result in values
    )
