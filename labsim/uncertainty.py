"""Deterministic ensemble utilities for uncertainty propagation."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .experiments import ExperimentConfig, run_experiment
from .models import ODEModel
from .solvers import ODESolution


@dataclass(frozen=True)
class EnsembleMember:
    """One parameter sample and the trajectory it produced."""

    parameter: float
    solution: ODESolution


@dataclass(frozen=True)
class EnsembleStatistics:
    """Pointwise mean and population standard deviation of an ensemble."""

    times: tuple[float, ...]
    mean_states: tuple[tuple[float, ...], ...]
    std_states: tuple[tuple[float, ...], ...]


def run_ensemble(
    model_factory: Callable[[float], ODEModel],
    initial_state: tuple[float, ...],
    parameters: Iterable[float],
    config: ExperimentConfig,
) -> tuple[EnsembleMember, ...]:
    """Run a deterministic ensemble over scalar model-parameter samples."""
    values = tuple(float(value) for value in parameters)
    if not values:
        raise ValueError("parameters must not be empty")
    return tuple(
        EnsembleMember(value, run_experiment(model_factory(value), initial_state, config))
        for value in values
    )


def ensemble_statistics(members: Iterable[EnsembleMember]) -> EnsembleStatistics:
    """Compute pointwise state means and population standard deviations.

    Ensemble trajectories must share an identical time grid and state dimension.
    This deliberate constraint keeps aggregation exact and predictable; adaptive
    trajectories can first be resampled with ``resample_uniform``.
    """
    items = tuple(members)
    if not items:
        raise ValueError("members must not be empty")

    reference = items[0].solution
    for member in items[1:]:
        solution = member.solution
        if solution.times != reference.times:
            raise ValueError("ensemble trajectories must share the same time grid")
        if solution.state_dimension != reference.state_dimension:
            raise ValueError("ensemble trajectories must share the same state dimension")

    mean_states: list[tuple[float, ...]] = []
    std_states: list[tuple[float, ...]] = []
    count = len(items)

    for sample_index in range(len(reference)):
        means = tuple(
            sum(member.solution.states[sample_index][component] for member in items) / count
            for component in range(reference.state_dimension)
        )
        stds = tuple(
            math.sqrt(
                sum(
                    (member.solution.states[sample_index][component] - means[component]) ** 2
                    for member in items
                )
                / count
            )
            for component in range(reference.state_dimension)
        )
        mean_states.append(means)
        std_states.append(stds)

    return EnsembleStatistics(reference.times, tuple(mean_states), tuple(std_states))
