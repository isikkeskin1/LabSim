"""Reproducible ensemble utilities for uncertainty propagation."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from .experiments import ExperimentConfig, run_experiment
from .models import ODEModel
from .solvers import ODESolution


@dataclass(frozen=True)
class UniformDistribution:
    """Continuous uniform distribution for one scalar model parameter."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError("uniform bounds must be finite")
        if self.high <= self.low:
            raise ValueError("high must be greater than low")

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)


@dataclass(frozen=True)
class NormalDistribution:
    """Normal distribution for one scalar model parameter."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ValueError("mean must be finite")
        if not math.isfinite(self.std) or self.std <= 0:
            raise ValueError("std must be positive and finite")

    def sample(self, rng: random.Random) -> float:
        return rng.gauss(self.mean, self.std)


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


@dataclass(frozen=True)
class EnsembleQuantiles:
    """Pointwise empirical quantiles for an ensemble."""

    times: tuple[float, ...]
    probabilities: tuple[float, ...]
    states: tuple[tuple[tuple[float, ...], ...], ...]


def sample_parameters(
    distribution: UniformDistribution | NormalDistribution,
    count: int,
    *,
    seed: int | None = None,
) -> tuple[float, ...]:
    """Draw reproducible scalar parameter samples without touching global RNG state."""
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    return tuple(distribution.sample(rng) for _ in range(count))


def run_ensemble(
    model_factory: Callable[[float], ODEModel],
    initial_state: tuple[float, ...],
    parameters: Iterable[float],
    config: ExperimentConfig,
) -> tuple[EnsembleMember, ...]:
    """Run an ensemble over scalar model-parameter samples."""
    values = tuple(float(value) for value in parameters)
    if not values:
        raise ValueError("parameters must not be empty")
    return tuple(
        EnsembleMember(value, run_experiment(model_factory(value), initial_state, config))
        for value in values
    )


def _aligned_members(members: Iterable[EnsembleMember]) -> tuple[EnsembleMember, ...]:
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
    return items


def ensemble_statistics(members: Iterable[EnsembleMember]) -> EnsembleStatistics:
    """Compute pointwise state means and population standard deviations."""
    items = _aligned_members(members)
    reference = items[0].solution
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
                ) / count
            )
            for component in range(reference.state_dimension)
        )
        mean_states.append(means)
        std_states.append(stds)

    return EnsembleStatistics(reference.times, tuple(mean_states), tuple(std_states))


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def ensemble_quantiles(
    members: Iterable[EnsembleMember],
    probabilities: Iterable[float] = (0.05, 0.5, 0.95),
) -> EnsembleQuantiles:
    """Compute linearly interpolated empirical quantiles at every trajectory sample."""
    items = _aligned_members(members)
    probs = tuple(float(value) for value in probabilities)
    if not probs:
        raise ValueError("probabilities must not be empty")
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in probs):
        raise ValueError("probabilities must lie between 0 and 1")
    if any(right <= left for left, right in zip(probs, probs[1:])):
        raise ValueError("probabilities must be strictly increasing")

    reference = items[0].solution
    quantile_states = []
    for probability in probs:
        samples = []
        for sample_index in range(len(reference)):
            samples.append(tuple(
                _quantile(
                    [member.solution.states[sample_index][component] for member in items],
                    probability,
                )
                for component in range(reference.state_dimension)
            ))
        quantile_states.append(tuple(samples))

    return EnsembleQuantiles(reference.times, probs, tuple(quantile_states))
