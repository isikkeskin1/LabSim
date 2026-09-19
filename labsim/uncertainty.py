"""Reproducible ensemble utilities for uncertainty propagation."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from .experiments import ExperimentConfig, run_experiment
from .models import ODEModel
from .solvers import ODESolution


@dataclass(frozen=True)
class UniformDistribution:
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
    mean: float
    std: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ValueError("mean must be finite")
        if not math.isfinite(self.std) or self.std <= 0:
            raise ValueError("std must be positive and finite")

    def sample(self, rng: random.Random) -> float:
        return rng.gauss(self.mean, self.std)


Distribution = UniformDistribution | NormalDistribution


@dataclass(frozen=True)
class EnsembleMember:
    parameter: float
    solution: ODESolution


@dataclass(frozen=True)
class NamedEnsembleMember:
    """One named parameter sample and the trajectory it produced."""

    parameters: tuple[tuple[str, float], ...]
    solution: ODESolution

    def parameter(self, name: str) -> float:
        for key, value in self.parameters:
            if key == name:
                return value
        raise KeyError(name)

    @property
    def parameter_dict(self) -> dict[str, float]:
        return dict(self.parameters)


@dataclass(frozen=True)
class EnsembleStatistics:
    times: tuple[float, ...]
    mean_states: tuple[tuple[float, ...], ...]
    std_states: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class EnsembleQuantiles:
    times: tuple[float, ...]
    probabilities: tuple[float, ...]
    states: tuple[tuple[tuple[float, ...], ...], ...]


def sample_parameters(distribution: Distribution, count: int, *, seed: int | None = None) -> tuple[float, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    return tuple(distribution.sample(rng) for _ in range(count))


def sample_parameter_sets(
    distributions: Mapping[str, Distribution],
    count: int,
    *,
    seed: int | None = None,
) -> tuple[dict[str, float], ...]:
    """Draw reproducible independent samples for several named parameters."""
    if count <= 0:
        raise ValueError("count must be positive")
    if not distributions:
        raise ValueError("distributions must not be empty")
    names = tuple(distributions)
    if any(not name for name in names):
        raise ValueError("parameter names must not be empty")
    rng = random.Random(seed)
    return tuple(
        {name: distributions[name].sample(rng) for name in names}
        for _ in range(count)
    )


def run_ensemble(
    model_factory: Callable[[float], ODEModel],
    initial_state: tuple[float, ...],
    parameters: Iterable[float],
    config: ExperimentConfig,
) -> tuple[EnsembleMember, ...]:
    values = tuple(float(value) for value in parameters)
    if not values:
        raise ValueError("parameters must not be empty")
    return tuple(
        EnsembleMember(value, run_experiment(model_factory(value), initial_state, config))
        for value in values
    )


def run_named_ensemble(
    model_factory: Callable[[Mapping[str, float]], ODEModel],
    initial_state: tuple[float, ...],
    parameter_sets: Iterable[Mapping[str, float]],
    config: ExperimentConfig,
) -> tuple[NamedEnsembleMember, ...]:
    """Run models built from immutable snapshots of named parameter sets."""
    samples = tuple(parameter_sets)
    if not samples:
        raise ValueError("parameter_sets must not be empty")
    members = []
    for sample in samples:
        if not sample:
            raise ValueError("parameter sets must not be empty")
        normalized = tuple((str(name), float(value)) for name, value in sample.items())
        if any(not name or not math.isfinite(value) for name, value in normalized):
            raise ValueError("parameter names must be non-empty and values finite")
        parameters = dict(normalized)
        solution = run_experiment(model_factory(parameters), initial_state, config)
        members.append(NamedEnsembleMember(normalized, solution))
    return tuple(members)


def _solutions(members: Iterable[EnsembleMember | NamedEnsembleMember]) -> tuple[ODESolution, ...]:
    items = tuple(members)
    if not items:
        raise ValueError("members must not be empty")
    solutions = tuple(member.solution for member in items)
    reference = solutions[0]
    for solution in solutions[1:]:
        if solution.times != reference.times:
            raise ValueError("ensemble trajectories must share the same time grid")
        if solution.state_dimension != reference.state_dimension:
            raise ValueError("ensemble trajectories must share the same state dimension")
    return solutions


def ensemble_statistics(members: Iterable[EnsembleMember | NamedEnsembleMember]) -> EnsembleStatistics:
    solutions = _solutions(members)
    reference = solutions[0]
    mean_states = []
    std_states = []
    count = len(solutions)
    for sample_index in range(len(reference)):
        means = tuple(
            sum(solution.states[sample_index][component] for solution in solutions) / count
            for component in range(reference.state_dimension)
        )
        stds = tuple(
            math.sqrt(sum((solution.states[sample_index][component] - means[component]) ** 2 for solution in solutions) / count)
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
    members: Iterable[EnsembleMember | NamedEnsembleMember],
    probabilities: Iterable[float] = (0.05, 0.5, 0.95),
) -> EnsembleQuantiles:
    solutions = _solutions(members)
    probs = tuple(float(value) for value in probabilities)
    if not probs:
        raise ValueError("probabilities must not be empty")
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in probs):
        raise ValueError("probabilities must lie between 0 and 1")
    if any(right <= left for left, right in zip(probs, probs[1:])):
        raise ValueError("probabilities must be strictly increasing")
    reference = solutions[0]
    quantile_states = []
    for probability in probs:
        samples = []
        for sample_index in range(len(reference)):
            samples.append(tuple(
                _quantile([solution.states[sample_index][component] for solution in solutions], probability)
                for component in range(reference.state_dimension)
            ))
        quantile_states.append(tuple(samples))
    return EnsembleQuantiles(reference.times, probs, tuple(quantile_states))
