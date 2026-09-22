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

    def quantile(self, probability: float) -> float:
        return self.low + probability * (self.high - self.low)


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

    def quantile(self, probability: float) -> float:
        if probability <= 0 or probability >= 1:
            raise ValueError("normal quantile probability must lie strictly between 0 and 1")
        return self.mean + self.std * _standard_normal_quantile(probability)


Distribution = UniformDistribution | NormalDistribution


@dataclass(frozen=True)
class EnsembleMember:
    parameter: float
    solution: ODESolution


@dataclass(frozen=True)
class NamedEnsembleMember:
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


@dataclass(frozen=True)
class MonteCarloEstimate:
    samples: int
    mean: float
    std: float
    standard_error: float


@dataclass(frozen=True)
class MonteCarloConvergence:
    estimates: tuple[MonteCarloEstimate, ...]

    @property
    def final(self) -> MonteCarloEstimate:
        return self.estimates[-1]


@dataclass(frozen=True)
class SamplingEfficiency:
    """Replicated estimator variability for Monte Carlo and Latin hypercube designs."""

    samples_per_replication: int
    replications: int
    monte_carlo_mean: float
    latin_hypercube_mean: float
    monte_carlo_std: float
    latin_hypercube_std: float

    @property
    def variance_ratio(self) -> float:
        """Monte Carlo variance divided by Latin-hypercube variance."""
        lhs_variance = self.latin_hypercube_std ** 2
        if lhs_variance == 0.0:
            return math.inf if self.monte_carlo_std > 0.0 else 1.0
        return self.monte_carlo_std ** 2 / lhs_variance


def _standard_normal_quantile(probability: float) -> float:
    """Approximate the inverse standard-normal CDF (Acklam rational fit)."""
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416)
    low = 0.02425
    high = 1.0 - low
    if probability < low:
        q = math.sqrt(-2.0 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if probability <= high:
        q = probability - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - probability))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def sample_parameters(distribution: Distribution, count: int, *, seed: int | None = None) -> tuple[float, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    return tuple(distribution.sample(rng) for _ in range(count))


def sample_parameter_sets(distributions: Mapping[str, Distribution], count: int, *, seed: int | None = None) -> tuple[dict[str, float], ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    if not distributions:
        raise ValueError("distributions must not be empty")
    names = tuple(distributions)
    if any(not name for name in names):
        raise ValueError("parameter names must not be empty")
    rng = random.Random(seed)
    return tuple({name: distributions[name].sample(rng) for name in names} for _ in range(count))


def latin_hypercube_parameter_sets(distributions: Mapping[str, Distribution], count: int, *, seed: int | None = None) -> tuple[dict[str, float], ...]:
    """Draw a Latin-hypercube design from independent named marginals."""
    if count <= 0:
        raise ValueError("count must be positive")
    if not distributions:
        raise ValueError("distributions must not be empty")
    names = tuple(distributions)
    if any(not name for name in names):
        raise ValueError("parameter names must not be empty")
    rng = random.Random(seed)
    columns: dict[str, tuple[float, ...]] = {}
    for name in names:
        probabilities = [(index + rng.random()) / count for index in range(count)]
        values = [distributions[name].quantile(probability) for probability in probabilities]
        rng.shuffle(values)
        columns[name] = tuple(values)
    return tuple({name: columns[name][row] for name in names} for row in range(count))


def compare_sampling_efficiency(
    distributions: Mapping[str, Distribution],
    observable: Callable[[Mapping[str, float]], float],
    *,
    samples: int,
    replications: int = 32,
    seed: int | None = None,
) -> SamplingEfficiency:
    """Compare replicated estimator variability for plain MC and Latin hypercube.

    The observable is evaluated directly on parameter sets, so this diagnostic can
    be used for cheap analytical checks or for expensive simulation-derived scalar
    outputs supplied by a caller. Each replication receives deterministic child
    seeds from a local RNG; global random state is never touched.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    if replications < 2:
        raise ValueError("replications must be at least 2")
    if not distributions:
        raise ValueError("distributions must not be empty")

    rng = random.Random(seed)
    mc_estimates: list[float] = []
    lhs_estimates: list[float] = []
    for _ in range(replications):
        mc_seed = rng.randrange(2**63)
        lhs_seed = rng.randrange(2**63)
        mc_design = sample_parameter_sets(distributions, samples, seed=mc_seed)
        lhs_design = latin_hypercube_parameter_sets(distributions, samples, seed=lhs_seed)
        mc_values = [float(observable(point)) for point in mc_design]
        lhs_values = [float(observable(point)) for point in lhs_design]
        if any(not math.isfinite(value) for value in (*mc_values, *lhs_values)):
            raise ValueError("observable must return finite values")
        mc_estimates.append(sum(mc_values) / samples)
        lhs_estimates.append(sum(lhs_values) / samples)

    def summarize(values: Sequence[float]) -> tuple[float, float]:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return mean, math.sqrt(variance)

    mc_mean, mc_std = summarize(mc_estimates)
    lhs_mean, lhs_std = summarize(lhs_estimates)
    return SamplingEfficiency(samples, replications, mc_mean, lhs_mean, mc_std, lhs_std)


def run_ensemble(model_factory: Callable[[float], ODEModel], initial_state: tuple[float, ...], parameters: Iterable[float], config: ExperimentConfig) -> tuple[EnsembleMember, ...]:
    values = tuple(float(value) for value in parameters)
    if not values:
        raise ValueError("parameters must not be empty")
    return tuple(EnsembleMember(value, run_experiment(model_factory(value), initial_state, config)) for value in values)


def run_named_ensemble(model_factory: Callable[[Mapping[str, float]], ODEModel], initial_state: tuple[float, ...], parameter_sets: Iterable[Mapping[str, float]], config: ExperimentConfig) -> tuple[NamedEnsembleMember, ...]:
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
        members.append(NamedEnsembleMember(normalized, run_experiment(model_factory(parameters), initial_state, config)))
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
        means = tuple(sum(solution.states[sample_index][component] for solution in solutions) / count for component in range(reference.state_dimension))
        stds = tuple(math.sqrt(sum((solution.states[sample_index][component] - means[component]) ** 2 for solution in solutions) / count) for component in range(reference.state_dimension))
        mean_states.append(means)
        std_states.append(stds)
    return EnsembleStatistics(reference.times, tuple(mean_states), tuple(std_states))


def monte_carlo_convergence(members: Iterable[EnsembleMember | NamedEnsembleMember], observable: Callable[[ODESolution], float], checkpoints: Iterable[int] | None = None) -> MonteCarloConvergence:
    items = tuple(members)
    if not items:
        raise ValueError("members must not be empty")
    if checkpoints is None:
        sizes = tuple(range(1, len(items) + 1))
    else:
        sizes = tuple(int(value) for value in checkpoints)
        if not sizes:
            raise ValueError("checkpoints must not be empty")
        if any(value <= 0 or value > len(items) for value in sizes):
            raise ValueError("checkpoints must lie between 1 and the ensemble size")
        if any(right <= left for left, right in zip(sizes, sizes[1:])):
            raise ValueError("checkpoints must be strictly increasing")
    values = []
    for member in items:
        value = float(observable(member.solution))
        if not math.isfinite(value):
            raise ValueError("observable must return finite values")
        values.append(value)
    estimates = []
    running_sum = running_square_sum = 0.0
    checkpoint_index = 0
    for index, value in enumerate(values, start=1):
        running_sum += value
        running_square_sum += value * value
        if index != sizes[checkpoint_index]:
            continue
        mean = running_sum / index
        variance = max(0.0, running_square_sum / index - mean * mean)
        std = math.sqrt(variance)
        estimates.append(MonteCarloEstimate(index, mean, std, std / math.sqrt(index)))
        checkpoint_index += 1
        if checkpoint_index == len(sizes):
            break
    return MonteCarloConvergence(tuple(estimates))


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


def ensemble_quantiles(members: Iterable[EnsembleMember | NamedEnsembleMember], probabilities: Iterable[float] = (0.05, 0.5, 0.95)) -> EnsembleQuantiles:
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
            samples.append(tuple(_quantile([solution.states[sample_index][component] for solution in solutions], probability) for component in range(reference.state_dimension)))
        quantile_states.append(tuple(samples))
    return EnsembleQuantiles(reference.times, probs, tuple(quantile_states))
