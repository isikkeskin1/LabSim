"""Variance-based sensitivity analysis for independent parameter designs."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .uncertainty import Distribution, sample_parameter_sets


@dataclass(frozen=True)
class SensitivityIndex:
    """First-order and total-effect indices for one model parameter."""

    parameter: str
    first_order: float
    total_effect: float


@dataclass(frozen=True)
class VarianceSensitivity:
    """Saltelli-style variance attribution for a scalar model response."""

    variance: float
    indices: tuple[SensitivityIndex, ...]

    def for_parameter(self, name: str) -> SensitivityIndex:
        for index in self.indices:
            if index.parameter == name:
                return index
        raise KeyError(name)


def variance_sensitivity(
    distributions: Mapping[str, Distribution],
    observable: Callable[[Mapping[str, float]], float],
    *,
    samples: int,
    seed: int | None = None,
) -> VarianceSensitivity:
    """Estimate first-order and total-effect sensitivity indices.

    Two independent Monte Carlo matrices A and B are sampled. For each parameter
    i, A_Bi is A with column i replaced by B. The returned first-order estimator
    follows Saltelli's covariance form and the total-effect estimator follows
    Jansen's squared-difference form. Inputs are assumed independent; correlated
    designs require a different attribution model.
    """
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if not distributions:
        raise ValueError("distributions must not be empty")

    # Derive independent deterministic streams without touching global RNG state.
    import random

    rng = random.Random(seed)
    a = sample_parameter_sets(distributions, samples, seed=rng.randrange(2**63))
    b = sample_parameter_sets(distributions, samples, seed=rng.randrange(2**63))

    def evaluate(points: tuple[dict[str, float], ...]) -> tuple[float, ...]:
        values = tuple(float(observable(point)) for point in points)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("observable must return finite values")
        return values

    ya = evaluate(a)
    yb = evaluate(b)
    pooled = ya + yb
    mean = sum(pooled) / len(pooled)
    variance = sum((value - mean) ** 2 for value in pooled) / (len(pooled) - 1)
    if variance <= 0.0 or not math.isfinite(variance):
        raise ValueError("observable must have positive finite variance")

    indices: list[SensitivityIndex] = []
    for name in distributions:
        hybrids = tuple({**row_a, name: row_b[name]} for row_a, row_b in zip(a, b))
        yab = evaluate(hybrids)
        first = sum(yb_j * (yab_j - ya_j) for ya_j, yb_j, yab_j in zip(ya, yb, yab)) / samples / variance
        total = sum((ya_j - yab_j) ** 2 for ya_j, yab_j in zip(ya, yab)) / (2.0 * samples * variance)
        indices.append(SensitivityIndex(name, first, total))

    return VarianceSensitivity(variance, tuple(indices))
