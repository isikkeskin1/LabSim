"""Replicated uncertainty intervals for variance-based sensitivity estimates."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .sensitivity_variance import variance_sensitivity
from .uncertainty import Distribution


@dataclass(frozen=True)
class SensitivityInterval:
    """Replicated estimate and empirical interval for one sensitivity index."""

    parameter: str
    first_order_mean: float
    first_order_lower: float
    first_order_upper: float
    total_effect_mean: float
    total_effect_lower: float
    total_effect_upper: float


@dataclass(frozen=True)
class SensitivityUncertainty:
    """Empirical uncertainty summary across independent sensitivity replicates."""

    confidence: float
    replications: int
    samples_per_replication: int
    intervals: tuple[SensitivityInterval, ...]

    def for_parameter(self, name: str) -> SensitivityInterval:
        for interval in self.intervals:
            if interval.parameter == name:
                return interval
        raise KeyError(name)


def _quantile(values: tuple[float, ...], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def sensitivity_uncertainty(
    distributions: Mapping[str, Distribution],
    observable: Callable[[Mapping[str, float]], float],
    *,
    samples: int,
    replications: int = 20,
    confidence: float = 0.95,
    seed: int | None = None,
) -> SensitivityUncertainty:
    """Estimate empirical intervals from independent sensitivity replications.

    Each replication runs the existing Saltelli/Jansen estimator with an
    independent child seed. The reported interval is the central empirical
    ``confidence`` interval across replicated estimates. It measures Monte
    Carlo estimator uncertainty; it is not a confidence interval for model or
    measurement error.
    """
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if replications < 2:
        raise ValueError("replications must be at least 2")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    rng = random.Random(seed)
    results = tuple(
        variance_sensitivity(
            distributions,
            observable,
            samples=samples,
            seed=rng.randrange(2**63),
        )
        for _ in range(replications)
    )

    alpha = (1.0 - confidence) / 2.0
    intervals: list[SensitivityInterval] = []
    for name in distributions:
        first = tuple(result.for_parameter(name).first_order for result in results)
        total = tuple(result.for_parameter(name).total_effect for result in results)
        intervals.append(
            SensitivityInterval(
                parameter=name,
                first_order_mean=sum(first) / replications,
                first_order_lower=_quantile(first, alpha),
                first_order_upper=_quantile(first, 1.0 - alpha),
                total_effect_mean=sum(total) / replications,
                total_effect_lower=_quantile(total, alpha),
                total_effect_upper=_quantile(total, 1.0 - alpha),
            )
        )

    return SensitivityUncertainty(confidence, replications, samples, tuple(intervals))
