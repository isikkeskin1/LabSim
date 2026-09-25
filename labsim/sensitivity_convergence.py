"""Convergence diagnostics for variance-based sensitivity estimates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .sensitivity_variance import VarianceSensitivity, variance_sensitivity
from .uncertainty import Distribution


@dataclass(frozen=True)
class SensitivityConvergencePoint:
    """Sensitivity estimate produced at one Monte Carlo sample budget."""

    samples: int
    result: VarianceSensitivity


@dataclass(frozen=True)
class SensitivityConvergence:
    """Ordered sensitivity estimates across increasing sample budgets."""

    points: tuple[SensitivityConvergencePoint, ...]

    def for_samples(self, samples: int) -> VarianceSensitivity:
        for point in self.points:
            if point.samples == samples:
                return point.result
        raise KeyError(samples)


def sensitivity_convergence(
    distributions: Mapping[str, Distribution],
    observable: Callable[[Mapping[str, float]], float],
    sample_counts: Sequence[int],
    *,
    seed: int | None = None,
) -> SensitivityConvergence:
    """Re-estimate variance sensitivity across increasing sample budgets."""
    counts = tuple(int(count) for count in sample_counts)
    if not counts:
        raise ValueError("sample_counts must not be empty")
    if any(count < 2 for count in counts):
        raise ValueError("sample counts must be at least 2")
    if any(later <= earlier for earlier, later in zip(counts, counts[1:])):
        raise ValueError("sample_counts must be strictly increasing")

    points = tuple(
        SensitivityConvergencePoint(
            count,
            variance_sensitivity(distributions, observable, samples=count, seed=seed),
        )
        for count in counts
    )
    return SensitivityConvergence(points)
