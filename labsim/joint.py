"""Joint parameter distributions for correlated uncertainty designs."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CorrelatedNormalDistribution:
    """Multivariate normal distribution parameterized by marginal stds and correlation.

    The correlation matrix is validated and factorized with a small dependency-free
    Cholesky implementation. Sampling uses a caller-provided RNG so reproducibility
    remains under the experiment's control.
    """

    names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        dimension = len(self.names)
        if dimension == 0:
            raise ValueError("names must not be empty")
        if len(set(self.names)) != dimension or any(not name for name in self.names):
            raise ValueError("parameter names must be non-empty and unique")
        if len(self.means) != dimension or len(self.stds) != dimension:
            raise ValueError("means and stds must match the number of names")
        if any(not math.isfinite(value) for value in self.means):
            raise ValueError("means must be finite")
        if any(not math.isfinite(value) or value <= 0 for value in self.stds):
            raise ValueError("stds must be positive and finite")
        if len(self.correlation) != dimension or any(len(row) != dimension for row in self.correlation):
            raise ValueError("correlation matrix must be square and match the parameter dimension")
        for row in self.correlation:
            if any(not math.isfinite(value) or abs(value) > 1.0 for value in row):
                raise ValueError("correlations must be finite and lie between -1 and 1")
        for i in range(dimension):
            if not math.isclose(self.correlation[i][i], 1.0, abs_tol=1e-12):
                raise ValueError("correlation matrix diagonal must equal 1")
            for j in range(i):
                if not math.isclose(self.correlation[i][j], self.correlation[j][i], abs_tol=1e-12):
                    raise ValueError("correlation matrix must be symmetric")
        _cholesky(self.correlation)

    @property
    def dimension(self) -> int:
        return len(self.names)

    def sample(self, rng: random.Random) -> dict[str, float]:
        factor = _cholesky(self.correlation)
        independent = tuple(rng.gauss(0.0, 1.0) for _ in self.names)
        correlated = tuple(
            sum(factor[row][column] * independent[column] for column in range(row + 1))
            for row in range(self.dimension)
        )
        return {
            name: mean + std * value
            for name, mean, std, value in zip(self.names, self.means, self.stds, correlated)
        }


def _cholesky(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Return the lower Cholesky factor of a positive-definite matrix."""
    size = len(matrix)
    factor = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            remainder = sum(factor[row][k] * factor[column][k] for k in range(column))
            if row == column:
                diagonal = float(matrix[row][row]) - remainder
                if diagonal <= 1e-14:
                    raise ValueError("correlation matrix must be positive definite")
                factor[row][column] = math.sqrt(diagonal)
            else:
                factor[row][column] = (float(matrix[row][column]) - remainder) / factor[column][column]
    return tuple(tuple(row) for row in factor)


def sample_joint_parameter_sets(
    distribution: CorrelatedNormalDistribution,
    count: int,
    *,
    seed: int | None = None,
) -> tuple[dict[str, float], ...]:
    """Draw reproducible named parameter sets from a joint distribution."""
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    return tuple(distribution.sample(rng) for _ in range(count))


def sample_correlation(samples: Sequence[Mapping[str, float]], left: str, right: str) -> float:
    """Return the sample Pearson correlation between two named sampled parameters."""
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    try:
        x = tuple(float(sample[left]) for sample in samples)
        y = tuple(float(sample[right]) for sample in samples)
    except KeyError as exc:
        raise ValueError(f"missing parameter {exc.args[0]!r}") from exc
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    centered_x = tuple(value - mean_x for value in x)
    centered_y = tuple(value - mean_y for value in y)
    sum_xx = sum(value * value for value in centered_x)
    sum_yy = sum(value * value for value in centered_y)
    if sum_xx == 0.0 or sum_yy == 0.0:
        raise ValueError("sample correlation is undefined for a constant parameter")
    return sum(a * b for a, b in zip(centered_x, centered_y)) / math.sqrt(sum_xx * sum_yy)
