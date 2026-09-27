"""Copula-based joint distributions with arbitrary scalar marginals."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Protocol

from .joint import _cholesky


class QuantileDistribution(Protocol):
    """Scalar distribution that can transform a unit-interval quantile."""

    def quantile(self, probability: float) -> float:
        ...


@dataclass(frozen=True)
class GaussianCopulaDistribution:
    """Join arbitrary scalar marginals with Gaussian dependence.

    The supplied matrix describes dependence in latent standard-normal space.
    Marginals only need to implement ``quantile(probability)``.
    """

    marginals: Mapping[str, QuantileDistribution]
    correlation: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        names = tuple(self.marginals)
        if not names or any(not name for name in names):
            raise ValueError("marginals must have non-empty parameter names")
        dimension = len(names)
        if len(self.correlation) != dimension or any(len(row) != dimension for row in self.correlation):
            raise ValueError("correlation matrix must be square and match the number of marginals")
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
        for marginal in self.marginals.values():
            if not callable(getattr(marginal, "quantile", None)):
                raise TypeError("each marginal must provide quantile(probability)")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.marginals)

    @property
    def dimension(self) -> int:
        return len(self.marginals)

    def sample(self, rng: random.Random) -> dict[str, float]:
        factor = _cholesky(self.correlation)
        independent = tuple(rng.gauss(0.0, 1.0) for _ in self.names)
        latent = tuple(
            sum(factor[row][column] * independent[column] for column in range(row + 1))
            for row in range(self.dimension)
        )
        normal = NormalDist()
        return {
            name: float(marginal.quantile(normal.cdf(value)))
            for name, marginal, value in zip(self.names, self.marginals.values(), latent)
        }


def sample_copula_parameter_sets(
    distribution: GaussianCopulaDistribution,
    count: int,
    *,
    seed: int | None = None,
) -> tuple[dict[str, float], ...]:
    """Draw reproducible named parameter sets from a Gaussian copula."""
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    return tuple(distribution.sample(rng) for _ in range(count))
