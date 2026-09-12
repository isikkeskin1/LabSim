"""Finite-difference tools for local parameter sensitivity studies."""

from __future__ import annotations

from collections.abc import Callable


def central_difference(
    function: Callable[[float], float],
    parameter: float,
    step: float,
) -> float:
    """Approximate the derivative with a symmetric finite difference."""
    if step <= 0:
        raise ValueError("step must be positive")
    forward = float(function(parameter + step))
    backward = float(function(parameter - step))
    return (forward - backward) / (2 * step)
