"""Physical diagnostics for simulated trajectories."""

from __future__ import annotations

from collections.abc import Sequence


def harmonic_energy(
    position: float,
    velocity: float,
    *,
    mass: float = 1.0,
    angular_frequency: float = 1.0,
) -> float:
    """Return total mechanical energy of an undamped harmonic oscillator."""
    if mass <= 0:
        raise ValueError("mass must be positive")
    if angular_frequency <= 0:
        raise ValueError("angular_frequency must be positive")
    kinetic = 0.5 * mass * velocity**2
    potential = 0.5 * mass * angular_frequency**2 * position**2
    return kinetic + potential


def relative_drift(values: Sequence[float]) -> float:
    """Return the largest absolute deviation from the initial value, relatively."""
    if not values:
        raise ValueError("values must not be empty")
    reference = abs(float(values[0]))
    if reference == 0:
        return max(abs(float(value)) for value in values)
    return max(abs(float(value) - float(values[0])) for value in values) / reference
