"""Reusable model interfaces and small physical systems for LabSim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

State = Sequence[float]


class ODEModel:
    """Base interface for a model expressible as ``y' = f(t, y)``."""

    @property
    def state_size(self) -> int:
        """Number of state variables required by the model."""
        raise NotImplementedError

    def derivative(self, time: float, state: State) -> tuple[float, ...]:
        """Evaluate the model's state derivative at a point in time."""
        raise NotImplementedError

    def __call__(self, time: float, state: State) -> tuple[float, ...]:
        return self.derivative(time, state)


@dataclass(frozen=True)
class HarmonicOscillator(ODEModel):
    """Undamped one-dimensional oscillator with displacement and velocity state."""

    angular_frequency: float = 1.0

    def __post_init__(self) -> None:
        if self.angular_frequency <= 0:
            raise ValueError("angular_frequency must be positive")

    @property
    def state_size(self) -> int:
        return 2

    def derivative(self, time: float, state: State) -> tuple[float, ...]:
        del time
        if len(state) != self.state_size:
            raise ValueError("harmonic oscillator requires a two-value state")
        position, velocity = state
        return float(velocity), -self.angular_frequency**2 * float(position)
