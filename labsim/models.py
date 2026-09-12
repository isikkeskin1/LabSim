"""Reusable model interfaces and small physical systems for LabSim."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

State = Sequence[float]


class ODEModel:
    """Base interface for a model expressible as ``y' = f(t, y)``."""

    @property
    def state_size(self) -> int:
        raise NotImplementedError

    def derivative(self, time: float, state: State) -> tuple[float, ...]:
        raise NotImplementedError

    def __call__(self, time: float, state: State) -> tuple[float, ...]:
        return self.derivative(time, state)


@dataclass(frozen=True)
class HarmonicOscillator(ODEModel):
    """Undamped one-dimensional oscillator."""
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


@dataclass(frozen=True)
class DampedOscillator(ODEModel):
    """One-dimensional mass-spring-damper system."""
    mass: float = 1.0
    stiffness: float = 1.0
    damping: float = 0.1

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        if self.stiffness <= 0:
            raise ValueError("stiffness must be positive")
        if self.damping < 0:
            raise ValueError("damping must be non-negative")

    @property
    def state_size(self) -> int:
        return 2

    def derivative(self, time: float, state: State) -> tuple[float, ...]:
        del time
        if len(state) != self.state_size:
            raise ValueError("damped oscillator requires a two-value state")
        position, velocity = (float(value) for value in state)
        acceleration = (-self.stiffness * position - self.damping * velocity) / self.mass
        return velocity, acceleration


@dataclass(frozen=True)
class LotkaVolterra(ODEModel):
    """Classical predator-prey population dynamics model."""
    prey_growth: float = 1.0
    predation: float = 0.1
    predator_decay: float = 1.5
    predator_growth: float = 0.075

    def __post_init__(self) -> None:
        if self.prey_growth <= 0 or self.predation <= 0 or self.predator_decay <= 0 or self.predator_growth <= 0:
            raise ValueError("Lotka-Volterra parameters must be positive")

    @property
    def state_size(self) -> int:
        return 2

    def derivative(self, time: float, state: State) -> tuple[float, ...]:
        del time
        if len(state) != self.state_size:
            raise ValueError("Lotka-Volterra requires prey and predator state")
        prey, predator = (float(value) for value in state)
        return (
            self.prey_growth * prey - self.predation * prey * predator,
            self.predator_growth * prey * predator - self.predator_decay * predator,
        )


@dataclass(frozen=True)
class LogisticGrowth(ODEModel):
    """Single-population logistic growth model with a carrying capacity."""
    growth_rate: float = 1.0
    carrying_capacity: float = 100.0

    def __post_init__(self) -> None:
        if self.growth_rate <= 0:
            raise ValueError("growth_rate must be positive")
        if self.carrying_capacity <= 0:
            raise ValueError("carrying_capacity must be positive")

    @property
    def state_size(self) -> int:
        return 1

    def derivative(self, time: float, state: State) -> tuple[float, ...]:
        del time
        if len(state) != self.state_size:
            raise ValueError("logistic growth requires one population value")
        population = float(state[0])
        return (self.growth_rate * population * (1 - population / self.carrying_capacity),)


@dataclass(frozen=True)
class SimplePendulum(ODEModel):
    """Nonlinear simple pendulum using angle and angular velocity state."""
    gravity: float = 9.81
    length: float = 1.0

    def __post_init__(self) -> None:
        if self.gravity <= 0:
            raise ValueError("gravity must be positive")
        if self.length <= 0:
            raise ValueError("length must be positive")

    @property
    def state_size(self) -> int:
        return 2

    def derivative(self, time: float, state: State) -> tuple[float, ...]:
        del time
        if len(state) != self.state_size:
            raise ValueError("simple pendulum requires angle and angular velocity")
        angle, angular_velocity = (float(value) for value in state)
        return angular_velocity, -(self.gravity / self.length) * math.sin(angle)


@dataclass(frozen=True)
class SIRModel(ODEModel):
    """Normalized susceptible-infected-recovered epidemic model."""
    transmission_rate: float = 0.5
    recovery_rate: float = 0.1

    def __post_init__(self) -> None:
        if self.transmission_rate <= 0:
            raise ValueError("transmission_rate must be positive")
        if self.recovery_rate <= 0:
            raise ValueError("recovery_rate must be positive")

    @property
    def state_size(self) -> int:
        return 3

    def derivative(self, time: float, state: State) -> tuple[float, ...]:
        del time
        if len(state) != self.state_size:
            raise ValueError("SIR model requires susceptible, infected, and recovered state")
        susceptible, infected, _recovered = (float(value) for value in state)
        new_infections = self.transmission_rate * susceptible * infected
        recoveries = self.recovery_rate * infected
        return (-new_infections, new_infections - recoveries, recoveries)
