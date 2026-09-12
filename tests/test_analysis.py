import math

import pytest

from labsim import HarmonicOscillator, integrate_ode, max_state_error


def exact_oscillator(time: float, state: tuple[float, ...]) -> tuple[float, ...]:
    del state
    return (math.cos(time), -math.sin(time))


def test_max_state_error_matches_exact_samples():
    solution = integrate_ode(
        HarmonicOscillator(),
        (1.0, 0.0),
        dt=0.1,
        steps=10,
    )

    error = max_state_error(solution, exact_oscillator)

    assert error > 0.0
    assert error < 1e-4


def test_max_state_error_rejects_dimension_mismatch():
    solution = integrate_ode(lambda time, state: (0.0,), (1.0,), steps=1)

    with pytest.raises(ValueError, match="wrong dimension"):
        max_state_error(solution, lambda time, state: (1.0, 2.0))
