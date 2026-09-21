import math

import pytest

from labsim import DampedOscillator, HarmonicOscillator, SIRModel, SimplePendulum


def test_harmonic_oscillator_derivative():
    model = HarmonicOscillator(angular_frequency=2.0)
    assert model.state_size == 2
    assert model(0.0, (3.0, 4.0)) == (4.0, -12.0)


def test_harmonic_oscillator_rejects_invalid_frequency():
    with pytest.raises(ValueError, match="positive"):
        HarmonicOscillator(angular_frequency=0.0)


def test_damped_oscillator_derivative_includes_damping():
    model = DampedOscillator(mass=2.0, stiffness=8.0, damping=2.0)
    assert model(0.0, (1.0, 3.0)) == (3.0, -7.0)


def test_damped_oscillator_accepts_zero_damping():
    assert DampedOscillator(damping=0.0).damping == 0.0


def test_damped_oscillator_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        DampedOscillator(mass=0.0)
    with pytest.raises(ValueError):
        DampedOscillator(stiffness=0.0)
    with pytest.raises(ValueError):
        DampedOscillator(damping=-1.0)


def test_simple_pendulum_uses_nonlinear_gravity_term():
    model = SimplePendulum(gravity=9.81, length=2.0)
    assert model(0.0, (math.pi / 2, 0.5)) == pytest.approx((0.5, -4.905))


def test_sir_model_conserves_normalized_population():
    model = SIRModel()
    derivative = model(0.0, (0.97, 0.03, 0.0))
    assert sum(derivative) == pytest.approx(0.0)


def test_sir_model_rejects_wrong_dimension():
    with pytest.raises(ValueError, match="susceptible, infected, and recovered"):
        SIRModel()(0.0, (0.9, 0.1))
