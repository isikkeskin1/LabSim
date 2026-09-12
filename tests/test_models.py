import pytest

from labsim import DampedOscillator, HarmonicOscillator


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
