import math

import pytest

from labsim import HarmonicOscillator


def test_harmonic_oscillator_derivative():
    model = HarmonicOscillator(angular_frequency=2.0)

    assert model.state_size == 2
    assert model(0.0, (3.0, 4.0)) == (4.0, -12.0)


def test_harmonic_oscillator_has_expected_period():
    model = HarmonicOscillator()
    period = 2 * math.pi / model.angular_frequency

    assert period == pytest.approx(2 * math.pi)


def test_harmonic_oscillator_rejects_invalid_frequency():
    with pytest.raises(ValueError, match="positive"):
        HarmonicOscillator(angular_frequency=0.0)


def test_harmonic_oscillator_rejects_wrong_state_size():
    model = HarmonicOscillator()

    with pytest.raises(ValueError, match="two-value"):
        model.derivative(0.0, (1.0,))
