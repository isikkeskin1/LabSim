import pytest

from labsim.physics import harmonic_energy, relative_drift


def test_harmonic_energy_adds_kinetic_and_potential_terms():
    assert harmonic_energy(1.0, 2.0) == pytest.approx(2.5)


def test_harmonic_energy_scales_with_mass():
    assert harmonic_energy(1.0, 2.0, mass=2.0) == pytest.approx(5.0)


def test_relative_drift_is_zero_for_constant_signal():
    assert relative_drift((4.0, 4.0, 4.0)) == 0.0


def test_relative_drift_handles_zero_reference():
    assert relative_drift((0.0, 0.25, -0.5)) == pytest.approx(0.5)


def test_physical_parameters_must_be_positive():
    with pytest.raises(ValueError):
        harmonic_energy(0.0, 0.0, mass=0.0)
