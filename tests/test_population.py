import pytest

from labsim import LotkaVolterra


def test_lotka_volterra_derivative_matches_equations():
    model = LotkaVolterra()
    assert model(0.0, (10.0, 5.0)) == pytest.approx((5.0, -3.75))


def test_lotka_volterra_parameter_validation():
    with pytest.raises(ValueError):
        LotkaVolterra(prey_growth=0.0)
    with pytest.raises(ValueError):
        LotkaVolterra(predation=0.0)
    with pytest.raises(ValueError):
        LotkaVolterra(predator_decay=0.0)
    with pytest.raises(ValueError):
        LotkaVolterra(predator_growth=0.0)


def test_lotka_volterra_has_two_population_states():
    assert LotkaVolterra().state_size == 2
