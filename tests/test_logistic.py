import pytest

from labsim import LogisticGrowth


def test_logistic_growth_derivative():
    model = LogisticGrowth(growth_rate=0.5, carrying_capacity=100.0)
    assert model(0.0, (20.0,)) == pytest.approx((8.0,))


def test_logistic_growth_equilibrium_has_zero_derivative():
    model = LogisticGrowth(carrying_capacity=100.0)
    assert model(0.0, (100.0,)) == pytest.approx((0.0,))


def test_logistic_growth_validates_parameters():
    with pytest.raises(ValueError):
        LogisticGrowth(growth_rate=0.0)
    with pytest.raises(ValueError):
        LogisticGrowth(carrying_capacity=0.0)
