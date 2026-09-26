import pytest

from labsim.sensitivity_uncertainty import sensitivity_uncertainty
from labsim.uncertainty import UniformDistribution


def _distributions():
    return {
        "x": UniformDistribution(0.0, 1.0),
        "y": UniformDistribution(0.0, 1.0),
    }


def test_replicated_intervals_track_known_additive_sensitivity():
    result = sensitivity_uncertainty(
        _distributions(),
        lambda p: p["x"] + 2.0 * p["y"],
        samples=1200,
        replications=12,
        confidence=0.9,
        seed=17,
    )

    x = result.for_parameter("x")
    y = result.for_parameter("y")
    assert x.first_order_mean == pytest.approx(0.2, abs=0.06)
    assert y.first_order_mean == pytest.approx(0.8, abs=0.06)
    assert x.first_order_lower <= x.first_order_mean <= x.first_order_upper
    assert y.total_effect_lower <= y.total_effect_mean <= y.total_effect_upper


def test_replicated_intervals_are_seed_reproducible():
    kwargs = dict(samples=200, replications=4, confidence=0.8, seed=3)
    first = sensitivity_uncertainty(_distributions(), lambda p: p["x"] + p["y"], **kwargs)
    second = sensitivity_uncertainty(_distributions(), lambda p: p["x"] + p["y"], **kwargs)
    assert first == second


def test_replicated_intervals_validate_configuration():
    with pytest.raises(ValueError, match="replications"):
        sensitivity_uncertainty(_distributions(), lambda p: p["x"], samples=10, replications=1)
    with pytest.raises(ValueError, match="confidence"):
        sensitivity_uncertainty(_distributions(), lambda p: p["x"], samples=10, confidence=1.0)
    with pytest.raises(ValueError, match="samples"):
        sensitivity_uncertainty(_distributions(), lambda p: p["x"], samples=1)


def test_parameter_lookup_rejects_unknown_name():
    result = sensitivity_uncertainty(
        _distributions(), lambda p: p["x"] + p["y"], samples=20, replications=2, seed=5
    )
    with pytest.raises(KeyError):
        result.for_parameter("missing")
