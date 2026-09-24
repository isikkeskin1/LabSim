import pytest

from labsim.sensitivity_variance import variance_sensitivity
from labsim.uncertainty import UniformDistribution


def test_additive_model_attributes_more_variance_to_larger_coefficient():
    result = variance_sensitivity(
        {"x": UniformDistribution(0.0, 1.0), "y": UniformDistribution(0.0, 1.0)},
        lambda p: p["x"] + 2.0 * p["y"],
        samples=12000,
        seed=42,
    )

    x = result.for_parameter("x")
    y = result.for_parameter("y")
    assert x.first_order == pytest.approx(0.2, abs=0.04)
    assert y.first_order == pytest.approx(0.8, abs=0.04)
    assert x.total_effect == pytest.approx(0.2, abs=0.04)
    assert y.total_effect == pytest.approx(0.8, abs=0.04)


def test_interaction_increases_total_effect_over_first_order():
    result = variance_sensitivity(
        {"x": UniformDistribution(0.0, 1.0), "y": UniformDistribution(0.0, 1.0)},
        lambda p: p["x"] * p["y"],
        samples=16000,
        seed=7,
    )

    for name in ("x", "y"):
        index = result.for_parameter(name)
        assert index.total_effect > index.first_order
        assert index.first_order == pytest.approx(3.0 / 7.0, abs=0.05)
        assert index.total_effect == pytest.approx(4.0 / 7.0, abs=0.05)


def test_sampling_is_reproducible():
    distributions = {"x": UniformDistribution(-1.0, 1.0)}
    first = variance_sensitivity(distributions, lambda p: p["x"] ** 2, samples=500, seed=11)
    second = variance_sensitivity(distributions, lambda p: p["x"] ** 2, samples=500, seed=11)
    assert first == second


def test_rejects_invalid_sample_count_and_constant_response():
    distribution = {"x": UniformDistribution(0.0, 1.0)}
    with pytest.raises(ValueError, match="at least 2"):
        variance_sensitivity(distribution, lambda p: p["x"], samples=1)
    with pytest.raises(ValueError, match="positive finite variance"):
        variance_sensitivity(distribution, lambda _p: 1.0, samples=20, seed=1)
