import pytest

from labsim import UniformDistribution
from labsim.sensitivity_convergence import sensitivity_convergence


def test_sensitivity_convergence_records_requested_budgets():
    distributions = {"x": UniformDistribution(0.0, 1.0), "y": UniformDistribution(0.0, 1.0)}
    study = sensitivity_convergence(distributions, lambda p: p["x"] + 2.0 * p["y"], (200, 800, 2000), seed=17)
    assert [point.samples for point in study.points] == [200, 800, 2000]
    final = study.for_samples(2000)
    assert final.for_parameter("x").first_order == pytest.approx(0.2, abs=0.08)
    assert final.for_parameter("y").first_order == pytest.approx(0.8, abs=0.08)
    assert final.for_parameter("x").total_effect == pytest.approx(0.2, abs=0.08)
    assert final.for_parameter("y").total_effect == pytest.approx(0.8, abs=0.08)


def test_sensitivity_convergence_is_seed_reproducible():
    distributions = {"x": UniformDistribution(-1.0, 1.0)}
    first = sensitivity_convergence(distributions, lambda p: p["x"] ** 2, (50, 100), seed=9)
    second = sensitivity_convergence(distributions, lambda p: p["x"] ** 2, (50, 100), seed=9)
    assert first == second


def test_sensitivity_convergence_validates_checkpoints():
    distributions = {"x": UniformDistribution(0.0, 1.0)}
    observable = lambda p: p["x"]
    with pytest.raises(ValueError, match="must not be empty"):
        sensitivity_convergence(distributions, observable, (), seed=1)
    with pytest.raises(ValueError, match="at least 2"):
        sensitivity_convergence(distributions, observable, (1, 10), seed=1)
    with pytest.raises(ValueError, match="strictly increasing"):
        sensitivity_convergence(distributions, observable, (20, 20), seed=1)
