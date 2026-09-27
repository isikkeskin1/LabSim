import pytest

from labsim.copula import GaussianCopulaDistribution, sample_copula_parameter_sets
from labsim.joint import sample_correlation
from labsim.uncertainty import NormalDistribution, UniformDistribution


def test_copula_sampling_is_seeded_and_respects_uniform_bounds():
    distribution = GaussianCopulaDistribution(
        marginals={
            "mass": UniformDistribution(1.0, 3.0),
            "stiffness": NormalDistribution(8.0, 1.0),
        },
        correlation=((1.0, 0.6), (0.6, 1.0)),
    )
    first = sample_copula_parameter_sets(distribution, 32, seed=17)
    assert first == sample_copula_parameter_sets(distribution, 32, seed=17)
    assert first != sample_copula_parameter_sets(distribution, 32, seed=18)
    assert all(1.0 <= sample["mass"] <= 3.0 for sample in first)


def test_copula_preserves_latent_dependence_across_non_normal_marginals():
    distribution = GaussianCopulaDistribution(
        marginals={
            "x": UniformDistribution(-1.0, 1.0),
            "y": UniformDistribution(10.0, 20.0),
        },
        correlation=((1.0, 0.75), (0.75, 1.0)),
    )
    samples = sample_copula_parameter_sets(distribution, 6000, seed=9)
    # Pearson correlation changes slightly under nonlinear marginal transforms,
    # but should retain strong positive dependence.
    assert sample_correlation(samples, "x", "y") > 0.68


def test_copula_rejects_invalid_correlation():
    marginals = {"x": UniformDistribution(0.0, 1.0), "y": UniformDistribution(0.0, 1.0)}
    with pytest.raises(ValueError, match="symmetric"):
        GaussianCopulaDistribution(marginals, ((1.0, 0.5), (0.2, 1.0)))
    with pytest.raises(ValueError, match="positive definite"):
        GaussianCopulaDistribution(marginals, ((1.0, 1.0), (1.0, 1.0)))


def test_copula_validates_count_and_marginal_protocol():
    distribution = GaussianCopulaDistribution(
        {"x": UniformDistribution(0.0, 1.0)},
        ((1.0,),),
    )
    with pytest.raises(ValueError, match="count"):
        sample_copula_parameter_sets(distribution, 0)
    with pytest.raises(TypeError, match="quantile"):
        GaussianCopulaDistribution({"x": object()}, ((1.0,),))
