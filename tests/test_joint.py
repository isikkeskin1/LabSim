import pytest

from labsim.joint import (
    CorrelatedNormalDistribution,
    sample_correlation,
    sample_joint_parameter_sets,
)


def test_joint_sampling_is_seeded_and_preserves_names():
    distribution = CorrelatedNormalDistribution(
        names=("mass", "stiffness"),
        means=(2.0, 8.0),
        stds=(0.2, 1.0),
        correlation=((1.0, 0.7), (0.7, 1.0)),
    )
    first = sample_joint_parameter_sets(distribution, 8, seed=31)
    assert first == sample_joint_parameter_sets(distribution, 8, seed=31)
    assert first != sample_joint_parameter_sets(distribution, 8, seed=32)
    assert tuple(first[0]) == ("mass", "stiffness")


def test_sampled_correlation_tracks_requested_positive_dependence():
    distribution = CorrelatedNormalDistribution(
        names=("x", "y"),
        means=(0.0, 0.0),
        stds=(1.0, 2.0),
        correlation=((1.0, 0.8), (0.8, 1.0)),
    )
    samples = sample_joint_parameter_sets(distribution, 5000, seed=7)
    assert sample_correlation(samples, "x", "y") == pytest.approx(0.8, abs=0.03)


def test_joint_distribution_rejects_invalid_correlation_matrices():
    kwargs = dict(names=("x", "y"), means=(0.0, 0.0), stds=(1.0, 1.0))
    with pytest.raises(ValueError, match="symmetric"):
        CorrelatedNormalDistribution(correlation=((1.0, 0.5), (0.2, 1.0)), **kwargs)
    with pytest.raises(ValueError, match="positive definite"):
        CorrelatedNormalDistribution(correlation=((1.0, 1.0), (1.0, 1.0)), **kwargs)
    with pytest.raises(ValueError, match="diagonal"):
        CorrelatedNormalDistribution(correlation=((0.9, 0.0), (0.0, 1.0)), **kwargs)


def test_joint_distribution_validates_parameter_metadata():
    with pytest.raises(ValueError, match="unique"):
        CorrelatedNormalDistribution(("x", "x"), (0.0, 0.0), (1.0, 1.0), ((1.0, 0.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match="stds"):
        CorrelatedNormalDistribution(("x",), (0.0,), (0.0,), ((1.0,),))
    with pytest.raises(ValueError, match="count"):
        distribution = CorrelatedNormalDistribution(("x",), (0.0,), (1.0,), ((1.0,),))
        sample_joint_parameter_sets(distribution, 0)


def test_sample_correlation_validates_inputs():
    with pytest.raises(ValueError, match="at least two"):
        sample_correlation(({"x": 1.0, "y": 2.0},), "x", "y")
    with pytest.raises(ValueError, match="missing parameter"):
        sample_correlation(({"x": 1.0}, {"x": 2.0}), "x", "y")
