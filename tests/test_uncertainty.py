import pytest

from labsim import ExperimentConfig, LogisticGrowth
from labsim.solvers import ODESolution
from labsim.uncertainty import (
    EnsembleMember,
    NormalDistribution,
    UniformDistribution,
    ensemble_quantiles,
    ensemble_statistics,
    latin_hypercube_parameter_sets,
    monte_carlo_convergence,
    run_ensemble,
    run_named_ensemble,
    sample_parameter_sets,
    sample_parameters,
)


def test_run_ensemble_preserves_parameter_samples():
    members = run_ensemble(lambda rate: LogisticGrowth(growth_rate=rate, carrying_capacity=100.0), (10.0,), (0.5, 1.0, 1.5), ExperimentConfig(duration=1.0, dt=0.1))
    assert [member.parameter for member in members] == [0.5, 1.0, 1.5]
    assert members[0].solution.final_state[0] < members[-1].solution.final_state[0]


def test_seeded_parameter_sampling_is_reproducible_and_local():
    distribution = UniformDistribution(0.5, 1.5)
    first = sample_parameters(distribution, 5, seed=42)
    assert first == sample_parameters(distribution, 5, seed=42)
    assert first != sample_parameters(distribution, 5, seed=43)


def test_named_parameter_sampling_is_reproducible():
    distributions = {"growth_rate": UniformDistribution(0.5, 1.5), "capacity": NormalDistribution(100.0, 5.0)}
    first = sample_parameter_sets(distributions, 4, seed=12)
    assert first == sample_parameter_sets(distributions, 4, seed=12)
    assert first != sample_parameter_sets(distributions, 4, seed=13)
    assert tuple(first[0]) == ("growth_rate", "capacity")


def test_latin_hypercube_is_reproducible_and_stratifies_each_uniform_marginal():
    distributions = {"x": UniformDistribution(0.0, 1.0), "y": UniformDistribution(10.0, 20.0)}
    samples = latin_hypercube_parameter_sets(distributions, 8, seed=17)
    assert samples == latin_hypercube_parameter_sets(distributions, 8, seed=17)
    assert samples != latin_hypercube_parameter_sets(distributions, 8, seed=18)
    for name, low, high in (("x", 0.0, 1.0), ("y", 10.0, 20.0)):
        width = (high - low) / 8
        strata = sorted(int((sample[name] - low) / width) for sample in samples)
        assert strata == list(range(8))


def test_latin_hypercube_supports_normal_marginals_without_infinities():
    samples = latin_hypercube_parameter_sets({"rate": NormalDistribution(2.0, 0.5)}, 32, seed=3)
    values = [sample["rate"] for sample in samples]
    assert all(value == value and abs(value) < float("inf") for value in values)
    assert min(values) < 2.0 < max(values)


def test_latin_hypercube_validates_design_shape():
    with pytest.raises(ValueError, match="count"):
        latin_hypercube_parameter_sets({"x": UniformDistribution(0.0, 1.0)}, 0)
    with pytest.raises(ValueError, match="must not be empty"):
        latin_hypercube_parameter_sets({}, 4)


def test_named_ensemble_propagates_multiple_parameters():
    samples = ({"growth_rate": 0.5, "capacity": 80.0}, {"growth_rate": 1.5, "capacity": 120.0})
    members = run_named_ensemble(
        lambda p: LogisticGrowth(growth_rate=p["growth_rate"], carrying_capacity=p["capacity"]),
        (10.0,), samples, ExperimentConfig(duration=1.0, dt=0.1),
    )
    assert members[0].parameter("growth_rate") == 0.5
    assert members[1].parameter_dict == samples[1]
    assert members[0].solution.final_state != members[1].solution.final_state
    assert len(ensemble_statistics(members).mean_states) == len(members[0].solution)


def test_named_ensemble_rejects_invalid_samples():
    config = ExperimentConfig()
    with pytest.raises(ValueError, match="must not be empty"):
        run_named_ensemble(lambda p: LogisticGrowth(), (10.0,), (), config)
    with pytest.raises(ValueError, match="finite"):
        run_named_ensemble(lambda p: LogisticGrowth(), (10.0,), ({"rate": float("nan")},), config)


def test_normal_distribution_validates_standard_deviation():
    with pytest.raises(ValueError, match="std"):
        NormalDistribution(1.0, 0.0)


def test_ensemble_statistics_computes_pointwise_mean_and_std():
    times = (0.0, 1.0)
    members = (EnsembleMember(1.0, ODESolution(times, ((1.0, 2.0), (3.0, 4.0)))), EnsembleMember(2.0, ODESolution(times, ((3.0, 4.0), (5.0, 8.0)))))
    stats = ensemble_statistics(members)
    assert stats.mean_states == ((2.0, 3.0), (4.0, 6.0))
    assert stats.std_states == pytest.approx(((1.0, 1.0), (1.0, 2.0)))


def test_monte_carlo_convergence_tracks_prefix_estimates():
    members = tuple(EnsembleMember(float(value), ODESolution((0.0,), ((float(value),),))) for value in (1.0, 2.0, 3.0, 4.0))
    convergence = monte_carlo_convergence(members, lambda solution: solution.final_state[0], checkpoints=(1, 2, 4))
    assert [estimate.samples for estimate in convergence.estimates] == [1, 2, 4]
    assert [estimate.mean for estimate in convergence.estimates] == pytest.approx([1.0, 1.5, 2.5])
    assert convergence.estimates[0].standard_error == 0.0
    assert convergence.final.std == pytest.approx(5 ** 0.5 / 2)
    assert convergence.final.standard_error == pytest.approx(5 ** 0.5 / 4)


def test_monte_carlo_convergence_validates_checkpoints_and_observable():
    member = EnsembleMember(1.0, ODESolution((0.0,), ((1.0,),)))
    with pytest.raises(ValueError, match="strictly increasing"):
        monte_carlo_convergence((member,), lambda solution: 1.0, checkpoints=(1, 1))
    with pytest.raises(ValueError, match="ensemble size"):
        monte_carlo_convergence((member,), lambda solution: 1.0, checkpoints=(2,))
    with pytest.raises(ValueError, match="finite"):
        monte_carlo_convergence((member,), lambda solution: float("nan"))


def test_ensemble_quantiles_interpolate_pointwise():
    times = (0.0, 1.0)
    members = tuple(EnsembleMember(float(value), ODESolution(times, ((float(value),), (2.0 * value,)))) for value in (0, 10, 20, 30))
    quantiles = ensemble_quantiles(members, (0.25, 0.5, 0.75))
    assert quantiles.states[0] == pytest.approx(((7.5,), (15.0,)))
    assert quantiles.states[1] == pytest.approx(((15.0,), (30.0,)))
    assert quantiles.states[2] == pytest.approx(((22.5,), (45.0,)))


def test_ensemble_quantiles_reject_invalid_probabilities():
    member = EnsembleMember(1.0, ODESolution((0.0,), ((1.0,),)))
    with pytest.raises(ValueError, match="between 0 and 1"):
        ensemble_quantiles((member,), (-0.1, 0.5))
    with pytest.raises(ValueError, match="strictly increasing"):
        ensemble_quantiles((member,), (0.5, 0.5))


def test_ensemble_statistics_rejects_different_time_grids():
    members = (EnsembleMember(1.0, ODESolution((0.0, 1.0), ((1.0,), (2.0,)))), EnsembleMember(2.0, ODESolution((0.0, 0.5, 1.0), ((1.0,), (1.5,), (2.0,)))))
    with pytest.raises(ValueError, match="time grid"):
        ensemble_statistics(members)
