import pytest

from labsim import ExperimentConfig, LogisticGrowth
from labsim.uncertainty import EnsembleMember, ensemble_statistics, run_ensemble
from labsim.solvers import ODESolution


def test_run_ensemble_preserves_parameter_samples():
    members = run_ensemble(
        lambda rate: LogisticGrowth(growth_rate=rate, carrying_capacity=100.0),
        (10.0,),
        (0.5, 1.0, 1.5),
        ExperimentConfig(duration=1.0, dt=0.1),
    )

    assert [member.parameter for member in members] == [0.5, 1.0, 1.5]
    assert members[0].solution.final_state[0] < members[-1].solution.final_state[0]


def test_ensemble_statistics_computes_pointwise_mean_and_std():
    times = (0.0, 1.0)
    members = (
        EnsembleMember(1.0, ODESolution(times, ((1.0, 2.0), (3.0, 4.0)))),
        EnsembleMember(2.0, ODESolution(times, ((3.0, 4.0), (5.0, 8.0)))),
    )

    stats = ensemble_statistics(members)

    assert stats.mean_states == ((2.0, 3.0), (4.0, 6.0))
    assert stats.std_states == pytest.approx(((1.0, 1.0), (1.0, 2.0)))


def test_ensemble_statistics_rejects_different_time_grids():
    members = (
        EnsembleMember(1.0, ODESolution((0.0, 1.0), ((1.0,), (2.0,)))),
        EnsembleMember(2.0, ODESolution((0.0, 0.5, 1.0), ((1.0,), (1.5,), (2.0,)))),
    )

    with pytest.raises(ValueError, match="time grid"):
        ensemble_statistics(members)


def test_run_ensemble_rejects_empty_samples():
    with pytest.raises(ValueError, match="must not be empty"):
        run_ensemble(
            lambda rate: LogisticGrowth(growth_rate=rate),
            (10.0,),
            (),
            ExperimentConfig(),
        )
