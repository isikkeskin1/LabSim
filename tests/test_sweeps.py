import pytest

from labsim import DampedOscillator, ExperimentConfig, HarmonicOscillator, parameter_sweep
from labsim.sweeps import summarize_sweep


def test_parameter_sweep_runs_each_parameter_once():
    results = parameter_sweep(
        lambda damping: DampedOscillator(damping=damping),
        (1.0, 0.0),
        (0.0, 0.2, 0.4),
        ExperimentConfig(duration=0.2, dt=0.1),
    )
    assert [result.parameter for result in results] == [0.0, 0.2, 0.4]
    assert all(len(result.solution) == 3 for result in results)


def test_parameter_sweep_can_change_physical_response():
    results = parameter_sweep(
        lambda damping: DampedOscillator(damping=damping),
        (1.0, 0.0),
        (0.0, 1.0),
        ExperimentConfig(duration=1.0, dt=0.05),
    )
    assert results[0].solution.states[-1][0] != results[1].solution.states[-1][0]


def test_summarize_sweep_returns_final_state_norms():
    config = ExperimentConfig(duration=0.1, dt=0.1)
    results = parameter_sweep(
        lambda frequency: HarmonicOscillator(angular_frequency=frequency),
        (1.0, 0.0),
        (1.0, 2.0),
        config,
    )
    summary = summarize_sweep(results)
    assert tuple(item.parameter for item in summary) == (1.0, 2.0)
    assert summary[0].final_state == results[0].solution.final_state
    assert summary[0].final_norm > 0


def test_summarize_sweep_rejects_empty_results():
    with pytest.raises(ValueError, match="must not be empty"):
        summarize_sweep(())
