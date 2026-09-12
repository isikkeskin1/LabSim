import pytest

from labsim import DampedOscillator, ExperimentConfig, run_experiment


def test_experiment_config_calculates_steps():
    config = ExperimentConfig(duration=1.0, dt=0.1)
    assert config.steps == 10


def test_run_experiment_uses_model_state_dimension():
    solution = run_experiment(
        DampedOscillator(damping=0.2),
        (1.0, 0.0),
        ExperimentConfig(duration=0.2, dt=0.1),
    )
    assert len(solution) == 3
    assert len(solution.states[-1]) == 2


def test_experiment_rejects_incompatible_duration():
    config = ExperimentConfig(duration=1.0, dt=0.3)
    with pytest.raises(ValueError, match="integer multiple"):
        _ = config.steps


def test_experiment_rejects_wrong_initial_state():
    with pytest.raises(ValueError, match="state_size"):
        run_experiment(
            DampedOscillator(),
            (1.0,),
            ExperimentConfig(duration=0.1, dt=0.1),
        )
