from labsim import DampedOscillator, ExperimentConfig, parameter_sweep


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
