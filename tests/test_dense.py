import math

import pytest

from labsim import integrate_rk23
from labsim.dense import resample_uniform, sample_hermite


def decay(_time, state):
    return (-state[0],)


def test_hermite_sampling_reconstructs_exponential_decay():
    solution = integrate_rk23(
        decay,
        (1.0,),
        duration=2.0,
        dt=0.5,
        rtol=1e-5,
        atol=1e-8,
    )
    requested = (0.0, 0.25, 0.75, 1.25, 2.0)
    dense = sample_hermite(solution, decay, requested)

    assert dense.times == requested
    for time, state in zip(dense.times, dense.states):
        assert state[0] == pytest.approx(math.exp(-time), rel=5e-4)


def test_uniform_resampling_includes_original_domain_endpoints():
    solution = integrate_rk23(decay, (1.0,), duration=1.0, dt=0.3)
    dense = resample_uniform(solution, decay, dt=0.2)

    assert dense.times[0] == pytest.approx(0.0)
    assert dense.times[-1] == pytest.approx(1.0)
    assert len(dense.times) == 6


def test_exact_solver_samples_are_preserved():
    solution = integrate_rk23(decay, (1.0,), duration=0.5, dt=0.1)
    dense = sample_hermite(solution, decay, solution.times)

    assert dense.states == solution.states


def test_dense_sampling_rejects_out_of_domain_and_unsorted_times():
    solution = integrate_rk23(decay, (1.0,), duration=1.0, dt=0.1)

    with pytest.raises(ValueError, match="domain"):
        sample_hermite(solution, decay, (-0.1, 0.5))
    with pytest.raises(ValueError, match="non-decreasing"):
        sample_hermite(solution, decay, (0.5, 0.25))
    with pytest.raises(ValueError, match="positive"):
        resample_uniform(solution, decay, dt=0.0)


def test_dense_sampling_validates_derivative_dimension():
    solution = integrate_rk23(decay, (1.0,), duration=1.0, dt=0.1)

    with pytest.raises(ValueError, match="wrong dimension"):
        sample_hermite(solution, lambda _time, _state: (1.0, 2.0), (0.05,))
