import math

import pytest

from labsim import integrate_rk23


def test_rk23_solves_exponential_decay_to_requested_accuracy():
    solution = integrate_rk23(
        lambda _time, state: (-state[0],),
        (1.0,),
        duration=5.0,
        dt=0.5,
        rtol=1e-6,
        atol=1e-9,
    )

    assert solution.final_time == pytest.approx(5.0)
    assert solution.final_state[0] == pytest.approx(math.exp(-5.0), rel=2e-5)


def test_rk23_adapts_sample_spacing():
    solution = integrate_rk23(
        lambda _time, state: (-20.0 * state[0],),
        (1.0,),
        duration=1.0,
        dt=0.2,
        rtol=1e-5,
        atol=1e-8,
        max_dt=0.2,
    )
    steps = [right - left for left, right in zip(solution.times, solution.times[1:])]

    assert min(steps) < max(steps)
    assert solution.final_time == pytest.approx(1.0)


def test_rk23_rejects_derivative_dimension_mismatch():
    with pytest.raises(ValueError, match="same dimension"):
        integrate_rk23(lambda _time, _state: (1.0, 2.0), (0.0,), duration=0.1)


def test_rk23_validates_tolerances_and_step_limits():
    derivative = lambda _time, state: (-state[0],)

    with pytest.raises(ValueError, match="rtol and atol"):
        integrate_rk23(derivative, (1.0,), rtol=0.0)
    with pytest.raises(ValueError, match="min_dt"):
        integrate_rk23(derivative, (1.0,), dt=0.01, min_dt=0.1)
    with pytest.raises(ValueError, match="max_steps"):
        integrate_rk23(derivative, (1.0,), max_steps=0)
