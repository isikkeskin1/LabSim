import math

import pytest

from labsim import integrate_adaptive, integrate_ode


def test_rk4_solves_exponential_decay_with_high_accuracy():
    solution = integrate_ode(lambda _t, y: (-y[0],), (1.0,), dt=0.1, steps=10)
    assert solution.times[0] == 0.0
    assert solution.times[-1] == pytest.approx(1.0)
    assert solution.states[-1][0] == pytest.approx(math.exp(-1), rel=1e-5)


def test_euler_is_available_for_simple_linear_system():
    solution = integrate_ode(lambda _t, y: (2 * y[0],), (1.0,), dt=0.1, steps=2, method="euler")
    assert solution.states == ((1.0,), (1.2,), (1.44,))


def test_dimension_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same dimension"):
        integrate_ode(lambda _t, _y: (1.0, 2.0), (0.0,), dt=0.1)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        integrate_ode(lambda _t, y: y, (1.0,), dt=0)
    with pytest.raises(ValueError, match="method"):
        integrate_ode(lambda _t, y: y, (1.0,), method="bogus")


def test_adaptive_solver_reaches_requested_end_time():
    solution = integrate_adaptive(
        lambda _t, y: (-y[0],),
        (1.0,),
        duration=1.0,
        dt=0.2,
        rtol=1e-5,
        atol=1e-8,
    )
    assert solution.times[0] == 0.0
    assert solution.final_time == pytest.approx(1.0)
    assert solution.final_state[0] == pytest.approx(math.exp(-1), rel=2e-4)
    assert len(solution) > 2


def test_adaptive_solver_can_limit_maximum_step():
    solution = integrate_adaptive(
        lambda _t, y: (1.0,),
        (0.0,),
        duration=1.0,
        dt=0.2,
        max_dt=0.05,
    )
    assert max(right - left for left, right in zip(solution.times, solution.times[1:])) <= 0.05 + 1e-12


def test_adaptive_solver_rejects_invalid_tolerances():
    with pytest.raises(ValueError, match="rtol"):
        integrate_adaptive(lambda _t, y: y, (1.0,), rtol=0.0)
