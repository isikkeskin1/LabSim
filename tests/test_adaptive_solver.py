import math

import pytest

from labsim import integrate_adaptive


def test_adaptive_solver_reaches_requested_final_time():
    solution = integrate_adaptive(
        lambda _time, state: (-state[0],),
        (1.0,),
        duration=2.0,
        dt=0.2,
        rtol=1e-5,
        atol=1e-8,
    )
    assert solution.final_time == pytest.approx(2.0)
    assert solution.final_state[0] == pytest.approx(math.exp(-2), rel=2e-3)


def test_adaptive_solver_uses_more_than_one_step_when_needed():
    solution = integrate_adaptive(
        lambda _time, state: (-10.0 * state[0],),
        (1.0,),
        duration=1.0,
        dt=0.5,
        rtol=1e-6,
        atol=1e-9,
    )
    assert len(solution) > 3


def test_adaptive_solver_rejects_invalid_tolerances():
    with pytest.raises(ValueError):
        integrate_adaptive(lambda _t, y: y, (1.0,), rtol=0.0)
    with pytest.raises(ValueError):
        integrate_adaptive(lambda _t, y: y, (1.0,), min_dt=0.1, dt=0.01)
