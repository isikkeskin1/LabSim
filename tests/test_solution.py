import pytest

from labsim import integrate_ode


def test_solution_exposes_final_sample_metadata():
    solution = integrate_ode(lambda _time, _state: (2.0,), (1.0,), dt=0.25, steps=4)

    assert solution.final_time == pytest.approx(1.0)
    assert solution.final_state == pytest.approx((3.0,))
    assert solution.state_dimension == 1


def test_solution_rejects_mismatched_sample_counts():
    from labsim.solvers import ODESolution

    with pytest.raises(ValueError, match="same number"):
        ODESolution((0.0,), ((1.0,), (2.0,)))


def test_solution_rejects_inconsistent_dimensions():
    from labsim.solvers import ODESolution

    with pytest.raises(ValueError, match="same.*dimension"):
        ODESolution((0.0, 1.0), ((1.0,), (1.0, 2.0)))
