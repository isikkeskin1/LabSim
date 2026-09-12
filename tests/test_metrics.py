import pytest

from labsim import integrate_ode
from labsim.metrics import root_mean_square, state_range, time_average


def test_state_range_measures_component_span():
    solution = integrate_ode(lambda _time, state: (state[0],), (1.0,), dt=0.5, steps=2, method="euler")
    assert state_range(solution, 0) == pytest.approx(2.25)


def test_time_average_uses_trapezoidal_area():
    solution = integrate_ode(lambda _time, _state: (1.0,), (0.0,), dt=0.5, steps=4)
    assert time_average(solution, 0) == pytest.approx(1.0)


def test_time_average_of_single_sample_is_that_value():
    solution = integrate_ode(lambda _time, _state: (0.0,), (3.0,), steps=0)
    assert time_average(solution, 0) == 3.0


def test_invalid_component_is_rejected():
    solution = integrate_ode(lambda _time, _state: (0.0,), (1.0,), steps=0)
    with pytest.raises(IndexError):
        state_range(solution, 2)


def test_rms_stays_available():
    assert root_mean_square((3.0, 4.0)) == pytest.approx(3.5355339)
