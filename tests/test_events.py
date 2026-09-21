import math

import pytest

from labsim import HarmonicOscillator, integrate_ode
from labsim.events import first_crossing, first_crossing_hermite, first_crossing_linear, first_event


def make_ramp():
    return integrate_ode(lambda _time, _state: (1.0,), (-1.0,), dt=0.25, steps=8, method="euler")


def test_first_event_returns_first_matching_sample():
    solution = make_ramp()
    event = first_event(solution, lambda _time, state: state[0] >= 0.0)
    assert event == (1.0, (0.0,))


def test_first_crossing_detects_upward_crossing():
    solution = make_ramp()
    event = first_crossing(solution, 0, 0.0, direction=1)
    assert event == (1.0, (0.0,))


def test_first_crossing_detects_downward_crossing():
    solution = integrate_ode(lambda _time, _state: (-1.0,), (1.0,), dt=0.5, steps=4, method="euler")
    event = first_crossing(solution, 0, 0.0, direction=-1)
    assert event == (1.0, (0.0,))


def test_first_crossing_returns_none_when_no_crossing_exists():
    solution = integrate_ode(lambda _time, _state: (1.0,), (1.0,), dt=0.5, steps=2)
    assert first_crossing(solution, 0, 0.0) is None


def test_linear_crossing_localizes_between_samples():
    solution = integrate_ode(lambda _time, _state: (1.0,), (-0.3,), dt=0.2, steps=4)
    event = first_crossing_linear(solution, 0, 0.0)
    assert event is not None
    assert event[0] == pytest.approx(0.3)
    assert event[1][0] == pytest.approx(0.0)


def test_linear_crossing_can_preserve_sampled_behavior():
    solution = integrate_ode(lambda _time, _state: (1.0,), (-0.3,), dt=0.2, steps=4)
    event = first_crossing_linear(solution, 0, 0.0, interpolate=False)
    assert event is not None
    assert event[0] == pytest.approx(0.4)


def test_hermite_crossing_improves_coarse_oscillator_localization():
    model = HarmonicOscillator()
    solution = integrate_ode(model, (1.0, 0.0), dt=0.6, steps=4, method="rk4")

    linear = first_crossing_linear(solution, 0, 0.0, direction=-1)
    hermite = first_crossing_hermite(solution, model, 0, 0.0, direction=-1)

    assert linear is not None
    assert hermite is not None
    exact_time = math.pi / 2
    assert hermite[0] == pytest.approx(exact_time, abs=5e-3)
    assert hermite[1][0] == pytest.approx(0.0, abs=1e-10)


def test_hermite_crossing_validates_derivative_dimension():
    solution = make_ramp()
    with pytest.raises(ValueError, match="wrong dimension"):
        first_crossing_hermite(solution, lambda _time, _state: (1.0, 2.0), 0)


def test_hermite_crossing_rejects_nonpositive_iterations():
    solution = make_ramp()
    with pytest.raises(ValueError, match="iterations"):
        first_crossing_hermite(solution, lambda _time, _state: (1.0,), 0, iterations=0)
