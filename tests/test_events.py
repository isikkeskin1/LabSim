from labsim import integrate_ode
from labsim.events import first_crossing, first_crossing_linear, first_event


def make_ramp():
    return integrate_ode(lambda _time, _state: (1.0,), (-1.0,), dt=0.25, steps=8)


def test_first_event_returns_first_matching_sample():
    solution = make_ramp()
    event = first_event(solution, lambda _time, state: state[0] >= 0.0)
    assert event == (1.0, (0.0,))


def test_first_crossing_detects_upward_crossing():
    solution = make_ramp()
    event = first_crossing(solution, 0, 0.0, direction=1)
    assert event == (1.0, (0.0,))


def test_first_crossing_detects_downward_crossing():
    solution = integrate_ode(lambda _time, _state: (-1.0,), (1.0,), dt=0.5, steps=4)
    event = first_crossing(solution, 0, 0.0, direction=-1)
    assert event == (1.0, (0.0,))


def test_first_crossing_returns_none_when_no_crossing_exists():
    solution = integrate_ode(lambda _time, _state: (1.0,), (1.0,), dt=0.5, steps=2)
    assert first_crossing(solution, 0, 0.0) is None


def test_linear_crossing_localizes_between_samples():
    solution = integrate_ode(lambda _time, _state: (1.0,), (-0.3,), dt=0.2, steps=4)
    event = first_crossing_linear(solution, 0, 0.0)
    assert event is not None
    assert event[0] == 0.3
    assert event[1][0] == 0.0


def test_linear_crossing_can_preserve_sampled_behavior():
    solution = integrate_ode(lambda _time, _state: (1.0,), (-0.3,), dt=0.2, steps=4)
    event = first_crossing_linear(solution, 0, 0.0, interpolate=False)
    assert event is not None
    assert event[0] == 0.4
