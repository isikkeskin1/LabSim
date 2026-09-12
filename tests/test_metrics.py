import pytest

from labsim.metrics import final_state_error, root_mean_square
from labsim import integrate_ode


def test_final_state_error_uses_euclidean_distance():
    solution = integrate_ode(lambda _time, _state: (0.0, 0.0), (3.0, 4.0), steps=0)
    assert final_state_error(solution, (0.0, 0.0)) == pytest.approx(5.0)


def test_final_state_error_rejects_dimension_mismatch():
    solution = integrate_ode(lambda _time, _state: (0.0,), (1.0,), steps=0)
    with pytest.raises(ValueError, match="dimension"):
        final_state_error(solution, (1.0, 2.0))


def test_root_mean_square():
    assert root_mean_square((3.0, 4.0)) == pytest.approx(3.5355339)


def test_root_mean_square_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        root_mean_square(())
