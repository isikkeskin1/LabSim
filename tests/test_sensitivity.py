import pytest

from labsim.sensitivity import central_difference


def test_central_difference_matches_quadratic_derivative():
    assert central_difference(lambda x: x * x, 3.0, 1e-4) == pytest.approx(6.0, rel=1e-7)


def test_central_difference_rejects_non_positive_step():
    with pytest.raises(ValueError, match="positive"):
        central_difference(lambda x: x, 1.0, 0.0)
