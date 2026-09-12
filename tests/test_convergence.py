import math

import pytest

from labsim import HarmonicOscillator, run_convergence_study


def exact_oscillator(time: float) -> tuple[float, float]:
    return math.cos(time), -math.sin(time)


def test_rk4_convergence_is_close_to_fourth_order():
    model = HarmonicOscillator()
    study = run_convergence_study(
        model,
        (1.0, 0.0),
        exact_oscillator,
        duration=1.0,
        step_sizes=(0.2, 0.1, 0.05, 0.025),
        method="rk4",
    )

    assert study.observed_order == pytest.approx(4.0, abs=0.15)
    assert study.points[-1].error < study.points[0].error


def test_convergence_rejects_incompatible_duration():
    with pytest.raises(ValueError, match="integer multiple"):
        run_convergence_study(
            HarmonicOscillator(),
            (1.0, 0.0),
            exact_oscillator,
            duration=1.0,
            step_sizes=(0.3,),
        )
