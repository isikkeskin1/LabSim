"""Measure observed RK4 convergence on the harmonic oscillator."""

import math

from labsim import HarmonicOscillator, run_convergence_study


def exact(time: float) -> tuple[float, float]:
    return math.cos(time), -math.sin(time)


study = run_convergence_study(
    HarmonicOscillator(),
    (1.0, 0.0),
    exact,
    duration=1.0,
    step_sizes=(0.2, 0.1, 0.05, 0.025),
    method="rk4",
)

for point in study.points:
    print(f"dt={point.dt:.5f} error={point.error:.6e}")
print(f"observed order={study.observed_order:.3f}")
