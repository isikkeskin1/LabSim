"""Compare Euler and RK4 on the same harmonic oscillator experiment."""

import math

from labsim import HarmonicOscillator, integrate_ode, max_state_error


def exact(time: float, _state: tuple[float, ...]) -> tuple[float, float]:
    return math.cos(time), -math.sin(time)


model = HarmonicOscillator()
for method in ("euler", "rk4"):
    solution = integrate_ode(model, (1.0, 0.0), dt=0.05, steps=100, method=method)
    error = max_state_error(solution, exact)
    print(f"{method:>5}: max error = {error:.6e}")
