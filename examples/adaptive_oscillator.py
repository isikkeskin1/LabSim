"""Demonstrate adaptive integration on a harmonic oscillator."""

from labsim import HarmonicOscillator, integrate_adaptive


model = HarmonicOscillator(angular_frequency=4.0)
solution = integrate_adaptive(
    model,
    (1.0, 0.0),
    duration=2.0,
    dt=0.1,
    rtol=1e-7,
    atol=1e-9,
)

print(f"samples: {len(solution)}")
print(f"final time: {solution.final_time:.6f}")
print(f"final state: {solution.final_state}")
