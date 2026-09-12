"""Simulate a nonlinear pendulum and report its first downward crossing."""

import math

from labsim import ExperimentConfig, SimplePendulum, first_crossing, run_experiment


model = SimplePendulum(length=1.0)
config = ExperimentConfig(duration=8.0, dt=0.005, method="rk4")
solution = run_experiment(model, (math.radians(45.0), 0.0), config)

crossing = first_crossing(solution, component=0, threshold=0.0, direction=-1)
print(f"samples: {len(solution)}")
if crossing is not None:
    time, state = crossing
    print(f"first downward zero crossing: t={time:.4f}, angular velocity={state[1]:.4f}")
else:
    print("no downward zero crossing detected")
