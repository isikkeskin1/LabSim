# LabSim

LabSim is an open-source numerical simulation toolkit being built as a foundation for reproducible scientific experiments.

## Current milestone

The project now has a reusable ordinary differential equation (ODE) model layer on top of its numerical integration core. It currently provides:

- fixed-step Euler integration;
- classical fourth-order Runge-Kutta (RK4) integration;
- immutable sampled solution objects containing time/state trajectories;
- a common `ODEModel` interface for reusable dynamical systems;
- a validated harmonic oscillator model;
- pytest coverage for solver accuracy, model behavior, and failure cases.

Example:

```python
from labsim import HarmonicOscillator, integrate_ode

oscillator = HarmonicOscillator(angular_frequency=2.0)
solution = integrate_ode(oscillator, (1.0, 0.0), dt=0.01, steps=500)

print(solution.times[-1], solution.states[-1])
```

The next milestones will add more reusable physical models, numerical validation and convergence studies, benchmarking, experiment configuration, and visualization. Each milestone should remain independently useful and tested.

## Development

```bash
python -m pip install -e ".[test]"
pytest
```
