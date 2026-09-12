# LabSim

LabSim is an open-source numerical simulation toolkit being built as a foundation for reproducible scientific experiments.

## Current milestone

The first milestone establishes a small, dependency-light ordinary differential equation (ODE) engine. It currently provides:

- fixed-step Euler integration;
- classical fourth-order Runge-Kutta (RK4) integration;
- immutable sampled solution objects containing time/state trajectories;
- validation for step size, state dimensions, and solver selection;
- pytest coverage for accuracy and failure cases.

Example:

```python
from labsim import integrate_ode

solution = integrate_ode(
    lambda t, y: (-0.5 * y[0],),
    (10.0,),
    dt=0.01,
    steps=1000,
    method="rk4",
)

print(solution.times[-1], solution.states[-1])
```

The project will grow incrementally from these numerical primitives toward reusable physical models, experiment configuration, validation, benchmarking, and visualization. Each milestone should remain independently useful and tested.

## Development

```bash
python -m pip install -e ".[test]"
pytest
```
