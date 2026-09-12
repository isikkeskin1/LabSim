# LabSim

LabSim is an open-source numerical simulation toolkit for building reproducible scientific experiments in Python.

## What exists today

The project is organized as a small simulation stack rather than a collection of standalone formulas:

- **ODE solvers** — fixed-step Euler and classical RK4, plus adaptive Heun-Euler integration;
- **models** — reusable dynamical-system interfaces plus harmonic, damped-oscillator, pendulum, Lotka-Volterra, logistic-growth, and SIR models;
- **validation** — exact-solution error measurements and convergence-order studies;
- **experiment framework** — validated configurations, reproducible runs, and parameter sweeps;
- **diagnostics** — energy, relative drift, final-state error, RMS, range, and time-average metrics;
- **events** — first-event and directional threshold-crossing detection on trajectories;
- **data export** — trajectory CSV output;
- **visualization** — optional lazy-loaded Matplotlib plotting;
- **CLI** — `python -m labsim` or the installed `labsim` command;
- **benchmarks** — repeatable solver scaling measurements;
- **CI** — automated tests across Python 3.10–3.13.

## Quick start

```bash
python -m pip install -e ".[test]"
pytest
```

Run a built-in experiment:

```bash
python -m labsim oscillator --duration 6.28 --dt 0.01 --method rk4 --output oscillator.csv
```

Use the Python API:

```python
from labsim import ExperimentConfig, HarmonicOscillator, run_experiment

model = HarmonicOscillator(angular_frequency=2.0)
config = ExperimentConfig(duration=3.0, dt=0.01, method="rk4")
solution = run_experiment(model, (1.0, 0.0), config)

print(solution.final_time)
print(solution.final_state)
```

Adaptive integration is available directly when a variable time grid is preferable:

```python
from labsim import integrate_adaptive

solution = integrate_adaptive(
    lambda _t, y: (-y[0],),
    (1.0,),
    duration=8.0,
    dt=0.5,
    rtol=1e-6,
    atol=1e-9,
)
```

## Repository structure

```text
labsim/
  analysis.py       exact-solution error analysis
  cli.py            command-line interface
  config_io.py      experiment configuration persistence
  convergence.py    convergence studies
  events.py         trajectory event detection
  experiments.py    reproducible simulation configuration
  io.py              trajectory export
  metrics.py        general simulation metrics
  models.py         reusable dynamical-system models
  physics.py        physical diagnostics
  plotting.py       optional Matplotlib visualization
  sensitivity.py    parameter sensitivity utilities
  solvers.py        fixed and adaptive numerical integration
  sweeps.py         parameter studies and summaries
examples/           executable scientific experiments
benchmarks/         repeatable performance measurements
docs/               architecture and numerical-validation notes
tests/              regression and behavior tests
```

## Development principles

LabSim keeps numerical methods model-agnostic, treats validation as part of implementation, and prefers composable APIs over monolithic experiment classes. Optional capabilities such as plotting stay outside the core dependency path. See `docs/architecture.md` and `docs/numerical-validation.md` for the design rationale.

## Roadmap

Future milestones will add event localization, richer uncertainty propagation, batched experiment execution, benchmark reporting, more physically motivated models, improved visualization, and eventually an interactive experiment runner. The emphasis remains numerical correctness, reproducibility, and reviewable increments.
