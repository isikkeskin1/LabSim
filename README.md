# LabSim

LabSim is an open-source numerical simulation toolkit for building reproducible scientific experiments in Python.

## What exists today

The project is organized as a small simulation stack rather than a collection of standalone formulas:

- **ODE solvers** — fixed-step Euler and classical RK4 integration;
- **models** — reusable dynamical-system interfaces plus harmonic, damped-oscillator, and Lotka-Volterra models;
- **validation** — exact-solution error measurements and convergence-order studies;
- **experiment framework** — validated configurations, reproducible runs, and parameter sweeps;
- **diagnostics** — energy, relative drift, final-state error, and RMS metrics;
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

## Repository structure

```text
labsim/
  analysis.py       exact-solution error analysis
  cli.py            command-line interface
  convergence.py    convergence studies
  experiments.py    reproducible simulation configuration
  io.py              trajectory export
  metrics.py        general simulation metrics
  models.py         reusable dynamical-system models
  physics.py        physical diagnostics
  plotting.py       optional Matplotlib visualization
  solvers.py        numerical integration core
  sweeps.py         parameter studies
examples/           executable scientific experiments
benchmarks/         repeatable performance measurements
tests/              regression and behavior tests
```

## Roadmap

The next development stages will focus on richer parameterized models, experiment metadata and persistence, uncertainty/sensitivity analysis, benchmark reporting, visualization improvements, and eventually a small interactive experiment runner. The emphasis is on numerical correctness, reproducibility, and reviewable increments.
