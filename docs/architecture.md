# LabSim architecture

LabSim is intentionally split into small layers so numerical methods, physical assumptions, and experiment orchestration can evolve independently.

## Layers

### Solvers
`labsim.solvers` owns time integration. A solver consumes a derivative function and produces an immutable `ODESolution` containing sampled times and states.

### Models
`labsim.models` expresses domain equations through `ODEModel`. Models should own physical parameters and state validation, but should not implement numerical integration themselves.

### Experiments
`labsim.experiments` turns a model, initial state, and `ExperimentConfig` into a reproducible run. Configuration belongs here rather than inside individual physical models.

### Analysis and diagnostics
`labsim.analysis`, `labsim.convergence`, `labsim.metrics`, and `labsim.physics` evaluate numerical error, convergence, trajectory statistics, and physical invariants without changing the simulation itself.

### Data and presentation
`labsim.io` handles portable trajectory export. `labsim.plotting` is optional and lazy-loads Matplotlib so the numerical core remains usable without a plotting stack.

## Design principles

1. **Numerical code stays model-agnostic.** Physical equations should be implemented as models rather than special-cased in solvers.
2. **Experiments are reproducible.** A configuration should be sufficient to recreate a deterministic run.
3. **Validation is first-class.** Known analytical solutions, invariants, and convergence rates should be used to catch numerical regressions.
4. **Optional capabilities stay optional.** Visualization and future integrations should not make the core package heavier than necessary.
5. **Small APIs compose.** Sweeps, event detection, metrics, and exports operate on `ODESolution` so they can be combined without coupling.

Future work can build on this structure with adaptive integration, event localization, uncertainty propagation, richer reporting, and interactive experiment tooling.
