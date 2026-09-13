# LabSim architecture

LabSim is intentionally split into small layers so numerical methods, physical assumptions, and experiment orchestration can evolve independently.

## Layers

### Solvers
`labsim.solvers` owns time integration. Fixed-step solvers and adaptive solvers consume a derivative function and produce immutable `ODESolution` objects containing sampled times and states.

### Models
`labsim.models` expresses domain equations through `ODEModel`. Models own physical parameters and state validation, but do not implement numerical integration.

### Experiments
`labsim.experiments` turns a model, initial state, and `ExperimentConfig` into a reproducible run. Configuration belongs here rather than inside individual physical models.

### Analysis and diagnostics
`labsim.analysis`, `labsim.convergence`, `labsim.metrics`, and `labsim.physics` evaluate numerical error, convergence, trajectory statistics, and physical invariants without changing the simulation itself.

### Events
`labsim.events` detects predicates and threshold crossings on completed trajectories. Sampled crossings remain available for backwards-compatible behavior, while linear localization estimates an event between bracketing samples without modifying the numerical solution.

### Data and presentation
`labsim.io` handles portable trajectory export. `labsim.plotting` is optional and lazy-loads Matplotlib so the numerical core remains usable without a plotting stack.

## Design principles

1. **Numerical code stays model-agnostic.** Physical equations should be implemented as models rather than special-cased in solvers.
2. **Experiments are reproducible.** A configuration should be sufficient to recreate a deterministic run.
3. **Validation is first-class.** Known analytical solutions, invariants, convergence rates, and localized events should catch numerical regressions.
4. **Optional capabilities stay optional.** Visualization and future integrations should not make the core package heavier than necessary.
5. **Small APIs compose.** Sweeps, event detection, metrics, and exports operate on `ODESolution` so they can be combined without coupling.

Future work can build on this structure with dense-output event localization, higher-order adaptive methods, uncertainty propagation, richer reporting, and interactive experiment tooling.
