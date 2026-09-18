# LabSim architecture

LabSim is intentionally split into small layers so numerical methods, physical assumptions, and experiment orchestration can evolve independently.

## Layers

### Solvers
`labsim.solvers` owns the original fixed-step Euler/RK4 path and the embedded Heun-Euler adaptive integrator. `labsim.adaptive` contains higher-order adaptive methods; its Bogacki-Shampine RK2(3) implementation accepts the same derivative/state abstraction and returns the same immutable `ODESolution` type. Keeping the higher-order controller separate prevents the core solver module from becoming a collection of unrelated Butcher tableaux while preserving a common result interface.

### Models
`labsim.models` expresses domain equations through `ODEModel`. Models own physical parameters and state validation, but do not implement numerical integration.

### Experiments
`labsim.experiments` turns a model, initial state, and `ExperimentConfig` into a reproducible run. Configuration belongs here rather than inside individual physical models.

### Analysis and diagnostics
`labsim.analysis`, `labsim.convergence`, `labsim.metrics`, and `labsim.physics` evaluate numerical error, convergence, trajectory statistics, and physical invariants without changing the simulation itself.

### Events
`labsim.events` detects predicates and threshold crossings on completed trajectories. Sampled crossings remain available for backwards-compatible behavior, linear localization estimates an event between bracketing samples, and cubic Hermite localization uses endpoint derivatives to provide a higher-fidelity estimate without re-integrating the trajectory.

Hermite localization deliberately accepts the derivative function explicitly rather than storing solver internals in `ODESolution`. That keeps solutions lightweight and lets the same event API work with fixed-step and adaptive trajectories as long as the governing derivative is available.

### Dense reconstruction
`labsim.dense` generalizes the same endpoint-derivative idea from one event location to arbitrary trajectory sampling. `sample_hermite` reconstructs states at requested in-domain times with cubic Hermite interpolation, while `resample_uniform` converts irregular adaptive output to a regular grid suitable for comparison, export, and plotting. Endpoint derivatives are evaluated lazily and cached during a sampling call.

This is post-processing dense output rather than solver-native continuous extension: it needs only an `ODESolution` and the governing derivative, so it works uniformly across the current fixed-step and adaptive solvers. A future solver-native dense representation can provide method-specific interpolation polynomials while retaining these high-level sampling operations.

### Uncertainty and ensembles
`labsim.uncertainty` separates parameter sampling, simulation, and aggregation. Deterministic parameter sequences remain valid inputs to `run_ensemble`, while `UniformDistribution`, `NormalDistribution`, and `sample_parameters` provide seeded stochastic sampling without mutating Python's global random-number-generator state. This makes a seed plus distribution parameters sufficient to reproduce the sampled model parameters.

`ensemble_statistics` computes pointwise means and population standard deviations, and `ensemble_quantiles` provides empirical uncertainty bands using linearly interpolated sample quantiles. Both aggregation paths intentionally require a common time grid; adaptive trajectories should first be reconstructed with `resample_uniform`. Keeping alignment explicit prevents uncertainty statistics from silently depending on an interpolation policy.

The current distribution abstraction is intentionally scalar. Multi-parameter uncertainty should introduce named parameter samples rather than overloading the scalar `EnsembleMember.parameter` field. Future work can also add correlated sampling, Latin-hypercube or quasi-random designs, and ensemble convergence diagnostics without changing the solver contracts.

### Data and presentation
`labsim.io` handles portable trajectory export. `labsim.plotting` is optional and lazy-loads Matplotlib so the numerical core remains usable without a plotting stack.

## Design principles

1. **Numerical code stays model-agnostic.** Physical equations should be implemented as models rather than special-cased in solvers.
2. **Experiments are reproducible.** A configuration should be sufficient to recreate a deterministic run, and stochastic studies must make their random seed explicit.
3. **Validation is first-class.** Known analytical solutions, invariants, convergence rates, localized events, and statistical regression tests should catch numerical regressions.
4. **Optional capabilities stay optional.** Visualization and future integrations should not make the core package heavier than necessary.
5. **Small APIs compose.** Sweeps, event detection, metrics, dense reconstruction, uncertainty analysis, and exports operate on `ODESolution` so they can be combined without coupling.

The adaptive stack now has a low-order Heun-Euler controller and a third-order Bogacki-Shampine RK2(3) method. Adaptive trajectories can be reconstructed onto arbitrary or uniform output grids without re-integration. The uncertainty layer supports deterministic or seeded distribution-backed scalar-parameter ensembles, pointwise moments, and empirical quantile bands. Natural next milestones are named multi-parameter samples and convergence diagnostics for Monte Carlo studies, followed by solver-native continuous extensions/event-aware integration and eventually higher-order RK4(5) pairs.
