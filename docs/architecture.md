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
`labsim.uncertainty` provides the first uncertainty-propagation layer. `run_ensemble` evaluates a parameterized model over deterministic samples and `ensemble_statistics` computes pointwise means and population standard deviations. Aggregation intentionally requires a common time grid; adaptive trajectories should be reconstructed with `resample_uniform` before ensemble statistics are computed. This keeps uncertainty analysis independent from any particular sampling strategy while making numerical alignment explicit.

The deterministic ensemble API is deliberately small. Future work can add seeded random sampling, parameter distributions, quantiles/confidence bands, multi-parameter samples, and convergence diagnostics without changing the underlying solver or model contracts.

### Data and presentation
`labsim.io` handles portable trajectory export. `labsim.plotting` is optional and lazy-loads Matplotlib so the numerical core remains usable without a plotting stack.

## Design principles

1. **Numerical code stays model-agnostic.** Physical equations should be implemented as models rather than special-cased in solvers.
2. **Experiments are reproducible.** A configuration should be sufficient to recreate a deterministic run.
3. **Validation is first-class.** Known analytical solutions, invariants, convergence rates, and localized events should catch numerical regressions.
4. **Optional capabilities stay optional.** Visualization and future integrations should not make the core package heavier than necessary.
5. **Small APIs compose.** Sweeps, event detection, metrics, dense reconstruction, uncertainty analysis, and exports operate on `ODESolution` so they can be combined without coupling.

The adaptive stack now has a low-order Heun-Euler controller and a third-order Bogacki-Shampine RK2(3) method. Adaptive trajectories can be reconstructed onto arbitrary or uniform output grids without re-integration. The uncertainty layer now supports deterministic scalar-parameter ensembles on aligned grids. Natural next milestones are distribution-backed reproducible sampling and uncertainty summaries such as quantiles, followed by solver-native continuous extensions/event-aware integration and eventually higher-order RK4(5) pairs.
