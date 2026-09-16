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

### Data and presentation
`labsim.io` handles portable trajectory export. `labsim.plotting` is optional and lazy-loads Matplotlib so the numerical core remains usable without a plotting stack.

## Design principles

1. **Numerical code stays model-agnostic.** Physical equations should be implemented as models rather than special-cased in solvers.
2. **Experiments are reproducible.** A configuration should be sufficient to recreate a deterministic run.
3. **Validation is first-class.** Known analytical solutions, invariants, convergence rates, and localized events should catch numerical regressions.
4. **Optional capabilities stay optional.** Visualization and future integrations should not make the core package heavier than necessary.
5. **Small APIs compose.** Sweeps, event detection, metrics, dense reconstruction, and exports operate on `ODESolution` so they can be combined without coupling.

The adaptive stack now has a low-order Heun-Euler controller and a third-order Bogacki-Shampine RK2(3) method. Adaptive trajectories can be reconstructed onto arbitrary or uniform output grids without re-integration. Future numerical work can build from this toward solver-native continuous extensions, event-aware integration, and higher-order RK4(5) pairs. Beyond solvers, uncertainty propagation, richer reporting, and interactive experiment tooling remain natural roadmap stages.
