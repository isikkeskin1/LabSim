# LabSim architecture

LabSim is split into small layers so numerical methods, physical assumptions, and experiment orchestration can evolve independently.

## Layers

### Solvers
`labsim.solvers` owns fixed-step Euler/RK4 and the embedded Heun-Euler adaptive integrator. `labsim.adaptive` contains higher-order adaptive methods, currently Bogacki-Shampine RK2(3). All return the immutable `ODESolution` interface.

### Models
`labsim.models` expresses domain equations through `ODEModel`. Models own physical parameters and state validation, not numerical integration.

### Experiments
`labsim.experiments` combines a model, initial state, and `ExperimentConfig` into a reproducible run.

### Analysis and diagnostics
`labsim.analysis`, `labsim.convergence`, `labsim.metrics`, and `labsim.physics` evaluate numerical error, convergence, trajectory statistics, and physical invariants without changing simulations.

### Events and dense reconstruction
`labsim.events` provides sampled, linear, and cubic-Hermite event localization. `labsim.dense` generalizes endpoint-derivative interpolation to arbitrary sampling and uniform resampling. These remain post-processing operations over `ODESolution`, leaving room for future solver-native continuous extensions.

### Uncertainty and ensembles
`labsim.uncertainty` separates parameter sampling, simulation, and aggregation. Scalar studies remain supported by `run_ensemble`, while named ensembles extend the workflow to several independently sampled parameters. Seeded Monte Carlo, Latin hypercube designs, empirical quantiles, convergence diagnostics, and replicated sampling-efficiency comparisons remain explicit and composable.

`labsim.joint` introduces explicit dependence between uncertain parameters through correlated Gaussian designs. Independent and correlated sampling remain separate concepts so dependence is never silently destroyed by an independent design.

### Variance-based sensitivity
`labsim.sensitivity_variance` attributes scalar-response variance across independent uncertain inputs. `variance_sensitivity` constructs two independent Monte Carlo matrices and one hybrid matrix per parameter, reporting Saltelli-style first-order and Jansen total-effect indices.

`labsim.sensitivity_convergence` treats those indices as Monte Carlo estimates rather than exact outputs. `sensitivity_convergence` repeats attribution at strictly increasing sample budgets and records each complete result. Reusing the same seed preserves deterministic sample prefixes as budgets grow, making stabilization or drift in first-order and total-effect indices directly inspectable.

`labsim.sensitivity_uncertainty` complements convergence studies with replicated estimator uncertainty. It reruns the same sensitivity design using independent child seeds and reports the mean plus a central empirical interval for every first-order and total-effect index. These intervals quantify Monte Carlo estimator variability; they are not intervals for physical model discrepancy, measurement uncertainty, or correlated-input attribution.

Sensitivity observables remain model-agnostic: callers may analyze an analytical response or wrap a complete simulation and extract final state, peak response, event time, energy drift, or another scalar quantity. The estimator assumes independent marginals; applying these indices directly to correlated designs would change their interpretation and is intentionally unsupported.

### Data and presentation
`labsim.io` handles portable trajectory export. `labsim.plotting` is optional and lazy-loads Matplotlib.

## Design principles

1. **Numerical code stays model-agnostic.** Physical equations belong in models rather than solvers.
2. **Experiments are reproducible.** Deterministic configuration and stochastic seeds must be explicit.
3. **Validation is first-class.** Analytical solutions, invariants, convergence rates, localized events, and statistical regression tests guard numerical behavior.
4. **Optional capabilities stay optional.** Presentation dependencies do not burden the numerical core.
5. **Small APIs compose.** Sweeps, events, metrics, dense reconstruction, uncertainty analysis, and exports share `ODESolution` rather than coupling to one solver.

The adaptive stack now spans low-order and third-order embedded methods plus reusable dense reconstruction. The uncertainty layer supports scalar and named ensembles, seeded independent sampling, Latin hypercube stratification, correlated Gaussian designs, empirical uncertainty bands, Monte Carlo convergence diagnostics, sampling-efficiency comparisons, first-order/total-effect variance attribution, sensitivity-index convergence studies, and replicated empirical uncertainty intervals. Natural next milestones are broader joint-distribution support and more efficient sensitivity resampling, followed by solver-native continuous extensions/event-aware integration.
