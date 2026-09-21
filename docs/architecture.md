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
`labsim.uncertainty` separates parameter sampling, simulation, and aggregation. Scalar studies remain supported by `run_ensemble`, while `sample_parameter_sets` and `run_named_ensemble` extend the same workflow to several independently sampled, named parameters. A named ensemble member stores an immutable parameter snapshot so results retain the exact inputs that produced them.

`UniformDistribution` and `NormalDistribution` use a local seeded RNG. For multi-parameter samples, one RNG drives the distributions in mapping insertion order, making a seed plus an ordered distribution specification sufficient to reproduce the complete sample matrix without mutating Python's global RNG state.

`latin_hypercube_parameter_sets` adds a variance-reduced design for independent marginals. It partitions each marginal CDF into equally probable strata, draws once inside every stratum, and independently permutes the resulting values between dimensions. Distribution quantile transforms map those unit-interval samples into uniform or normal physical parameters. This guarantees one-dimensional stratification without pretending to model correlation between parameters.

`ensemble_statistics` and `ensemble_quantiles` accept both scalar and named members. Aggregation deliberately requires a common time grid; adaptive trajectories should first be reconstructed with `resample_uniform`. This prevents statistics from silently depending on an interpolation policy.

`monte_carlo_convergence` evaluates a scalar observable over one already-computed ensemble and records prefix estimates at requested sample counts. Each checkpoint reports the running mean, population standard deviation, and estimated standard error of the mean. Reusing prefixes makes convergence inspection deterministic and avoids rerunning simulations merely to compare sample sizes. The observable is explicit, so the same ensemble can be checked for final state, peak response, event time, energy drift, or another scalar scientific quantity.

Named sampling currently assumes independent marginals. Correlated distributions should be introduced as an explicit joint-distribution abstraction rather than hidden inside individual scalar distributions. Latin hypercube sampling improves marginal coverage but intentionally does not change that independence model. Natural next uncertainty milestones are comparing estimator efficiency between plain Monte Carlo and stratified designs, then adding explicit correlated/joint parameter distributions.

### Data and presentation
`labsim.io` handles portable trajectory export. `labsim.plotting` is optional and lazy-loads Matplotlib.

## Design principles

1. **Numerical code stays model-agnostic.** Physical equations belong in models rather than solvers.
2. **Experiments are reproducible.** Deterministic configuration and stochastic seeds must be explicit.
3. **Validation is first-class.** Analytical solutions, invariants, convergence rates, localized events, and statistical regression tests guard numerical behavior.
4. **Optional capabilities stay optional.** Presentation dependencies do not burden the numerical core.
5. **Small APIs compose.** Sweeps, events, metrics, dense reconstruction, uncertainty analysis, and exports share `ODESolution` rather than coupling to one solver.

The adaptive stack now spans low-order and third-order embedded methods plus reusable dense reconstruction. The uncertainty layer supports scalar and named multi-parameter ensembles, seeded independent random sampling, Latin hypercube stratification, pointwise moments, empirical quantile bands, and checkpointed Monte Carlo convergence diagnostics. Natural next milestones are estimator-efficiency diagnostics and correlated sampling designs, followed by solver-native continuous extensions/event-aware integration.
