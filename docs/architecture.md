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

`ensemble_statistics` and `ensemble_quantiles` accept both scalar and named members. Aggregation deliberately requires a common time grid; adaptive trajectories should first be reconstructed with `resample_uniform`. This prevents statistics from silently depending on an interpolation policy.

Named sampling currently assumes independent marginals. Correlated distributions should be introduced as an explicit joint-distribution abstraction rather than hidden inside individual scalar distributions. The next uncertainty milestone is Monte Carlo convergence diagnostics so users can measure whether estimated moments have stabilized as ensemble size grows; variance-reduction designs such as Latin hypercube and quasi-random sampling can follow.

### Data and presentation
`labsim.io` handles portable trajectory export. `labsim.plotting` is optional and lazy-loads Matplotlib.

## Design principles

1. **Numerical code stays model-agnostic.** Physical equations belong in models rather than solvers.
2. **Experiments are reproducible.** Deterministic configuration and stochastic seeds must be explicit.
3. **Validation is first-class.** Analytical solutions, invariants, convergence rates, localized events, and statistical regression tests guard numerical behavior.
4. **Optional capabilities stay optional.** Presentation dependencies do not burden the numerical core.
5. **Small APIs compose.** Sweeps, events, metrics, dense reconstruction, uncertainty analysis, and exports share `ODESolution` rather than coupling to one solver.

The adaptive stack now spans low-order and third-order embedded methods plus reusable dense reconstruction. The uncertainty layer supports scalar and named multi-parameter ensembles, seeded independent sampling, pointwise moments, and empirical quantile bands. Natural next milestones are Monte Carlo convergence diagnostics, correlated sampling designs, and then solver-native continuous extensions/event-aware integration.
