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

`compare_sampling_efficiency` measures whether that stratification actually improves a scalar estimator. It repeats plain Monte Carlo and Latin-hypercube designs at the same sample budget, evaluates an explicit parameter-set observable, and reports the between-replication standard deviation of each estimator plus the Monte-Carlo-to-LHS variance ratio. Replication seeds are derived from a local seeded RNG, so comparisons are repeatable and do not alter global random state. This diagnostic intentionally measures estimator variability rather than assuming that Latin hypercube is always superior; interactions and non-monotone responses can reduce or reverse its benefit.

`labsim.joint` introduces explicit dependence between uncertain parameters. `CorrelatedNormalDistribution` defines named Gaussian marginals together with a positive-definite correlation matrix and samples them through a dependency-free Cholesky transform. `sample_joint_parameter_sets` retains the same local-seed reproducibility contract as independent designs, while `sample_correlation` provides a small diagnostic for checking realized dependence. Correlation is therefore represented in the design itself rather than being hidden inside scalar marginal distributions.

The first joint abstraction is intentionally multivariate normal rather than a generic copula API. This keeps the statistical contract precise: means and standard deviations describe Gaussian marginals, and the supplied matrix is their Pearson correlation matrix. Broader non-Gaussian dependence can later be added as a separate copula/joint-distribution interface without changing the meaning of existing independent samplers.

`ensemble_statistics` and `ensemble_quantiles` accept both scalar and named members. Aggregation deliberately requires a common time grid; adaptive trajectories should first be reconstructed with `resample_uniform`. This prevents statistics from silently depending on an interpolation policy.

`monte_carlo_convergence` evaluates a scalar observable over one already-computed ensemble and records prefix estimates at requested sample counts. Each checkpoint reports the running mean, population standard deviation, and estimated standard error of the mean. Reusing prefixes makes convergence inspection deterministic and avoids rerunning simulations merely to compare sample sizes. The observable is explicit, so the same ensemble can be checked for final state, peak response, event time, energy drift, or another scalar scientific quantity.

Independent and correlated sampling are now separate, explicit concepts. Latin hypercube sampling currently targets independent marginals only; applying stratification while preserving dependence requires a dedicated design rather than silently reordering correlated draws. The next uncertainty milestone is sensitivity-oriented variance attribution, followed by broader copula-based joint distributions if needed.

### Data and presentation
`labsim.io` handles portable trajectory export. `labsim.plotting` is optional and lazy-loads Matplotlib.

## Design principles

1. **Numerical code stays model-agnostic.** Physical equations belong in models rather than solvers.
2. **Experiments are reproducible.** Deterministic configuration and stochastic seeds must be explicit.
3. **Validation is first-class.** Analytical solutions, invariants, convergence rates, localized events, and statistical regression tests guard numerical behavior.
4. **Optional capabilities stay optional.** Presentation dependencies do not burden the numerical core.
5. **Small APIs compose.** Sweeps, events, metrics, dense reconstruction, uncertainty analysis, and exports share `ODESolution` rather than coupling to one solver.

The adaptive stack now spans low-order and third-order embedded methods plus reusable dense reconstruction. The uncertainty layer supports scalar and named multi-parameter ensembles, seeded independent random sampling, Latin hypercube stratification, explicit correlated Gaussian designs, pointwise moments, empirical quantile bands, checkpointed Monte Carlo convergence diagnostics, and replicated sampling-efficiency comparisons. Natural next milestones are variance-attribution sensitivity designs, followed by broader joint-distribution support and solver-native continuous extensions/event-aware integration.
