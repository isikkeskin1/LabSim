"""Compare the parameter coverage of random and Latin-hypercube designs."""

from labsim import (
    ExperimentConfig,
    LogisticGrowth,
    UniformDistribution,
    ensemble_quantiles,
    latin_hypercube_parameter_sets,
    run_named_ensemble,
)


distributions = {
    "growth_rate": UniformDistribution(0.6, 1.4),
    "carrying_capacity": UniformDistribution(80.0, 120.0),
}
samples = latin_hypercube_parameter_sets(distributions, 32, seed=21)
members = run_named_ensemble(
    lambda p: LogisticGrowth(
        growth_rate=p["growth_rate"],
        carrying_capacity=p["carrying_capacity"],
    ),
    (10.0,),
    samples,
    ExperimentConfig(duration=5.0, dt=0.05),
)
bands = ensemble_quantiles(members, (0.05, 0.5, 0.95))

print("final population quantiles")
for probability, trajectory in zip(bands.probabilities, bands.states):
    print(f"q={probability:.2f}: {trajectory[-1][0]:.3f}")
