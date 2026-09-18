"""Propagate uncertain logistic growth rate into trajectory quantile bands."""

from labsim import ExperimentConfig, LogisticGrowth
from labsim.uncertainty import (
    NormalDistribution,
    ensemble_quantiles,
    run_ensemble,
    sample_parameters,
)

rates = sample_parameters(NormalDistribution(mean=0.8, std=0.08), 200, seed=2026)
members = run_ensemble(
    lambda rate: LogisticGrowth(growth_rate=rate, carrying_capacity=100.0),
    (10.0,),
    rates,
    ExperimentConfig(duration=5.0, dt=0.05),
)
bands = ensemble_quantiles(members, (0.05, 0.5, 0.95))

for probability, trajectory in zip(bands.probabilities, bands.states):
    print(f"q={probability:.2f}: final population={trajectory[-1][0]:.3f}")
