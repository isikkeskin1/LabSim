"""Propagate uncertainty in logistic growth rate through an ensemble."""

from labsim import ExperimentConfig, LogisticGrowth, ensemble_statistics, run_ensemble


config = ExperimentConfig(duration=5.0, dt=0.05, method="rk4")
members = run_ensemble(
    lambda rate: LogisticGrowth(growth_rate=rate, carrying_capacity=100.0),
    (10.0,),
    (0.8, 0.9, 1.0, 1.1, 1.2),
    config,
)
stats = ensemble_statistics(members)

print("growth_rate,final_population")
for member in members:
    print(f"{member.parameter:.2f},{member.solution.final_state[0]:.6f}")
print(f"ensemble final mean: {stats.mean_states[-1][0]:.6f}")
print(f"ensemble final std:  {stats.std_states[-1][0]:.6f}")
