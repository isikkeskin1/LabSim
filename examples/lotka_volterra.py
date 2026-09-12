"""Run a nonlinear predator-prey experiment and export the trajectory."""

from labsim import ExperimentConfig, LotkaVolterra, run_experiment, write_csv


model = LotkaVolterra()
config = ExperimentConfig(duration=20.0, dt=0.01, method="rk4")
solution = run_experiment(model, (10.0, 5.0), config)
write_csv(solution, "lotka_volterra.csv")

print(f"samples={len(solution)}")
print(f"final populations={solution.final_state}")
