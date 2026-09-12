"""Run a normalized SIR outbreak experiment and report peak infection."""

from labsim import ExperimentConfig, SIRModel, run_experiment


model = SIRModel(transmission_rate=0.8, recovery_rate=0.2)
config = ExperimentConfig(duration=40.0, dt=0.02, method="rk4")
solution = run_experiment(model, (0.99, 0.01, 0.0), config)

peak_index = max(range(len(solution.states)), key=lambda index: solution.states[index][1])
peak_time = solution.times[peak_index]
peak_infected = solution.states[peak_index][1]
print(f"peak infected fraction: {peak_infected:.4f}")
print(f"peak time: {peak_time:.2f}")
print(f"population at final time: {sum(solution.final_state):.6f}")
