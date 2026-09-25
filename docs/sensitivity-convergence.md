# Sensitivity convergence

Variance-based sensitivity indices are Monte Carlo estimates, so a single sample budget can give a misleading sense of precision. `sensitivity_convergence` repeats the same variance-attribution calculation at increasing budgets and records how the first-order and total-effect indices move.

```python
from labsim import UniformDistribution, sensitivity_convergence

distributions = {"x": UniformDistribution(0.0, 1.0), "y": UniformDistribution(0.0, 1.0)}
study = sensitivity_convergence(
    distributions,
    lambda p: p["x"] + 2 * p["y"],
    (250, 1000, 4000),
    seed=42,
)
```

The same seed is reused at each checkpoint. Because LabSim's independent sampler is deterministic for a seed, larger designs extend the same random streams and preserve the earlier sample prefix. This makes changes between checkpoints easier to interpret than unrelated reruns.

Convergence is diagnostic rather than a formal confidence interval: stable-looking indices can still have Monte Carlo error, and difficult models may require much larger budgets. Use several increasing checkpoints and look for stabilization of both first-order and total-effect estimates. Bootstrap or replicated uncertainty intervals remain a future extension.
