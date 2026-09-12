"""Small reproducible benchmark for fixed-step ODE solver scaling.

Run with ``python benchmarks/solver_scaling.py`` from the repository root.
"""

from __future__ import annotations

import time

from labsim import HarmonicOscillator, integrate_ode


def benchmark(method: str, steps: int) -> float:
    model = HarmonicOscillator()
    start = time.perf_counter()
    integrate_ode(model, (1.0, 0.0), dt=1.0 / steps, steps=steps, method=method)
    return time.perf_counter() - start


def main() -> None:
    print("method,steps,seconds")
    for method in ("euler", "rk4"):
        for steps in (1_000, 10_000, 100_000):
            elapsed = benchmark(method, steps)
            print(f"{method},{steps},{elapsed:.6f}")


if __name__ == "__main__":
    main()
