"""Command-line entry points for small LabSim experiments."""

from __future__ import annotations

import argparse

from .experiments import ExperimentConfig, run_experiment
from .io import write_csv
from .models import DampedOscillator, HarmonicOscillator, LotkaVolterra


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a built-in LabSim experiment")
    parser.add_argument("model", choices=("oscillator", "damped", "lotka-volterra"))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--method", choices=("euler", "rk4"), default="rk4")
    parser.add_argument("--output", help="optional CSV output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = {
        "oscillator": (HarmonicOscillator(), (1.0, 0.0)),
        "damped": (DampedOscillator(), (1.0, 0.0)),
        "lotka-volterra": (LotkaVolterra(), (10.0, 5.0)),
    }
    model, initial_state = models[args.model]
    solution = run_experiment(
        model,
        initial_state,
        ExperimentConfig(duration=args.duration, dt=args.dt, method=args.method),
    )
    if args.output:
        write_csv(solution, args.output)
    final = solution.states[-1]
    print(f"model={args.model} samples={len(solution)} final_state={final}")
    return 0
