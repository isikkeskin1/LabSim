"""Portable serialization helpers for simulation results."""

from __future__ import annotations

import csv
from pathlib import Path

from .solvers import ODESolution


def write_csv(solution: ODESolution, path: str | Path) -> None:
    """Write a sampled trajectory to CSV with one column per state variable."""
    destination = Path(path)
    if not solution.states:
        raise ValueError("solution must contain at least one state")
    width = len(solution.states[0])
    if any(len(state) != width for state in solution.states):
        raise ValueError("solution states must have consistent dimensions")

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", *[f"state_{index}" for index in range(width)]])
        writer.writerows((time, *state) for time, state in zip(solution.times, solution.states))
