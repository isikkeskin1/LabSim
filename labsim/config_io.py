"""JSON persistence for reproducible experiment configurations."""

from __future__ import annotations

import json
from pathlib import Path

from .experiments import ExperimentConfig


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Serialize an experiment configuration as readable JSON."""
    Path(path).write_text(
        json.dumps(
            {
                "t0": config.t0,
                "duration": config.duration,
                "dt": config.dt,
                "method": config.method,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(
        t0=float(payload.get("t0", 0.0)),
        duration=float(payload["duration"]),
        dt=float(payload["dt"]),
        method=str(payload.get("method", "rk4")),
    )
