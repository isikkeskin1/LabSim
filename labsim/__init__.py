"""Core numerical tools for LabSim."""

from .analysis import max_state_error
from .convergence import ConvergencePoint, ConvergenceStudy, run_convergence_study
from .experiments import ExperimentConfig, run_experiment
from .io import write_csv
from .metrics import final_state_error, root_mean_square
from .models import DampedOscillator, HarmonicOscillator, LogisticGrowth, LotkaVolterra, ODEModel
from .physics import harmonic_energy, relative_drift
from .sensitivity import central_difference
from .solvers import ODESolution, integrate_ode
from .sweeps import SweepResult, parameter_sweep

__all__ = [
    "ConvergencePoint",
    "ConvergenceStudy",
    "DampedOscillator",
    "ExperimentConfig",
    "HarmonicOscillator",
    "LogisticGrowth",
    "LotkaVolterra",
    "ODEModel",
    "ODESolution",
    "SweepResult",
    "central_difference",
    "final_state_error",
    "harmonic_energy",
    "integrate_ode",
    "max_state_error",
    "parameter_sweep",
    "relative_drift",
    "root_mean_square",
    "run_convergence_study",
    "run_experiment",
    "write_csv",
]
