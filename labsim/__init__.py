"""Core numerical tools for LabSim."""

from .analysis import max_state_error
from .convergence import ConvergencePoint, ConvergenceStudy, run_convergence_study
from .io import write_csv
from .models import DampedOscillator, HarmonicOscillator, ODEModel
from .physics import harmonic_energy, relative_drift
from .solvers import ODESolution, integrate_ode

__all__ = [
    "ConvergencePoint",
    "ConvergenceStudy",
    "DampedOscillator",
    "HarmonicOscillator",
    "ODEModel",
    "ODESolution",
    "harmonic_energy",
    "integrate_ode",
    "max_state_error",
    "relative_drift",
    "run_convergence_study",
    "write_csv",
]
