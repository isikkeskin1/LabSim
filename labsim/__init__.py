"""Core numerical tools for LabSim."""

from .analysis import max_state_error
from .convergence import ConvergencePoint, ConvergenceStudy, run_convergence_study
from .models import HarmonicOscillator, ODEModel
from .solvers import ODESolution, integrate_ode

__all__ = [
    "ConvergencePoint",
    "ConvergenceStudy",
    "HarmonicOscillator",
    "ODEModel",
    "ODESolution",
    "integrate_ode",
    "max_state_error",
    "run_convergence_study",
]
