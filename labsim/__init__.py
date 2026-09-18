"""Core numerical tools for LabSim."""

from .adaptive import integrate_rk23
from .analysis import max_state_error
from .config_io import load_config, save_config
from .convergence import ConvergencePoint, ConvergenceStudy, run_convergence_study
from .dense import resample_uniform, sample_hermite
from .events import first_crossing, first_crossing_hermite, first_crossing_linear, first_event
from .experiments import ExperimentConfig, run_experiment
from .io import write_csv
from .metrics import final_state_error, root_mean_square, state_range, time_average
from .models import DampedOscillator, HarmonicOscillator, LogisticGrowth, LotkaVolterra, ODEModel, SIRModel, SimplePendulum
from .physics import harmonic_energy, relative_drift
from .sensitivity import central_difference
from .solvers import ODESolution, integrate_adaptive, integrate_ode
from .sweeps import SweepResult, SweepSummary, parameter_sweep, summarize_sweep
from .uncertainty import (
    EnsembleMember,
    EnsembleQuantiles,
    EnsembleStatistics,
    NormalDistribution,
    UniformDistribution,
    ensemble_quantiles,
    ensemble_statistics,
    run_ensemble,
    sample_parameters,
)

__all__ = [
    "ConvergencePoint", "ConvergenceStudy", "DampedOscillator", "EnsembleMember", "EnsembleQuantiles", "EnsembleStatistics", "ExperimentConfig",
    "HarmonicOscillator", "LogisticGrowth", "LotkaVolterra", "NormalDistribution", "ODEModel", "ODESolution",
    "SIRModel", "SimplePendulum", "SweepResult", "SweepSummary", "UniformDistribution", "central_difference", "ensemble_quantiles", "ensemble_statistics",
    "final_state_error", "first_crossing", "first_crossing_hermite", "first_crossing_linear", "first_event",
    "harmonic_energy", "integrate_adaptive", "integrate_ode", "integrate_rk23", "load_config", "max_state_error",
    "parameter_sweep", "relative_drift", "resample_uniform", "root_mean_square", "run_convergence_study",
    "run_ensemble", "run_experiment", "sample_hermite", "sample_parameters", "save_config", "state_range", "summarize_sweep", "time_average", "write_csv",
]
