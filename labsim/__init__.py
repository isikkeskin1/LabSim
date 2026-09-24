"""Core numerical tools for LabSim."""

from .adaptive import integrate_rk23
from .analysis import max_state_error
from .config_io import load_config, save_config
from .convergence import ConvergencePoint, ConvergenceStudy, run_convergence_study
from .dense import resample_uniform, sample_hermite
from .events import first_crossing, first_crossing_hermite, first_crossing_linear, first_event
from .experiments import ExperimentConfig, run_experiment
from .io import write_csv
from .joint import CorrelatedNormalDistribution, sample_correlation, sample_joint_parameter_sets
from .metrics import final_state_error, root_mean_square, state_range, time_average
from .models import DampedOscillator, HarmonicOscillator, LogisticGrowth, LotkaVolterra, ODEModel, SIRModel, SimplePendulum
from .physics import harmonic_energy, relative_drift
from .sensitivity import central_difference
from .sensitivity_variance import SensitivityIndex, VarianceSensitivity, variance_sensitivity
from .solvers import ODESolution, integrate_adaptive, integrate_ode
from .sweeps import SweepResult, SweepSummary, parameter_sweep, summarize_sweep
from .uncertainty import (
    EnsembleMember, EnsembleQuantiles, EnsembleStatistics, MonteCarloConvergence,
    MonteCarloEstimate, NamedEnsembleMember, NormalDistribution, SamplingEfficiency,
    UniformDistribution, compare_sampling_efficiency, ensemble_quantiles,
    ensemble_statistics, latin_hypercube_parameter_sets, monte_carlo_convergence,
    run_ensemble, run_named_ensemble, sample_parameter_sets, sample_parameters,
)

__all__ = [
    "ConvergencePoint", "ConvergenceStudy", "CorrelatedNormalDistribution", "DampedOscillator", "EnsembleMember", "EnsembleQuantiles", "EnsembleStatistics", "ExperimentConfig",
    "HarmonicOscillator", "LogisticGrowth", "LotkaVolterra", "MonteCarloConvergence", "MonteCarloEstimate", "NamedEnsembleMember", "NormalDistribution", "ODEModel", "ODESolution",
    "SIRModel", "SamplingEfficiency", "SensitivityIndex", "SimplePendulum", "SweepResult", "SweepSummary", "UniformDistribution", "VarianceSensitivity", "central_difference", "compare_sampling_efficiency", "ensemble_quantiles", "ensemble_statistics",
    "final_state_error", "first_crossing", "first_crossing_hermite", "first_crossing_linear", "first_event", "harmonic_energy",
    "integrate_adaptive", "integrate_ode", "integrate_rk23", "latin_hypercube_parameter_sets", "load_config", "max_state_error", "monte_carlo_convergence", "parameter_sweep", "relative_drift",
    "resample_uniform", "root_mean_square", "run_convergence_study", "run_ensemble", "run_experiment", "run_named_ensemble",
    "sample_correlation", "sample_hermite", "sample_joint_parameter_sets", "sample_parameter_sets", "sample_parameters", "save_config", "state_range", "summarize_sweep", "time_average", "variance_sensitivity", "write_csv",
]
