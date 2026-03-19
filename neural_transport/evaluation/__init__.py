"""Evaluation metrics — single source of truth.

Submodules:
    pointwise   — RMSE, MAE, bias, R2, NSE (NumPy & xarray), error maps/scalars
    ensemble    — CRPS, spread-skill ratio, calibration, rank histogram
    distributional — energy distance, MMD, Wasserstein, power spectrum, etc.
"""

from neural_transport.evaluation.distributional import (  # noqa: F401
    compute_distributional_metrics,
    coverage_density,
    energy_distance,
    meridional_gradient_score,
    mmd_rbf,
    power_spectrum_distance,
    remove_spatial_mean,
    vendi_score,
    wasserstein_1d_marginals,
    zonal_mean_distance,
)
from neural_transport.evaluation.ensemble import (  # noqa: F401
    calibration_score,
    crps,
    crps_ensemble,
    rank_histogram,
    spread_skill_ratio,
)
from neural_transport.evaluation.pointwise import (  # noqa: F401
    METRICS_NP,
    METRICS_XR,
    bias_np,
    bias_xr,
    compute_error_maps,
    compute_error_scalars,
    mae_np,
    mae_xr,
    nse_np,
    nse_xr,
    r2_np,
    r2_xr,
    rel_mean_np,
    rel_mean_xr,
    rel_std_np,
    rel_std_xr,
    rmse_np,
    rmse_xr,
)
from neural_transport.evaluation.suite import EvalResult, EvaluationSuite  # noqa: F401
