"""Standalone metrics module for OSSE evaluation.

All metric functions accept numpy arrays. Dataclasses for structured results.
Extracted and extended from compare_conditioning_osse.py (experiment 08).
"""

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from neural_transport.tools.metrics import crps


def _to_numpy(x) -> np.ndarray:
    """Convert torch.Tensor or xr.DataArray to numpy ndarray."""
    if hasattr(x, "values"):  # xarray
        return x.values
    if hasattr(x, "detach"):  # torch
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_xco2_column(field, pressure_weights, ak):
    """Compute column-averaged XCO2 from 3D field.

    XCO2 = sum(h_k * a_k * x_k) over levels axis.

    Parameters
    ----------
    field : np.ndarray, shape [..., nlev]
    pressure_weights : np.ndarray, broadcastable to field shape
    ak : np.ndarray, broadcastable to field shape

    Returns
    -------
    np.ndarray, shape [...]  (levels axis summed out)
    """
    return (pressure_weights * ak * field).sum(axis=-1)


def rmse_3d(pred, gt, weights=None):
    """Weighted RMSE over full 3D field.

    Parameters
    ----------
    pred, gt : np.ndarray, shape [nlat, nlon, nlev]
    weights : np.ndarray, shape [nlat] (cos-lat), optional.
        Auto-broadcast to [nlat, 1, 1].

    Returns
    -------
    float
    """
    diff = pred - gt
    if weights is not None:
        w = np.asarray(weights).reshape(-1, 1, 1)
        w = w / w.mean()
        return float(np.sqrt(np.mean(w * diff ** 2)))
    return float(np.sqrt(np.mean(diff ** 2)))


def rmse_xco2(pred, gt, pressure_weights, ak, weights=None):
    """RMSE of column-averaged XCO2.

    Parameters
    ----------
    pred, gt : np.ndarray, shape [nlat, nlon, nlev]
    pressure_weights, ak : np.ndarray, shape [nlat, nlon, nlev]
    weights : np.ndarray, shape [nlat], optional (cos-lat)

    Returns
    -------
    float
    """
    xco2_pred = compute_xco2_column(pred, pressure_weights, ak)
    xco2_gt = compute_xco2_column(gt, pressure_weights, ak)
    diff = xco2_pred - xco2_gt
    if weights is not None:
        w = np.asarray(weights).reshape(-1, 1)
        w = w / w.mean()
        return float(np.sqrt(np.mean(w * diff ** 2)))
    return float(np.sqrt(np.mean(diff ** 2)))


def rmse_at_obs(pred, gt, mask_2d, weights=None):
    """RMSE restricted to observed spatial locations.

    Parameters
    ----------
    pred, gt : np.ndarray, shape [nlat, nlon, nlev]
    mask_2d : np.ndarray, shape [nlat, nlon], bool
    weights : unused (kept for API consistency)

    Returns
    -------
    float or nan if mask is None or empty.
    """
    if mask_2d is None or not mask_2d.any():
        return np.nan
    mask_3d = mask_2d[:, :, None].repeat(gt.shape[-1], axis=-1)
    diff = pred - gt
    return float(np.sqrt(np.mean(diff[mask_3d] ** 2)))


def rmse_away(pred, gt, mask_2d, weights=None):
    """RMSE restricted to unobserved spatial locations.

    Parameters
    ----------
    pred, gt : np.ndarray, shape [nlat, nlon, nlev]
    mask_2d : np.ndarray, shape [nlat, nlon], bool
    weights : unused (kept for API consistency)

    Returns
    -------
    float or nan if mask is None or no unobserved locations.
    """
    if mask_2d is None or not (~mask_2d).any():
        return np.nan
    mask_3d = mask_2d[:, :, None].repeat(gt.shape[-1], axis=-1)
    diff = pred - gt
    return float(np.sqrt(np.mean(diff[~mask_3d] ** 2)))


def crps_ensemble(samples, gt):
    """Thin wrapper around neural_transport.tools.metrics.crps().

    Parameters
    ----------
    samples : np.ndarray, shape [n_samples, nlat, nlon, (nlev)]
    gt : np.ndarray, shape [nlat, nlon, (nlev)]

    Returns
    -------
    crps_map : np.ndarray, shape [nlat, nlon, (nlev)]
    crps_mean : float
    """
    # For 3D fields, compute CRPS on column-mean to avoid axis reduction bug
    # in the upstream crps() when crps_map is 3D.
    if gt.ndim == 3:
        # Compute per-level CRPS by collapsing to 2D per level, then average
        nlev = gt.shape[-1]
        crps_maps = []
        crps_means = []
        for lev in range(nlev):
            cmap, cmean = crps(samples[:, :, :, lev], gt[:, :, lev])
            crps_maps.append(cmap)
            crps_means.append(cmean)
        crps_map = np.stack(crps_maps, axis=-1)  # [nlat, nlon, nlev]
        crps_mean = float(np.mean(crps_means))
    else:
        crps_map, crps_mean = crps(samples, gt)
    return crps_map, crps_mean


def spread_skill_ratio(samples, gt):
    """Ratio of ensemble std to ensemble-mean absolute error.

    Parameters
    ----------
    samples : np.ndarray, shape [n_samples, nlat, nlon, nlev]
    gt : np.ndarray, shape [nlat, nlon, nlev]

    Returns
    -------
    float
    """
    ens_mean = samples.mean(axis=0)
    ens_spread = samples.std(axis=0)
    ens_error = np.abs(ens_mean - gt)
    return float(np.mean(ens_spread) / max(np.mean(ens_error), 1e-12))


def calibration_score(samples, gt, quantiles=None):
    """PIT calibration: fraction of GT below ensemble quantiles.

    For each nominal quantile q, compute the fraction of grid points where
    the ground truth falls below the q-th percentile of the ensemble.
    A well-calibrated ensemble has observed fractions matching nominal ones.

    Parameters
    ----------
    samples : np.ndarray, shape [n_samples, ...]
    gt : np.ndarray, shape [...]
    quantiles : np.ndarray, optional. Defaults to linspace(0.05, 0.95, 19).

    Returns
    -------
    dict with keys 'nominal', 'observed', 'calibration_error'.
    """
    if quantiles is None:
        quantiles = np.linspace(0.05, 0.95, 19)
    quantiles = np.asarray(quantiles)

    # Compute ensemble percentiles at each quantile
    percentiles = np.percentile(samples, quantiles * 100, axis=0)  # [n_quantiles, ...]

    # For each quantile, fraction of grid points where gt < percentile
    gt_expanded = gt[None, ...]  # [1, ...]
    observed = np.array([
        float(np.mean(gt_expanded[0] < percentiles[i]))
        for i in range(len(quantiles))
    ])

    calibration_error = float(np.mean((observed - quantiles) ** 2) ** 0.5)

    return {
        "nominal": quantiles.tolist(),
        "observed": observed.tolist(),
        "calibration_error": calibration_error,
    }


def rank_histogram(samples, gt):
    """Talagrand rank histogram.

    For each grid point, compute the rank of the ground truth among ensemble
    members. A uniform histogram indicates a well-calibrated ensemble.

    Parameters
    ----------
    samples : np.ndarray, shape [n_samples, ...]
    gt : np.ndarray, shape [...]

    Returns
    -------
    np.ndarray, shape [n_samples + 1]. Histogram counts.
    """
    n_samples = samples.shape[0]
    # Count how many ensemble members are below gt at each grid point
    ranks = np.sum(samples < gt[None, ...], axis=0).ravel()  # [N_gridpoints]
    histogram = np.bincount(ranks, minlength=n_samples + 1).astype(float)
    # Normalize to fractions
    histogram = histogram / histogram.sum()
    return histogram


def spatial_roughness(field_2d):
    """Laplacian second-differences roughness measure.

    Parameters
    ----------
    field_2d : np.ndarray, shape [nlat, nlon]

    Returns
    -------
    dict with 'roughness_lat' and 'roughness_lon'.
    """
    laplacian_lat = np.diff(field_2d, n=2, axis=0)
    laplacian_lon = np.diff(field_2d, n=2, axis=1)
    return {
        "roughness_lat": float(np.std(laplacian_lat)),
        "roughness_lon": float(np.std(laplacian_lon)),
    }


@dataclass
class MetricsResult:
    """Structured container for all OSSE evaluation metrics."""

    rmse_3d_full: float = np.nan
    rmse_3d_obs: float = np.nan
    rmse_3d_away: float = np.nan
    rmse_xco2_full: float = np.nan
    rmse_xco2_obs: float = np.nan
    rmse_xco2_away: float = np.nan
    rmse_col: float = np.nan
    r2: float = np.nan
    spread_skill: float = np.nan
    roughness_lat: float = np.nan
    roughness_lon: float = np.nan
    sample_spread: float = np.nan
    crps_mean: float = np.nan
    calibration_error: float = np.nan

    def to_dict(self):
        return asdict(self)


@dataclass
class OSSEResult:
    """Full result container for a single OSSE experiment."""

    name: str
    config: dict
    metrics: MetricsResult
    samples: np.ndarray  # [n_samples, nlat, nlon, nlev]
    ensemble_mean: np.ndarray  # [nlat, nlon, nlev]
    gt: np.ndarray  # [nlat, nlon, nlev]
    mask_2d: Optional[np.ndarray] = None  # [nlat, nlon] bool
    pressure_weights: Optional[np.ndarray] = None  # [nlat, nlon, nlev]
    ak: Optional[np.ndarray] = None  # [nlat, nlon, nlev]
    rank_hist: Optional[np.ndarray] = None  # [n_samples + 1]
    calibration_data: Optional[dict] = None
    crps_map: Optional[np.ndarray] = None  # [nlat, nlon, (nlev)]
    lat: Optional[np.ndarray] = None  # [nlat]
    lon: Optional[np.ndarray] = None  # [nlon]


def compute_all_metrics(samples, gt, mask_2d=None, pressure_weights=None,
                        ak=None, lat=None):
    """Orchestrator: compute all metrics from ensemble samples and ground truth.

    Parameters
    ----------
    samples : np.ndarray, shape [n_samples, nlat, nlon, nlev]
    gt : np.ndarray, shape [nlat, nlon, nlev]
    mask_2d : np.ndarray, shape [nlat, nlon], bool, optional
    pressure_weights : np.ndarray, shape [nlat, nlon, nlev], optional
    ak : np.ndarray, shape [nlat, nlon, nlev], optional
    lat : np.ndarray, shape [nlat], optional (for cos-lat weighting)

    Returns
    -------
    metrics : MetricsResult
    extra_data : dict with rank_hist, calibration_data, crps_map
    """
    ens_mean = samples.mean(axis=0)  # [nlat, nlon, nlev]

    # Cos-lat weights
    cos_w = None
    if lat is not None:
        cos_w = np.cos(np.radians(lat))

    # 3D RMSE
    val_rmse_3d_full = rmse_3d(ens_mean, gt, weights=cos_w)
    val_rmse_3d_obs = rmse_at_obs(ens_mean, gt, mask_2d)
    val_rmse_3d_away = rmse_away(ens_mean, gt, mask_2d)

    # Column-mean RMSE (average over levels, then spatial RMSE)
    diff_col = (ens_mean - gt).mean(axis=-1)  # [nlat, nlon]
    if cos_w is not None:
        cos_w_2d = cos_w.reshape(-1, 1)
        cos_w_2d = cos_w_2d / cos_w_2d.mean()
        val_rmse_col = float(np.sqrt(np.mean(cos_w_2d * diff_col ** 2)))
    else:
        val_rmse_col = float(np.sqrt(np.mean(diff_col ** 2)))

    # R2
    diff = ens_mean - gt
    if cos_w is not None:
        w3d = cos_w.reshape(-1, 1, 1)
        w3d = w3d / w3d.mean()
        ss_res = np.sum(w3d * diff ** 2)
        gt_mean = np.mean(w3d * gt) / np.mean(w3d)
        ss_tot = np.sum(w3d * (gt - gt_mean) ** 2)
    else:
        ss_res = np.sum(diff ** 2)
        gt_mean = np.mean(gt)
        ss_tot = np.sum((gt - gt_mean) ** 2)
    val_r2 = float(1 - ss_res / max(ss_tot, 1e-12))

    # XCO2 metrics
    val_rmse_xco2_full = np.nan
    val_rmse_xco2_obs = np.nan
    val_rmse_xco2_away = np.nan
    if pressure_weights is not None and ak is not None:
        val_rmse_xco2_full = rmse_xco2(ens_mean, gt, pressure_weights, ak,
                                        weights=cos_w)
        # XCO2 at obs / away
        if mask_2d is not None:
            h_ak = pressure_weights * ak
            xco2_pred = compute_xco2_column(ens_mean, pressure_weights, ak)
            xco2_gt = compute_xco2_column(gt, pressure_weights, ak)
            xco2_diff = xco2_pred - xco2_gt
            if mask_2d.any():
                val_rmse_xco2_obs = float(
                    np.sqrt(np.mean(xco2_diff[mask_2d] ** 2)))
            if (~mask_2d).any():
                val_rmse_xco2_away = float(
                    np.sqrt(np.mean(xco2_diff[~mask_2d] ** 2)))

    # Spread-skill ratio
    val_spread_skill = spread_skill_ratio(samples, gt)

    # Sample spread
    val_sample_spread = float(samples.std(axis=0).mean())

    # Spatial roughness
    pred_col = ens_mean.mean(axis=-1)  # [nlat, nlon]
    roughness = spatial_roughness(pred_col)

    # CRPS
    try:
        crps_map_val, val_crps_mean = crps_ensemble(samples, gt)
    except Exception:
        crps_map_val = None
        val_crps_mean = np.nan

    # Calibration
    try:
        cal_data = calibration_score(samples, gt)
        val_calibration_error = cal_data["calibration_error"]
    except Exception:
        cal_data = None
        val_calibration_error = np.nan

    # Rank histogram
    try:
        rank_hist_val = rank_histogram(samples, gt)
    except Exception:
        rank_hist_val = None

    metrics = MetricsResult(
        rmse_3d_full=val_rmse_3d_full,
        rmse_3d_obs=val_rmse_3d_obs,
        rmse_3d_away=val_rmse_3d_away,
        rmse_xco2_full=val_rmse_xco2_full,
        rmse_xco2_obs=val_rmse_xco2_obs,
        rmse_xco2_away=val_rmse_xco2_away,
        rmse_col=val_rmse_col,
        r2=val_r2,
        spread_skill=val_spread_skill,
        roughness_lat=roughness["roughness_lat"],
        roughness_lon=roughness["roughness_lon"],
        sample_spread=val_sample_spread,
        crps_mean=val_crps_mean,
        calibration_error=val_calibration_error,
    )

    extra_data = {
        "rank_hist": rank_hist_val,
        "calibration_data": cal_data,
        "crps_map": crps_map_val,
    }

    return metrics, extra_data
