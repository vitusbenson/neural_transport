"""Standalone metrics module for OSSE evaluation.

All metric functions accept numpy arrays. Dataclasses for structured results.
Extracted and extended from compare_conditioning_osse.py (experiment 08).
"""

from dataclasses import asdict, dataclass

import numpy as np

from neural_transport.evaluation.ensemble import (
    calibration_score,
    crps_ensemble,
    rank_histogram,
    spread_skill_ratio,
)
from neural_transport.evaluation.pointwise import _to_numpy  # noqa: F401


def compute_xco2_column(field, pressure_weights, ak):
    """Compute column-averaged XCO2 from 3D field.

    XCO2 = sum(h_k * a_k * x_k) over levels axis.

    This is the simple NumPy/physical-space version. For the full normalized
    PyTorch forward model (with prior correction, targshift, obs normalization),
    see :class:`neural_transport.forward_model.XCO2ForwardModel`.

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
        return float(np.sqrt(np.mean(w * diff**2)))
    return float(np.sqrt(np.mean(diff**2)))


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
        return float(np.sqrt(np.mean(w * diff**2)))
    return float(np.sqrt(np.mean(diff**2)))


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


def gradient_at_boundary(field_2d, mask_2d):
    """Ratio of spatial gradient magnitude at mask boundary vs interior.

    A ratio >> 1 indicates sharp discontinuities at the observation mask edge
    (e.g., orbit-track stripe artifacts).

    Parameters
    ----------
    field_2d : np.ndarray, shape [nlat, nlon]
    mask_2d : np.ndarray, shape [nlat, nlon], bool

    Returns
    -------
    dict with 'gradient_ratio', 'gradient_boundary', 'gradient_interior'.
    Returns NaN if boundary or interior is too small.
    """
    # Compute gradient magnitude via finite differences
    grad_lat = np.diff(field_2d, axis=0)  # [nlat-1, nlon]
    grad_lon = np.diff(field_2d, axis=1)  # [nlat, nlon-1]
    # Pad to original size (repeat last row/col)
    grad_lat = np.concatenate([grad_lat, grad_lat[-1:, :]], axis=0)
    grad_lon = np.concatenate([grad_lon, grad_lon[:, -1:]], axis=1)
    grad_mag = np.sqrt(grad_lat**2 + grad_lon**2)

    # Boundary = mask edge (dilated mask XOR original mask)
    from scipy.ndimage import binary_dilation

    dilated = binary_dilation(mask_2d, iterations=1)
    boundary = dilated & ~mask_2d  # 1-pixel band outside the mask edge

    # Interior = not boundary and not within 2 pixels of boundary
    far_from_boundary = ~binary_dilation(boundary, iterations=2)
    interior = far_from_boundary

    n_boundary = boundary.sum()
    n_interior = interior.sum()

    if n_boundary < 5 or n_interior < 5:
        return {"gradient_ratio": np.nan, "gradient_boundary": np.nan, "gradient_interior": np.nan}

    g_boundary = float(np.mean(grad_mag[boundary]))
    g_interior = float(np.mean(grad_mag[interior]))
    ratio = g_boundary / max(g_interior, 1e-12)

    return {"gradient_ratio": ratio, "gradient_boundary": g_boundary, "gradient_interior": g_interior}


def xco2_obs_residual(ensemble_mean, gt, mask_2d, pressure_weights, ak):
    """Column XCO2 RMSE at observed locations.

    Computes XCO2 for both ensemble mean and GT, then measures RMSE
    only at observed spatial locations.

    Parameters
    ----------
    ensemble_mean : np.ndarray, shape [nlat, nlon, nlev]
    gt : np.ndarray, shape [nlat, nlon, nlev]
    mask_2d : np.ndarray, shape [nlat, nlon], bool
    pressure_weights : np.ndarray, shape [nlat, nlon, nlev] or [nlev]
    ak : np.ndarray, shape [nlat, nlon, nlev] or [nlev]

    Returns
    -------
    float — RMSE of column XCO2 at observed locations, or NaN.
    """
    if mask_2d is None or not mask_2d.any():
        return np.nan
    xco2_pred = compute_xco2_column(ensemble_mean, pressure_weights, ak)
    xco2_gt = compute_xco2_column(gt, pressure_weights, ak)
    diff = xco2_pred - xco2_gt
    return float(np.sqrt(np.mean(diff[mask_2d] ** 2)))


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
    gradient_ratio: float = np.nan
    xco2_obs_residual: float = np.nan

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
    mask_2d: np.ndarray | None = None  # [nlat, nlon] bool
    pressure_weights: np.ndarray | None = None  # [nlat, nlon, nlev]
    ak: np.ndarray | None = None  # [nlat, nlon, nlev]
    rank_hist: np.ndarray | None = None  # [n_samples + 1]
    calibration_data: dict | None = None
    crps_map: np.ndarray | None = None  # [nlat, nlon, (nlev)]
    lat: np.ndarray | None = None  # [nlat]
    lon: np.ndarray | None = None  # [nlon]


def compute_all_metrics(samples, gt, mask_2d=None, pressure_weights=None, ak=None, lat=None):
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
        val_rmse_col = float(np.sqrt(np.mean(cos_w_2d * diff_col**2)))
    else:
        val_rmse_col = float(np.sqrt(np.mean(diff_col**2)))

    # R2
    diff = ens_mean - gt
    if cos_w is not None:
        w3d = cos_w.reshape(-1, 1, 1)
        w3d = w3d / w3d.mean()
        ss_res = np.sum(w3d * diff**2)
        gt_mean = np.mean(w3d * gt) / np.mean(w3d)
        ss_tot = np.sum(w3d * (gt - gt_mean) ** 2)
    else:
        ss_res = np.sum(diff**2)
        gt_mean = np.mean(gt)
        ss_tot = np.sum((gt - gt_mean) ** 2)
    val_r2 = float(1 - ss_res / max(ss_tot, 1e-12))

    # XCO2 metrics
    val_rmse_xco2_full = np.nan
    val_rmse_xco2_obs = np.nan
    val_rmse_xco2_away = np.nan
    if pressure_weights is not None and ak is not None:
        val_rmse_xco2_full = rmse_xco2(ens_mean, gt, pressure_weights, ak, weights=cos_w)
        # XCO2 at obs / away
        if mask_2d is not None:
            xco2_pred = compute_xco2_column(ens_mean, pressure_weights, ak)
            xco2_gt = compute_xco2_column(gt, pressure_weights, ak)
            xco2_diff = xco2_pred - xco2_gt
            if mask_2d.any():
                val_rmse_xco2_obs = float(np.sqrt(np.mean(xco2_diff[mask_2d] ** 2)))
            if (~mask_2d).any():
                val_rmse_xco2_away = float(np.sqrt(np.mean(xco2_diff[~mask_2d] ** 2)))

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
