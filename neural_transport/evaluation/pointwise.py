"""Pointwise metrics — NumPy and xarray implementations.

This module is the single source of truth for RMSE, MAE, bias, R2, NSE,
relative mean/std, and spatial error map/scalar computations.
"""

import numpy as np
import xskillscore


def _to_numpy(x) -> np.ndarray:
    """Convert torch.Tensor or xr.DataArray to numpy ndarray."""
    if hasattr(x, "values"):  # xarray
        return x.values
    if hasattr(x, "detach"):  # torch
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# NumPy implementations
# ---------------------------------------------------------------------------


def rmse_np(pred, targ, weights=None):
    """Root mean squared error (NumPy).

    Parameters
    ----------
    pred, targ : array-like
    weights : array-like, optional. Must be broadcastable to pred/targ shape.

    Returns
    -------
    float
    """
    pred, targ = np.asarray(pred, dtype=np.float64), np.asarray(targ, dtype=np.float64)
    diff2 = (pred - targ) ** 2
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.mean()
        return float(np.sqrt(np.mean(w * diff2)))
    return float(np.sqrt(np.mean(diff2)))


def mae_np(pred, targ, weights=None):
    """Mean absolute error (NumPy).

    Parameters
    ----------
    pred, targ : array-like
    weights : array-like, optional.

    Returns
    -------
    float
    """
    pred, targ = np.asarray(pred, dtype=np.float64), np.asarray(targ, dtype=np.float64)
    absdiff = np.abs(pred - targ)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.mean()
        return float(np.mean(w * absdiff))
    return float(np.mean(absdiff))


def bias_np(pred, targ, weights=None):
    """Mean bias (pred - targ) (NumPy).

    Parameters
    ----------
    pred, targ : array-like
    weights : array-like, optional.

    Returns
    -------
    float
    """
    pred, targ = np.asarray(pred, dtype=np.float64), np.asarray(targ, dtype=np.float64)
    diff = pred - targ
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.mean()
        return float(np.mean(w * diff))
    return float(np.mean(diff))


def r2_np(pred, targ, weights=None):
    """Coefficient of determination R^2 = 1 - SS_res / SS_tot (NumPy).

    Parameters
    ----------
    pred, targ : array-like
    weights : array-like, optional.

    Returns
    -------
    float
    """
    pred, targ = np.asarray(pred, dtype=np.float64), np.asarray(targ, dtype=np.float64)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.mean()
        ss_res = np.sum(w * (pred - targ) ** 2)
        targ_mean = np.sum(w * targ) / np.sum(w)
        ss_tot = np.sum(w * (targ - targ_mean) ** 2)
    else:
        ss_res = np.sum((pred - targ) ** 2)
        targ_mean = np.mean(targ)
        ss_tot = np.sum((targ - targ_mean) ** 2)
    return float(1 - ss_res / max(ss_tot, 1e-12))


def nse_np(pred, targ, weights=None):
    """Nash-Sutcliffe Efficiency (same formula as R^2) (NumPy).

    Parameters
    ----------
    pred, targ : array-like
    weights : array-like, optional.

    Returns
    -------
    float
    """
    return r2_np(pred, targ, weights=weights)


def rel_mean_np(pred, targ, weights=None):
    """Ratio of weighted means: mean(pred) / mean(targ) (NumPy).

    Parameters
    ----------
    pred, targ : array-like
    weights : array-like, optional.

    Returns
    -------
    float
    """
    pred, targ = np.asarray(pred, dtype=np.float64), np.asarray(targ, dtype=np.float64)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.mean()
        return float(np.mean(w * pred) / np.mean(w * targ))
    return float(np.mean(pred) / np.mean(targ))


def rel_std_np(pred, targ, weights=None):
    """Ratio of weighted stds: std(pred) / std(targ) (NumPy).

    Parameters
    ----------
    pred, targ : array-like
    weights : array-like, optional.

    Returns
    -------
    float
    """
    pred, targ = np.asarray(pred, dtype=np.float64), np.asarray(targ, dtype=np.float64)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.mean()
        pred_std = float(np.sqrt(np.mean(w * (pred - np.mean(w * pred) / np.mean(w)) ** 2)))
        targ_std = float(np.sqrt(np.mean(w * (targ - np.mean(w * targ) / np.mean(w)) ** 2)))
    else:
        pred_std = float(np.std(pred))
        targ_std = float(np.std(targ))
    return pred_std / max(targ_std, 1e-12)


METRICS_NP = dict(
    rmse=rmse_np,
    mae=mae_np,
    bias=bias_np,
    r2=r2_np,
    nse=nse_np,
    rel_mean=rel_mean_np,
    rel_std=rel_std_np,
)

# ---------------------------------------------------------------------------
# xarray implementations
# ---------------------------------------------------------------------------


def rmse_xr(pred, targ, weights, dims=["lat", "lon"]):
    """RMSE (xarray)."""
    return ((pred - targ) ** 2 * weights).mean(dims) ** 0.5


def mae_xr(pred, targ, weights, dims=["lat", "lon"]):
    """MAE (xarray)."""
    import numpy as _np

    return (_np.abs(pred - targ) * weights).mean(dims)


def bias_xr(pred, targ, weights, dims=["lat", "lon"]):
    """Bias (xarray)."""
    return (pred * weights).mean(dims) - (targ * weights).mean(dims)


def r2_xr(pred, targ, weights, dims=["lat", "lon"]):
    """R^2 via xskillscore (xarray)."""
    return (
        xskillscore.pearson_r(
            pred,
            targ,
            dim=dims,
            weights=weights.isel(**{d: 0 for d in weights.dims if d not in dims}),
        )
        ** 2
    )


def nse_xr(pred, targ, weights, dims=["lat", "lon"]):
    """NSE via xskillscore (xarray)."""
    return xskillscore.r2(
        pred,
        targ,
        dim=dims,
        weights=weights.isel(**{d: 0 for d in weights.dims if d not in dims}),
    )


def rel_mean_xr(pred, targ, weights, dims=["lat", "lon"]):
    """Relative mean (xarray)."""
    return (pred * weights).mean(dims) / (targ * weights).mean(dims)


def rel_std_xr(pred, targ, weights, dims=["lat", "lon"]):
    """Relative std (xarray)."""
    return (pred * weights).std(dims) / (targ * weights).std(dims)


METRICS_XR = dict(
    rmse=rmse_xr,
    mae=mae_xr,
    bias=bias_xr,
    r2=r2_xr,
    nse=nse_xr,
    rel_mean=rel_mean_xr,
    rel_std=rel_std_xr,
)


# ---------------------------------------------------------------------------
# Error maps & scalars (migrated from tools/metrics.py)
# ---------------------------------------------------------------------------


def compute_error_maps(gen_samples, gt):
    """Compute spatial bias, RMSE, and ensemble spread maps.

    Parameters
    ----------
    gen_samples : xr.DataArray, shape [n_samples, lat, lon]
        Generated ensemble samples.
    gt : xr.DataArray, shape [lat, lon]
        Ground truth field.

    Returns
    -------
    bias_map, rmse_map, mean_map, spread_map : np.ndarray, shape [lat, lon]
    """
    gen_samples = _to_numpy(gen_samples)
    gt = _to_numpy(gt)

    mean_map = np.mean(gen_samples, axis=0)

    mask = np.isfinite(gt) & np.isfinite(mean_map)
    gt_masked = np.where(mask, gt, np.nan)
    gen_samples_masked = np.where(mask, gen_samples, np.nan)

    bias_map = np.nanmean(gen_samples_masked, axis=0) - gt_masked
    rmse_map = np.sqrt(np.nanmean((gen_samples_masked - gt_masked) ** 2, axis=0))
    spread_map = np.nanstd(gen_samples_masked, axis=0)
    mean_map = np.where(mask, mean_map, np.nan)

    return bias_map, rmse_map, mean_map, spread_map


def compute_error_scalars(bias_map, rmse_map, mean_map, spread_map, weights=None):
    """Compute scalar error metrics from spatial maps.

    Parameters
    ----------
    bias_map, rmse_map, mean_map, spread_map : np.ndarray, shape [lat, lon]
        Spatial error maps.
    weights : np.ndarray, shape [lat, lon], optional

    Returns
    -------
    bias_scalar, rmse_scalar, mean_scalar, spread_scalar : float
        Mean values of the error metrics.
    """
    if weights is not None:
        weights = weights / np.sum(weights)
        bias_scalar = np.sum(bias_map * weights)
        rmse_scalar = np.sum(rmse_map * weights)
        mean_scalar = np.sum(mean_map * weights)
        spread_scalar = np.sum(spread_map * weights)
    else:
        bias_scalar = np.mean(bias_map)
        rmse_scalar = np.mean(rmse_map)
        mean_scalar = np.mean(mean_map)
        spread_scalar = np.mean(spread_map)

    return bias_scalar, rmse_scalar, mean_scalar, spread_scalar
