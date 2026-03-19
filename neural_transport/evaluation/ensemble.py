"""Ensemble metrics — CRPS, spread-skill ratio, calibration, rank histogram.

Migrated from ``tools/metrics.py`` (crps) and ``inference/metrics.py``
(crps_ensemble, spread_skill_ratio, calibration_score, rank_histogram).
"""

import numpy as np
import torch
import xarray as xr


def crps(preds, tests) -> tuple[xr.DataArray | np.ndarray, float]:
    """Compute the Continuous Ranked Probability Score (CRPS)
    for ensemble predictions vs. ground truth.

    Parameters
    ----------
    preds : xarray.DataArray or np.ndarray
        Ensemble predictions. Expected shape:
          - with levels: (sample, lat, lon, level)
          - without levels: (sample, lat, lon)
    tests : xarray.DataArray, torch.Tensor, or np.ndarray
        Ground truth / test samples. Shape:
          - with levels: (level, lat, lon)
          - without levels: (lat, lon)

    Returns
    -------
    crps_map : xarray.DataArray or np.ndarray
        CRPS per gridpoint (lat x lon) or (lat x lon x level) if levels present.
    crps_mean : float
        Mean CRPS averaged over all gridpoints.
    """
    if isinstance(preds, xr.DataArray):
        preds = preds.values
    if isinstance(tests, xr.DataArray):
        tests = tests.values
    if torch.is_tensor(tests):
        tests = tests.cpu().numpy()

    if tests.ndim == 3:
        tests = np.moveaxis(tests, 0, -1)  # [lat, lon, level]
        lat_t, lon_t, C_t = tests.shape
        tests_flat = tests.reshape(-1, C_t)  # [N, C]
    elif tests.ndim == 2:
        lat_t, lon_t = tests.shape
        C_t = None
        tests_flat = tests.reshape(-1)  # [N,]

    if preds.ndim == 4:
        S, lat, lon, C = preds.shape
        assert C == C_t, f"Level mismatch: {C} vs {C_t}"
        assert lat == lat_t and lon == lon_t, f"Spatial mismatch: {lat}x{lon} vs {lat_t}x{lon_t}"
        preds_flat = preds.reshape(S, -1, C)  # [samples, N, C]

    elif preds.ndim == 3:
        S, lat, lon = preds.shape
        preds_flat = preds.reshape(S, lat * lon)  # [samples, N]

    # Compute CRPS across ensemble dimension
    # term1 = mean(|x_i - obs|)
    term1 = np.mean(np.abs(preds_flat - tests_flat[None, ...]), axis=0)  # [N, (C)]
    # term2 = 0.5 * mean(|x_i - x_j|)
    diffs = np.abs(preds_flat[:, None, ...] - preds_flat[None, :, ...])  # [2xsamples, N, (C)]
    term2 = 0.5 * np.mean(diffs, axis=(0, 1))  # [N, (C)]

    crps_flat = term1 - term2  # [N, (C)]
    crps_map = crps_flat.reshape(lat, lon, C_t) if preds.ndim == 4 else crps_flat.reshape(lat, lon)
    crps_mean = float(np.mean(crps_map, axis=(0, 1)))
    return crps_map, crps_mean


def crps_ensemble(samples, gt):
    """Thin wrapper around :func:`crps` for 3D fields.

    For 3D fields, computes CRPS per-level to avoid axis reduction issues.

    Parameters
    ----------
    samples : np.ndarray, shape [n_samples, nlat, nlon, (nlev)]
    gt : np.ndarray, shape [nlat, nlon, (nlev)]

    Returns
    -------
    crps_map : np.ndarray, shape [nlat, nlon, (nlev)]
    crps_mean : float
    """
    if gt.ndim == 3:
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
    observed = np.array([float(np.mean(gt < percentiles[i])) for i in range(len(quantiles))])

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
    np.ndarray, shape [n_samples + 1]. Histogram counts (fractions).
    """
    n_samples = samples.shape[0]
    ranks = np.sum(samples < gt[None, ...], axis=0).ravel()
    histogram = np.bincount(ranks, minlength=n_samples + 1).astype(float)
    histogram = histogram / histogram.sum()
    return histogram
