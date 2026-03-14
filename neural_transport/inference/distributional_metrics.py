"""Distributional metrics for comparing pools of CO2 anomaly fields.

Compare GT and generated CO2 fields as two empirical distributions.
Key insight: compare anomaly patterns (spatial mean removed), not absolute values.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from neural_transport.configs import MAX_N_DISTRIBUTIONAL


def remove_spatial_mean(fields):
    """Remove per-sample spatial mean.

    Args:
        fields: [N, nlat, nlon] or [N, nlat, nlon, nlev]

    Returns:
        Anomaly fields with per-sample spatial mean removed.
    """
    if fields.ndim == 3:
        mean = fields.mean(axis=(1, 2), keepdims=True)
    elif fields.ndim == 4:
        mean = fields.mean(axis=(1, 2), keepdims=True)  # mean over lat, lon; keep level
    else:
        raise ValueError(f"Expected 3D or 4D array, got {fields.ndim}D")
    return fields - mean


def _flatten_samples(fields):
    """Flatten spatial dims: [N, ...] -> [N, D]."""
    return fields.reshape(fields.shape[0], -1)


def energy_distance(samples_p, samples_q):
    """Two-sample energy distance.

    Args:
        samples_p: [N, ...] first sample pool
        samples_q: [M, ...] second sample pool

    Returns:
        Energy distance (float).
    """
    p = _flatten_samples(samples_p).astype(np.float64)
    q = _flatten_samples(samples_q).astype(np.float64)

    # E[||X-Y||] - 0.5*E[||X-X'||] - 0.5*E[||Y-Y'||]
    # Use subset for efficiency if large
    max_n = min(MAX_N_DISTRIBUTIONAL, len(p), len(q))
    if len(p) > max_n:
        idx = np.random.choice(len(p), max_n, replace=False)
        p = p[idx]
    if len(q) > max_n:
        idx = np.random.choice(len(q), max_n, replace=False)
        q = q[idx]

    d_pq = cdist(p, q, metric='euclidean').mean()
    d_pp = cdist(p, p, metric='euclidean').mean()
    d_qq = cdist(q, q, metric='euclidean').mean()

    return float(2 * d_pq - d_pp - d_qq)


def mmd_rbf(samples_p, samples_q, bandwidth="median"):
    """Maximum Mean Discrepancy with RBF kernel.

    Args:
        samples_p: [N, ...] first sample pool
        samples_q: [M, ...] second sample pool
        bandwidth: "median" for median heuristic, or float value

    Returns:
        MMD^2 value (float).
    """
    p = _flatten_samples(samples_p).astype(np.float64)
    q = _flatten_samples(samples_q).astype(np.float64)

    max_n = min(MAX_N_DISTRIBUTIONAL, len(p), len(q))
    if len(p) > max_n:
        p = p[np.random.choice(len(p), max_n, replace=False)]
    if len(q) > max_n:
        q = q[np.random.choice(len(q), max_n, replace=False)]

    all_samples = np.concatenate([p, q], axis=0)
    dists = cdist(all_samples, all_samples, metric='sqeuclidean')

    if bandwidth == "median":
        sigma2 = np.median(dists[dists > 0])
    else:
        sigma2 = float(bandwidth) ** 2

    K = np.exp(-dists / (2 * sigma2))

    n = len(p)
    K_pp = K[:n, :n]
    K_qq = K[n:, n:]
    K_pq = K[:n, n:]

    mmd2 = K_pp.mean() + K_qq.mean() - 2 * K_pq.mean()
    return float(mmd2)


def wasserstein_1d_marginals(samples_p, samples_q):
    """Per-level and per-lat-band 1D Wasserstein distances.

    Args:
        samples_p: [N, nlat, nlon, nlev] or [N, nlat, nlon]
        samples_q: [M, nlat, nlon, nlev] or [M, nlat, nlon]

    Returns:
        dict with per-level and per-lat-band Wasserstein distances.
    """
    result = {}

    if samples_p.ndim == 4:
        nlev = samples_p.shape[-1]
        for lev in range(nlev):
            p_flat = samples_p[:, :, :, lev].ravel()
            q_flat = samples_q[:, :, :, lev].ravel()
            result[f"wasserstein_level_{lev}"] = float(wasserstein_distance(p_flat, q_flat))
        result["wasserstein_all_levels_mean"] = np.mean([result[f"wasserstein_level_{lev}"] for lev in range(nlev)])
    else:
        p_flat = samples_p.ravel()
        q_flat = samples_q.ravel()
        result["wasserstein_all"] = float(wasserstein_distance(p_flat, q_flat))

    # Per lat-band (divide into 4 bands)
    nlat = samples_p.shape[1]
    n_bands = min(4, nlat)
    band_size = nlat // n_bands
    for i in range(n_bands):
        lat_start = i * band_size
        lat_end = (i + 1) * band_size if i < n_bands - 1 else nlat
        if samples_p.ndim == 4:
            p_band = samples_p[:, lat_start:lat_end, :, :].ravel()
            q_band = samples_q[:, lat_start:lat_end, :, :].ravel()
        else:
            p_band = samples_p[:, lat_start:lat_end, :].ravel()
            q_band = samples_q[:, lat_start:lat_end, :].ravel()
        result[f"wasserstein_latband_{i}"] = float(wasserstein_distance(p_band, q_band))

    return result


def power_spectrum_distance(fields_p, fields_q, lat, lon):
    """Power spectrum distance via 2D FFT and azimuthal average.

    Args:
        fields_p: [N, nlat, nlon] or [N, nlat, nlon, nlev]
        fields_q: [M, nlat, nlon] or [M, nlat, nlon, nlev]
        lat: 1D array of latitudes
        lon: 1D array of longitudes

    Returns:
        dict with log-spectral distance and per-level values.
    """

    def _compute_mean_power_spectrum(fields):
        """Compute mean power spectrum over samples."""
        if fields.ndim == 3:
            fields = fields[..., np.newaxis]

        nlev = fields.shape[-1]
        spectra = []
        for lev in range(nlev):
            level_spectra = []
            for i in range(len(fields)):
                field = fields[i, :, :, lev]
                # Apply latitude weighting
                cos_lat = np.cos(np.deg2rad(lat))[:, np.newaxis]
                field_weighted = field * np.sqrt(cos_lat)
                # 2D FFT
                fft2 = np.fft.fft2(field_weighted)
                power = np.abs(fft2) ** 2
                level_spectra.append(power)
            spectra.append(np.mean(level_spectra, axis=0))
        return spectra

    def _azimuthal_average(power_spectrum):
        """Compute azimuthal average of 2D power spectrum."""
        ny, nx = power_spectrum.shape
        cy, cx = ny // 2, nx // 2
        # Shift so DC is at center
        power_shifted = np.fft.fftshift(power_spectrum)
        y, x = np.ogrid[-cy : ny - cy, -cx : nx - cx]
        r = np.sqrt(x**2 + y**2).astype(int)
        max_r = min(cy, cx)
        radial_profile = np.zeros(max_r)
        for i in range(max_r):
            mask = r == i
            if mask.any():
                radial_profile[i] = power_shifted[mask].mean()
        return radial_profile

    spectra_p = _compute_mean_power_spectrum(fields_p)
    spectra_q = _compute_mean_power_spectrum(fields_q)

    result = {}
    log_spectral_dists = []
    for lev, (sp, sq) in enumerate(zip(spectra_p, spectra_q)):
        az_p = _azimuthal_average(sp)
        az_q = _azimuthal_average(sq)
        # Log spectral distance (avoid log(0))
        eps = 1e-12
        log_dist = np.sqrt(np.mean((np.log(az_p + eps) - np.log(az_q + eps)) ** 2))
        result[f"log_spectral_dist_level_{lev}"] = float(log_dist)
        log_spectral_dists.append(log_dist)

    result["log_spectral_dist_mean"] = float(np.mean(log_spectral_dists))
    return result


def zonal_mean_distance(fields_p, fields_q, lat):
    """Zonal mean profile RMSE: compare lat×level mean and std.

    Args:
        fields_p: [N, nlat, nlon, nlev] or [N, nlat, nlon]
        fields_q: [M, nlat, nlon, nlev] or [M, nlat, nlon]
        lat: 1D array of latitudes

    Returns:
        dict with zonal mean RMSE and zonal std RMSE.
    """
    # Zonal mean: average over longitude
    if fields_p.ndim == 3:
        fields_p = fields_p[..., np.newaxis]
        fields_q = fields_q[..., np.newaxis]

    # [N, nlat, nlev] after averaging over nlon
    zonal_p = fields_p.mean(axis=2)
    zonal_q = fields_q.mean(axis=2)

    # Mean zonal profile [nlat, nlev]
    mean_p = zonal_p.mean(axis=0)
    mean_q = zonal_q.mean(axis=0)

    # Std zonal profile [nlat, nlev]
    std_p = zonal_p.std(axis=0)
    std_q = zonal_q.std(axis=0)

    # Latitude weights
    cos_lat = np.cos(np.deg2rad(lat))
    weights = cos_lat / cos_lat.sum()
    weights = weights[:, np.newaxis]  # [nlat, 1]

    rmse_mean = float(np.sqrt(np.sum(weights * (mean_p - mean_q) ** 2)))
    rmse_std = float(np.sqrt(np.sum(weights * (std_p - std_q) ** 2)))

    return {
        "zonal_mean_rmse": rmse_mean,
        "zonal_std_rmse": rmse_std,
    }


def meridional_gradient_score(fields_p, fields_q, lat):
    """Compare N-S gradient distributions.

    Args:
        fields_p: [N, nlat, nlon, nlev] or [N, nlat, nlon]
        fields_q: [M, nlat, nlon, nlev] or [M, nlat, nlon]
        lat: 1D array of latitudes

    Returns:
        Wasserstein distance between meridional gradient distributions (float).
    """

    def _meridional_gradient(fields):
        """Compute meridional gradient (finite difference in lat)."""
        dlat = np.diff(lat)
        # Gradient: df/dlat ≈ (f[i+1] - f[i]) / (lat[i+1] - lat[i])
        grad = np.diff(fields, axis=1)
        # Normalize by dlat
        if fields.ndim == 3:
            dlat_broadcast = dlat[np.newaxis, :, np.newaxis]
        else:
            dlat_broadcast = dlat[np.newaxis, :, np.newaxis, np.newaxis]
        grad = grad / np.maximum(np.abs(dlat_broadcast), 1e-6)
        return grad

    grad_p = _meridional_gradient(fields_p).ravel()
    grad_q = _meridional_gradient(fields_q).ravel()

    # Subsample for efficiency
    max_n = 500000
    if len(grad_p) > max_n:
        grad_p = np.random.choice(grad_p, max_n, replace=False)
    if len(grad_q) > max_n:
        grad_q = np.random.choice(grad_q, max_n, replace=False)

    return float(wasserstein_distance(grad_p, grad_q))


def coverage_density(real, gen, k=5):
    """k-NN based coverage and density metrics (Naeem et al. 2020).

    Args:
        real: [N, ...] real samples
        gen: [M, ...] generated samples
        k: number of nearest neighbors

    Returns:
        dict with 'coverage' and 'density' values.
    """
    real_flat = _flatten_samples(real).astype(np.float64)
    gen_flat = _flatten_samples(gen).astype(np.float64)

    # Subsample if too large
    max_n = MAX_N_DISTRIBUTIONAL
    if len(real_flat) > max_n:
        real_flat = real_flat[np.random.choice(len(real_flat), max_n, replace=False)]
    if len(gen_flat) > max_n:
        gen_flat = gen_flat[np.random.choice(len(gen_flat), max_n, replace=False)]

    n_real = len(real_flat)

    # Pairwise distances
    d_rr = cdist(real_flat, real_flat, metric='euclidean')
    d_rg = cdist(real_flat, gen_flat, metric='euclidean')

    # k-th nearest neighbor distance for each real sample (excluding self)
    np.fill_diagonal(d_rr, np.inf)
    kth_real = np.sort(d_rr, axis=1)[:, min(k - 1, n_real - 2)]

    # Coverage: fraction of real samples with at least one generated neighbor within kth-NN radius

    # Simpler: fraction of real samples that have at least one gen neighbor closer than kth real neighbor
    covered = np.any(d_rg <= kth_real[:, np.newaxis], axis=1)
    coverage_val = float(np.mean(covered))

    # Density: average number of generated samples within kth-NN ball of each real sample, normalized
    density_counts = np.sum(d_rg <= kth_real[:, np.newaxis], axis=1)
    density_val = float(np.mean(density_counts) / k)

    return {
        "coverage": coverage_val,
        "density": density_val,
    }


def vendi_score(samples, kernel="rbf"):
    """Vendi diversity score via eigenvalue entropy of kernel matrix.

    Args:
        samples: [N, ...] sample pool
        kernel: "rbf" or "cosine"

    Returns:
        Vendi score (float). Higher = more diverse.
    """
    flat = _flatten_samples(samples).astype(np.float64)

    max_n = MAX_N_DISTRIBUTIONAL
    if len(flat) > max_n:
        flat = flat[np.random.choice(len(flat), max_n, replace=False)]

    n = len(flat)
    if n <= 1:
        return 1.0

    if kernel == "rbf":
        dists = cdist(flat, flat, metric='sqeuclidean')
        sigma2 = np.median(dists[dists > 0])
        if sigma2 <= 0:
            sigma2 = 1.0
        K = np.exp(-dists / (2 * sigma2))
    elif kernel == "cosine":
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        flat_normed = flat / norms
        K = flat_normed @ flat_normed.T
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Normalize kernel matrix
    K = K / n

    # Eigenvalues
    eigvals = np.linalg.eigvalsh(K)
    eigvals = eigvals[eigvals > 1e-12]

    # Entropy: exp(-sum(p * log(p))) where p = eigvals
    entropy = -np.sum(eigvals * np.log(eigvals))
    return float(np.exp(entropy))


def compute_distributional_metrics(gt_fields, gen_fields, lat, lon):
    """Compute all distributional metrics comparing GT and generated fields.

    Handles spatial mean removal internally.

    Args:
        gt_fields: [N, nlat, nlon] or [N, nlat, nlon, nlev] — GT CO2 fields
        gen_fields: [M, nlat, nlon] or [M, nlat, nlon, nlev] — generated CO2 fields
        lat: 1D array of latitudes
        lon: 1D array of longitudes

    Returns:
        dict with all metric values.
    """
    # Remove spatial mean to get anomaly patterns
    gt_anom = remove_spatial_mean(gt_fields)
    gen_anom = remove_spatial_mean(gen_fields)

    metrics = {}

    # Two-sample distances
    metrics["energy_distance"] = energy_distance(gt_anom, gen_anom)
    metrics["mmd_rbf"] = mmd_rbf(gt_anom, gen_anom)

    # Marginal distances
    w1 = wasserstein_1d_marginals(gt_anom, gen_anom)
    metrics.update(w1)

    # Spectral distance
    ps = power_spectrum_distance(gt_anom, gen_anom, lat, lon)
    metrics.update(ps)

    # Zonal mean
    zm = zonal_mean_distance(gt_anom, gen_anom, lat)
    metrics.update(zm)

    # Meridional gradient
    metrics["meridional_gradient_w1"] = meridional_gradient_score(gt_anom, gen_anom, lat)

    # Coverage and density
    cd = coverage_density(gt_anom, gen_anom)
    metrics.update(cd)

    # Diversity of generated samples
    metrics["vendi_score_gen"] = vendi_score(gen_anom)
    metrics["vendi_score_gt"] = vendi_score(gt_anom)

    return metrics
