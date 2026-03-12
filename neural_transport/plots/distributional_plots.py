"""Distributional comparison plots for CO2 anomaly fields.

All plots compare anomaly fields (spatial mean removed).
Follow existing project plotting patterns.
"""

from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from neural_transport.inference.distributional_metrics import remove_spatial_mean
from neural_transport.plots.utilities.plot_utils import mpl_rc_params, save_figure

# --- Pressure-weighted column helper ---


def _pressure_weighted_column(fields, level_values):
    """Compute pressure-weighted column average (XCO2).

    Args:
        fields: [N, nlat, nlon, nlev] mixing ratio fields
        level_values: 1D array of pressure midpoints in hPa (e.g. [1013, ..., 73])

    Returns:
        [N, nlat, nlon] pressure-weighted column mean
    """
    if level_values is None:
        return fields.mean(axis=-1)

    level_values = np.asarray(level_values, dtype=np.float64)
    nlev = len(level_values)

    # Compute layer boundaries as midpoints between adjacent levels,
    # with surface extended and TOA = 0
    boundaries = np.zeros(nlev + 1)
    boundaries[0] = level_values[0] + (level_values[0] - level_values[1]) / 2  # surface
    boundaries[-1] = 0.0  # TOA
    for k in range(1, nlev):
        boundaries[k] = (level_values[k - 1] + level_values[k]) / 2

    dp = np.abs(np.diff(boundaries))  # [nlev]

    # Weighted mean: sum(dp_k * x_k) / sum(dp_k)
    dp_broadcast = dp[np.newaxis, np.newaxis, np.newaxis, :]  # [1,1,1,nlev]
    return (fields * dp_broadcast).sum(axis=-1) / dp.sum()


def _level_label_and_suffix(level_idx, level_hpa):
    """Return (title_label, filename_suffix) for a given level."""
    if level_idx is None:
        return "total column (XCO2)", "_total_column"
    if level_hpa is not None:
        return f"{level_hpa} hPa", f"_{level_hpa}hPa"
    return f"level {level_idx}", f"_lev{level_idx}"


def plot_marginal_distributions(
    gt_fields,
    gen_fields,
    out_dir,
    level_idx=0,
    level_hpa=None,
    level_values=None,
    imgformats=("pdf", "png"),
):
    """Plot histograms of raw values and per-sample statistics.

    Args:
        gt_fields: [N, nlat, nlon, nlev] GT anomaly fields
        gen_fields: [M, nlat, nlon, nlev] generated anomaly fields
        out_dir: output directory
        level_idx: which level to show (or None for total column)
        level_hpa: pressure value in hPa for labeling
        level_values: full array of level pressures for column weighting
        imgformats: output image formats
    """
    plt.rcParams.update(mpl_rc_params)
    out_dir = Path(out_dir)

    gt_anom = remove_spatial_mean(gt_fields)
    gen_anom = remove_spatial_mean(gen_fields)

    if gt_anom.ndim == 4 and level_idx is not None:
        gt_slice = gt_anom[:, :, :, level_idx]
        gen_slice = gen_anom[:, :, :, level_idx]
    elif gt_anom.ndim == 4:
        gt_slice = _pressure_weighted_column(gt_anom, level_values)
        gen_slice = _pressure_weighted_column(gen_anom, level_values)
    else:
        gt_slice = gt_anom
        gen_slice = gen_anom

    level_label, file_suffix = _level_label_and_suffix(level_idx, level_hpa)

    # Raw value histograms + per-sample statistics
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

    # Panel 0: Raw anomaly value distribution
    gt_flat = gt_slice.ravel()
    gen_flat = gen_slice.ravel()
    max_pts = 500_000
    if len(gt_flat) > max_pts:
        gt_flat = np.random.choice(gt_flat, max_pts, replace=False)
    if len(gen_flat) > max_pts:
        gen_flat = np.random.choice(gen_flat, max_pts, replace=False)
    bins = np.linspace(
        min(gt_flat.min(), gen_flat.min()),
        max(gt_flat.max(), gen_flat.max()),
        60,
    )
    axes[0].hist(gt_flat, bins=bins, alpha=0.6, label="GT", density=True, color="C0")
    axes[0].hist(gen_flat, bins=bins, alpha=0.6, label="Gen", density=True, color="C1")
    axes[0].set_xlabel("Anomaly Value")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Panels 1-3: Per-sample statistics
    stats = {}
    for name, func in [
        ("Spatial Std", lambda x: x.std(axis=(1, 2))),
        ("Spatial Skewness", lambda x: _skewness(x)),
        ("Spatial Kurtosis", lambda x: _kurtosis(x)),
    ]:
        stats[name] = (func(gt_slice), func(gen_slice))

    for ax, (name, (gt_vals, gen_vals)) in zip(axes[1:], stats.items()):
        bins = np.linspace(
            min(gt_vals.min(), gen_vals.min()),
            max(gt_vals.max(), gen_vals.max()),
            30,
        )
        ax.hist(gt_vals, bins=bins, alpha=0.6, label="GT", density=True, color="C0")
        ax.hist(gen_vals, bins=bins, alpha=0.6, label="Gen", density=True, color="C1")
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle(f"Marginal distributions ({level_label})")
    fig.tight_layout()
    save_figure(fig, out_dir, f"marginal_distributions{file_suffix}", imgformats=imgformats)


def plot_spatial_pattern_comparison(
    gt_fields,
    gen_fields,
    lat,
    lon,
    out_dir,
    level_idx=0,
    level_hpa=None,
    level_values=None,
    imgformats=("pdf", "png"),
):
    """Plot maps: GT mean anomaly | Gen mean anomaly | Diff; GT std | Gen std | Diff.

    Args:
        gt_fields: [N, nlat, nlon, nlev] GT fields
        gen_fields: [M, nlat, nlon, nlev] generated fields
        lat, lon: coordinate arrays
        out_dir: output directory
        level_idx: which level to plot (or None for total column)
        level_hpa: pressure value in hPa for labeling
        level_values: full array of level pressures for column weighting
        imgformats: output formats
    """
    plt.rcParams.update(mpl_rc_params)
    out_dir = Path(out_dir)

    gt_anom = remove_spatial_mean(gt_fields)
    gen_anom = remove_spatial_mean(gen_fields)

    if gt_anom.ndim == 4 and level_idx is not None:
        gt_slice = gt_anom[:, :, :, level_idx]
        gen_slice = gen_anom[:, :, :, level_idx]
    elif gt_anom.ndim == 4:
        gt_slice = _pressure_weighted_column(gt_anom, level_values)
        gen_slice = _pressure_weighted_column(gen_anom, level_values)
    else:
        gt_slice = gt_anom
        gen_slice = gen_anom

    level_label, file_suffix = _level_label_and_suffix(level_idx, level_hpa)

    gt_mean = gt_slice.mean(axis=0)
    gen_mean = gen_slice.mean(axis=0)
    gt_std = gt_slice.std(axis=0)
    gen_std = gen_slice.std(axis=0)

    projection = ccrs.Robinson()
    transform = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 7),
        subplot_kw={"projection": projection},
    )

    # Row 1: Mean anomaly
    titles_row1 = ["GT Mean Anomaly", "Gen Mean Anomaly", "Difference (Gen - GT)"]
    data_row1 = [gt_mean, gen_mean, gen_mean - gt_mean]
    vmax_mean = max(np.abs(gt_mean).max(), np.abs(gen_mean).max())

    for ax, title, data in zip(axes[0], titles_row1, data_row1):
        if "Diff" in title:
            vmax = np.abs(data).max()
            im = ax.pcolormesh(
                lon,
                lat,
                data,
                transform=transform,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
        else:
            im = ax.pcolormesh(
                lon,
                lat,
                data,
                transform=transform,
                cmap="RdBu_r",
                vmin=-vmax_mean,
                vmax=vmax_mean,
            )
        ax.coastlines(linewidth=0.5)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)

    # Row 2: Std
    titles_row2 = ["GT Std", "Gen Std", "Difference (Gen - GT)"]
    data_row2 = [gt_std, gen_std, gen_std - gt_std]
    vmax_std = max(gt_std.max(), gen_std.max())

    for ax, title, data in zip(axes[1], titles_row2, data_row2):
        if "Diff" in title:
            vmax = np.abs(data).max()
            im = ax.pcolormesh(
                lon,
                lat,
                data,
                transform=transform,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
        else:
            im = ax.pcolormesh(
                lon,
                lat,
                data,
                transform=transform,
                cmap="viridis",
                vmin=0,
                vmax=vmax_std,
            )
        ax.coastlines(linewidth=0.5)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)

    fig.suptitle(f"Spatial Pattern Comparison ({level_label})")
    fig.tight_layout()
    save_figure(fig, out_dir, f"spatial_pattern_comparison{file_suffix}", imgformats=imgformats)


def plot_lat_height_comparison(
    gt_fields,
    gen_fields,
    lat,
    out_dir,
    level_values=None,
    imgformats=("pdf", "png"),
):
    """Plot zonal mean cross-sections (lat x level): mean anomaly & std comparison.

    Args:
        gt_fields: [N, nlat, nlon, nlev]
        gen_fields: [M, nlat, nlon, nlev]
        lat: latitude array
        out_dir: output directory
        level_values: array of pressure values in hPa for y-axis labels
        imgformats: output formats
    """
    plt.rcParams.update(mpl_rc_params)
    out_dir = Path(out_dir)

    gt_anom = remove_spatial_mean(gt_fields)
    gen_anom = remove_spatial_mean(gen_fields)

    if gt_anom.ndim == 3:
        print("Skipping lat-height plot: no level dimension")
        return

    nlev = gt_anom.shape[-1]
    if level_values is not None:
        levels = np.asarray(level_values)
    else:
        levels = np.arange(nlev)

    # Zonal mean: [N, nlat, nlev]
    gt_zonal = gt_anom.mean(axis=2)
    gen_zonal = gen_anom.mean(axis=2)

    gt_mean = gt_zonal.mean(axis=0)  # [nlat, nlev]
    gen_mean = gen_zonal.mean(axis=0)
    gt_std = gt_zonal.std(axis=0)
    gen_std = gen_zonal.std(axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Row 1: Mean zonal anomaly
    vmax = max(np.abs(gt_mean).max(), np.abs(gen_mean).max())
    for ax, data, title in zip(
        axes[0],
        [gt_mean, gen_mean, gen_mean - gt_mean],
        ["GT Mean", "Gen Mean", "Difference"],
    ):
        if "Diff" in title:
            vmax_d = np.abs(data).max()
            im = ax.pcolormesh(lat, levels, data.T, cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d)
        else:
            im = ax.pcolormesh(lat, levels, data.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Pressure (hPa)" if level_values is not None else "Level")
        ax.set_title(title)
        if level_values is not None:
            ax.invert_yaxis()
        else:
            ax.invert_yaxis()
        plt.colorbar(im, ax=ax)

    # Row 2: Std zonal
    vmax_s = max(gt_std.max(), gen_std.max())
    for ax, data, title in zip(
        axes[1],
        [gt_std, gen_std, gen_std - gt_std],
        ["GT Std", "Gen Std", "Difference"],
    ):
        if "Diff" in title:
            vmax_d = np.abs(data).max()
            im = ax.pcolormesh(lat, levels, data.T, cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d)
        else:
            im = ax.pcolormesh(lat, levels, data.T, cmap="viridis", vmin=0, vmax=vmax_s)
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Pressure (hPa)" if level_values is not None else "Level")
        ax.set_title(title)
        if level_values is not None:
            ax.invert_yaxis()
        else:
            ax.invert_yaxis()
        plt.colorbar(im, ax=ax)

    fig.suptitle("Zonal Mean Cross-Section Comparison")
    fig.tight_layout()
    save_figure(fig, out_dir, "lat_height_comparison", imgformats=imgformats)


def plot_power_spectrum_comparison(
    gt_fields,
    gen_fields,
    lat,
    lon,
    out_dir,
    level_idx=0,
    level_hpa=None,
    level_values=None,
    imgformats=("pdf", "png"),
):
    """Plot log-log power spectra with uncertainty bands.

    Args:
        gt_fields: [N, nlat, nlon, nlev] or [N, nlat, nlon]
        gen_fields: [M, nlat, nlon, nlev] or [M, nlat, nlon]
        lat, lon: coordinate arrays
        out_dir: output directory
        level_idx: which level (for 4D), or None for total column
        level_hpa: pressure value in hPa for labeling
        level_values: full array of level pressures for column weighting
        imgformats: output formats
    """
    plt.rcParams.update(mpl_rc_params)
    out_dir = Path(out_dir)

    gt_anom = remove_spatial_mean(gt_fields)
    gen_anom = remove_spatial_mean(gen_fields)

    if gt_anom.ndim == 4 and level_idx is not None:
        gt_slice = gt_anom[:, :, :, level_idx]
        gen_slice = gen_anom[:, :, :, level_idx]
    elif gt_anom.ndim == 4:
        gt_slice = _pressure_weighted_column(gt_anom, level_values)
        gen_slice = _pressure_weighted_column(gen_anom, level_values)
    else:
        gt_slice = gt_anom
        gen_slice = gen_anom

    level_label, file_suffix = _level_label_and_suffix(level_idx, level_hpa)

    def _compute_spectra(fields):
        """Compute per-sample azimuthal power spectra."""
        cos_lat = np.cos(np.deg2rad(lat))[:, np.newaxis]
        spectra = []
        for i in range(len(fields)):
            field = fields[i] * np.sqrt(cos_lat)
            fft2 = np.fft.fft2(field)
            power = np.abs(fft2) ** 2
            power_shifted = np.fft.fftshift(power)
            ny, nx = power.shape
            cy, cx = ny // 2, nx // 2
            y, x = np.ogrid[-cy : ny - cy, -cx : nx - cx]
            r = np.sqrt(x**2 + y**2).astype(int)
            max_r = min(cy, cx)
            radial = np.zeros(max_r)
            for ri in range(max_r):
                mask = r == ri
                if mask.any():
                    radial[ri] = power_shifted[mask].mean()
            spectra.append(radial)
        return np.array(spectra)

    gt_spectra = _compute_spectra(gt_slice)
    gen_spectra = _compute_spectra(gen_slice)

    wavenumbers = np.arange(1, gt_spectra.shape[1])
    gt_mean = gt_spectra[:, 1:].mean(axis=0)
    gt_std = gt_spectra[:, 1:].std(axis=0)
    gen_mean = gen_spectra[:, 1:].mean(axis=0)
    gen_std = gen_spectra[:, 1:].std(axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(wavenumbers, gt_mean, label="GT", color="C0")
    ax.fill_between(wavenumbers, gt_mean - gt_std, gt_mean + gt_std, alpha=0.2, color="C0")
    ax.loglog(wavenumbers, gen_mean, label="Gen", color="C1")
    ax.fill_between(wavenumbers, gen_mean - gen_std, gen_mean + gen_std, alpha=0.2, color="C1")
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Power")
    ax.set_title(f"Power Spectrum ({level_label})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, out_dir, f"power_spectrum{file_suffix}", imgformats=imgformats)


def plot_distributional_metrics_summary(
    metrics,
    out_dir,
    level_values=None,
    imgformats=("pdf", "png"),
):
    """Plot summary bar chart of distributional metrics.

    Args:
        metrics: dict of metric name -> value
        out_dir: output directory
        level_values: array of pressure values for replacing level indices in names
        imgformats: output formats
    """
    plt.rcParams.update(mpl_rc_params)
    out_dir = Path(out_dir)

    # Select key scalar metrics
    key_metrics = {
        k: v
        for k, v in metrics.items()
        if isinstance(v, int | float)
        and not k.startswith("wasserstein_level")
        and not k.startswith("log_spectral_dist_level")
    }

    if not key_metrics:
        return

    # Replace level indices with hPa labels
    if level_values is not None:
        renamed = {}
        for k, v in key_metrics.items():
            new_k = k
            for i, hpa in enumerate(level_values):
                new_k = new_k.replace(f"_level_{i}", f"_{int(hpa)}hPa")
            renamed[new_k] = v
        key_metrics = renamed

    fig, ax = plt.subplots(figsize=(10, max(4, len(key_metrics) * 0.3)))
    names = list(key_metrics.keys())
    values = list(key_metrics.values())

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color="C0", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Value")
    ax.set_title("Distributional Metrics Summary")
    ax.invert_yaxis()

    fig.tight_layout()
    save_figure(fig, out_dir, "distributional_metrics_summary", imgformats=imgformats)


def plot_tuning_comparison(results_dict, out_dir, imgformats=("pdf", "png")):
    """Multi-config comparison bar plots + tables.

    Args:
        results_dict: dict of config_name -> metrics_dict
        out_dir: output directory
        imgformats: output formats
    """
    plt.rcParams.update(mpl_rc_params)
    out_dir = Path(out_dir)

    if not results_dict:
        return

    # Select key metrics to compare
    key_metric_names = [
        "energy_distance",
        "mmd_rbf",
        "zonal_mean_rmse",
        "zonal_std_rmse",
        "log_spectral_dist_mean",
        "meridional_gradient_w1",
        "coverage",
        "density",
    ]

    config_names = list(results_dict.keys())
    available_metrics = [m for m in key_metric_names if all(m in results_dict[c] for c in config_names)]

    if not available_metrics:
        return

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, available_metrics):
        values = [results_dict[c][metric_name] for c in config_names]
        x = np.arange(len(config_names))
        ax.bar(x, values, color="C0", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(config_names, rotation=45, ha="right", fontsize=6)
        ax.set_title(metric_name.replace("_", " "), fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Tuning Comparison")
    fig.tight_layout()
    save_figure(fig, out_dir, "tuning_comparison", imgformats=imgformats)


def plot_qq(
    gt_fields,
    gen_fields,
    out_dir,
    level_idx=0,
    level_hpa=None,
    level_values=None,
    imgformats=("pdf", "png"),
):
    """Q-Q plots comparing GT and generated distributions.

    Args:
        gt_fields: [N, nlat, nlon, nlev] or [N, nlat, nlon]
        gen_fields: [M, nlat, nlon, nlev] or [M, nlat, nlon]
        out_dir: output directory
        level_idx: which level (for 4D), or None for total column
        level_hpa: pressure value in hPa for labeling
        level_values: full array of level pressures for column weighting
        imgformats: output formats
    """
    plt.rcParams.update(mpl_rc_params)
    out_dir = Path(out_dir)

    gt_anom = remove_spatial_mean(gt_fields)
    gen_anom = remove_spatial_mean(gen_fields)

    if gt_anom.ndim == 4 and level_idx is not None:
        gt_vals = np.sort(gt_anom[:, :, :, level_idx].ravel())
        gen_vals = np.sort(gen_anom[:, :, :, level_idx].ravel())
    elif gt_anom.ndim == 4:
        gt_col = _pressure_weighted_column(gt_anom, level_values)
        gen_col = _pressure_weighted_column(gen_anom, level_values)
        gt_vals = np.sort(gt_col.ravel())
        gen_vals = np.sort(gen_col.ravel())
    else:
        gt_vals = np.sort(gt_anom.ravel())
        gen_vals = np.sort(gen_anom.ravel())

    level_label, file_suffix = _level_label_and_suffix(level_idx, level_hpa)

    n_quantiles = 200
    quantiles = np.linspace(0, 1, n_quantiles)
    gt_q = np.quantile(gt_vals, quantiles)
    gen_q = np.quantile(gen_vals, quantiles)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(gt_q, gen_q, s=5, alpha=0.5)
    lims = [min(gt_q.min(), gen_q.min()), max(gt_q.max(), gen_q.max())]
    ax.plot(lims, lims, 'k--', lw=0.5)
    ax.set_xlabel("GT quantiles")
    ax.set_ylabel("Gen quantiles")
    ax.set_title(f"Q-Q Plot ({level_label})")
    ax.set_aspect("equal")

    fig.tight_layout()
    save_figure(fig, out_dir, f"qq_plot{file_suffix}", imgformats=imgformats)


def plot_sample_grid(
    gt_fields,
    gen_fields,
    lat,
    lon,
    out_dir,
    level_values=None,
    n_samples=4,
    imgformats=("pdf", "png"),
):
    """Plot grid of sample total-column CO2 maps: GT vs Generated.

    Args:
        gt_fields: [N, nlat, nlon, nlev] GT fields
        gen_fields: [M, nlat, nlon, nlev] generated fields
        lat, lon: coordinate arrays
        out_dir: output directory
        level_values: pressure midpoints for column weighting
        n_samples: number of sample rows
        imgformats: output formats
    """
    plt.rcParams.update(mpl_rc_params)
    out_dir = Path(out_dir)

    # Compute total column
    if gt_fields.ndim == 4:
        gt_col = _pressure_weighted_column(gt_fields, level_values)
        gen_col = _pressure_weighted_column(gen_fields, level_values)
    else:
        gt_col = gt_fields
        gen_col = gen_fields

    n_gt = min(n_samples, len(gt_col))
    n_gen = min(n_samples, len(gen_col))
    n_rows = max(n_gt, n_gen)

    projection = ccrs.Robinson()
    transform = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(10, 3 * n_rows),
        subplot_kw={"projection": projection},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    vmin = min(gt_col[:n_gt].min(), gen_col[:n_gen].min())
    vmax = max(gt_col[:n_gt].max(), gen_col[:n_gen].max())

    for i in range(n_rows):
        for j, (col_data, n_avail, label) in enumerate(
            [
                (gt_col, n_gt, "GT"),
                (gen_col, n_gen, "Gen"),
            ]
        ):
            ax = axes[i, j]
            if i < n_avail:
                im = ax.pcolormesh(
                    lon,
                    lat,
                    col_data[i],
                    transform=transform,
                    cmap="RdBu_r",
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.coastlines(linewidth=0.5)
                plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
            else:
                ax.set_visible(False)
            if i == 0:
                ax.set_title(label)

    fig.suptitle("Sample Total Column CO2 (XCO2)")
    fig.tight_layout()
    save_figure(fig, out_dir, "sample_grid_total_column", imgformats=imgformats)


# --- Helper functions ---


def _skewness(x):
    """Compute skewness per sample. x: [N, nlat, nlon]."""
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True)
    std = np.maximum(std, 1e-12)
    return ((x - mean) ** 3).mean(axis=(1, 2)) / (std.squeeze() ** 3)


def _kurtosis(x):
    """Compute kurtosis per sample. x: [N, nlat, nlon]."""
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True)
    std = np.maximum(std, 1e-12)
    return ((x - mean) ** 4).mean(axis=(1, 2)) / (std.squeeze() ** 4) - 3.0
