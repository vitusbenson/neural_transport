"""Publication-quality diagnostic plots for conditioning evaluation.

Adapted from compare_conditioning_osse.py (experiment 08) and extended
with ensemble diagnostics (rank histogram, calibration, spread maps).
Uses Robinson projection with Spectral_r colormap for all maps.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neural_transport.inference.metrics import (
    compute_xco2_column,
)
from neural_transport.plots.utilities.plot_utils import (
    DEFAULT_CMAP_ERROR,
    DEFAULT_CMAP_FIELD,
    DEFAULT_PROJECTION,
    mpl_rc_params,
    plot_map,
    save_figure,
)


def _to_xco2(field_3d, pw, ak):
    """Convert 3D field to total column XCO2 if weights are available, else level mean."""
    if pw is not None and ak is not None and np.any(pw != 0):
        return compute_xco2_column(field_3d, pw, ak)
    return field_3d.mean(axis=-1)


def _get_latlon(result):
    """Extract lat/lon from OSSEResult, or generate default grid."""
    if result.lat is not None and result.lon is not None:
        return result.lat, result.lon
    nlat, nlon = result.gt.shape[:2]
    lat = np.linspace(-90 + 90 / nlat, 90 - 90 / nlat, nlat)
    lon = np.linspace(0, 360 - 360 / nlon, nlon)
    return lat, lon


def _get_shared_pw_ak(results):
    """Extract pressure_weights and ak from the first method that has them.

    This ensures all methods are projected to the same XCO2 space,
    even if some (e.g. unconditional) don't store their own pw/ak.
    """
    for res in results.values():
        pw, ak = res.pressure_weights, res.ak
        if pw is not None and ak is not None and np.any(pw != 0):
            return pw, ak
    return None, None


def plot_conditioning_comparison(results, out_dir, level_idx=-1, max_samples=3, imgformats=None):
    """Grid plot: rows=methods, cols=[GT | Observed | Ens.Mean | Error | Samples].

    Uses Robinson projection with Spectral_r colormap. When level_idx=-1 and
    pressure_weights/ak are available, shows total column XCO2. Error maps use
    a divergent (RdBu_r) colormap centered at zero.

    Parameters
    ----------
    results : dict[str, OSSEResult]
    out_dir : str or Path
    level_idx : int, vertical level to display (XCO2 column if -1)
    max_samples : int, max individual samples to show
    imgformats : list[str]
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec

    if imgformats is None:
        imgformats = ["png", "pdf"]

    plt.rcParams.update(mpl_rc_params)
    n_exp = len(results)
    n_cols = 4 + max_samples  # GT, Observed, Ens.Mean, Error, Samples
    projection = DEFAULT_PROJECTION()

    # Layout: n_exp map rows + 1 thin colorbar row
    fig = plt.figure(figsize=(4.5 * n_cols, 2.8 * n_exp + 0.7))
    gs = GridSpec(n_exp + 1, n_cols, figure=fig, height_ratios=[1] * n_exp + [0.04], wspace=0.02, hspace=0.15)

    axes = np.empty((n_exp, n_cols), dtype=object)
    for r in range(n_exp):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c], projection=projection)

    first_result = next(iter(results.values()))
    lat, lon = _get_latlon(first_result)

    # Use shared pw/ak from the first method that has them, so ALL methods
    # (including unconditional) are projected to the same XCO2 space.
    shared_pw, shared_ak = _get_shared_pw_ak(results)

    def _slice(field_3d):
        if level_idx == -1:
            return _to_xco2(field_3d, shared_pw, shared_ak)
        return field_3d[:, :, level_idx]

    gt_slice = _slice(first_result.gt)
    vmin = float(np.nanpercentile(gt_slice, 2))
    vmax = float(np.nanpercentile(gt_slice, 98))

    # Shared error scale across all methods
    err_absmax = 0.0
    for name, res in results.items():
        err_absmax = max(err_absmax, float(np.nanpercentile(np.abs(_slice(res.ensemble_mean) - gt_slice), 98)))
    err_absmax = max(err_absmax, 1e-8)

    for row, (name, res) in enumerate(results.items()):
        ens_slice = _slice(res.ensemble_mean)
        diff = ens_slice - gt_slice

        plot_map(
            axes[row, 0],
            gt_slice,
            lat,
            lon,
            cmap=DEFAULT_CMAP_FIELD,
            vmin=vmin,
            vmax=vmax,
            title="Ground Truth" if row == 0 else "",
        )

        if res.mask_2d is not None:
            plot_map(
                axes[row, 1],
                np.where(res.mask_2d, gt_slice, np.nan),
                lat,
                lon,
                cmap=DEFAULT_CMAP_FIELD,
                vmin=vmin,
                vmax=vmax,
                title="Observed" if row == 0 else "",
            )
        else:
            axes[row, 1].set_global()
            axes[row, 1].coastlines(linewidth=0.4)
            axes[row, 1].text(
                0.5, 0.5, "No obs", transform=axes[row, 1].transAxes, ha="center", va="center", fontsize=9, color="gray"
            )
            if row == 0:
                axes[row, 1].set_title("Observed", fontsize=9)

        plot_map(
            axes[row, 2],
            ens_slice,
            lat,
            lon,
            cmap=DEFAULT_CMAP_FIELD,
            vmin=vmin,
            vmax=vmax,
            title="Ens. Mean" if row == 0 else "",
        )

        plot_map(
            axes[row, 3],
            diff,
            lat,
            lon,
            cmap=DEFAULT_CMAP_ERROR,
            vmin=-err_absmax,
            vmax=err_absmax,
            title="Error" if row == 0 else "",
        )

        n_show = min(max_samples, res.samples.shape[0])
        for i in range(n_show):
            plot_map(
                axes[row, 4 + i],
                _slice(res.samples[i]),
                lat,
                lon,
                cmap=DEFAULT_CMAP_FIELD,
                vmin=vmin,
                vmax=vmax,
                title=f"Sample {i}" if row == 0 else "",
            )

        label = name.replace("_", "\n")
        m = res.metrics
        axes[row, 0].text(
            -0.15,
            0.5,
            f"{label}\nRMSE={m.rmse_3d_full:.2f}\nSS={m.spread_skill:.2f}",
            transform=axes[row, 0].transAxes,
            fontsize=7,
            va="center",
            ha="right",
        )

        for i in range(n_show + 4, n_cols):
            axes[row, i].set_visible(False)

    col_label = "Total Column XCO2" if level_idx == -1 else f"Level {level_idx}"
    unit = "ppm" if level_idx == -1 else "mixing ratio"
    fig.suptitle(f"Conditioning Comparison — {col_label}", fontsize=12)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.96)

    # Colorbars in the dedicated thin bottom row
    # XCO2 colorbar: spans cols 0-2 (GT, Obs, Ens.Mean) on the left side
    ax_cb_field = fig.add_subplot(gs[n_exp, 0:3])
    ax_cb_field.set_axis_off()
    sm_field = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=DEFAULT_CMAP_FIELD)
    cb_f = fig.colorbar(
        sm_field, ax=ax_cb_field, orientation="horizontal", fraction=1.0, pad=0.1, aspect=25, shrink=0.9
    )
    cb_f.set_label(f"{col_label} [{unit}]", fontsize=8)
    cb_f.ax.tick_params(labelsize=7)

    # Error colorbar: col 3 only
    ax_cb_err = fig.add_subplot(gs[n_exp, 3])
    ax_cb_err.set_axis_off()
    sm_error = ScalarMappable(norm=Normalize(vmin=-err_absmax, vmax=err_absmax), cmap=DEFAULT_CMAP_ERROR)
    cb_e = fig.colorbar(sm_error, ax=ax_cb_err, orientation="horizontal", fraction=1.0, pad=0.1, aspect=8, shrink=0.9)
    cb_e.set_label(f"Error [{unit}]", fontsize=8)
    cb_e.ax.tick_params(labelsize=7)

    save_figure(fig, out_dir, "conditioning_comparison", imgformats=imgformats)


def plot_metrics_summary(results, out_dir, imgformats=None):
    """Bar chart comparing all metrics across methods.

    Parameters
    ----------
    results : dict[str, OSSEResult]
    out_dir : str or Path
    imgformats : list[str]
    """
    if imgformats is None:
        imgformats = ["png", "pdf"]

    plt.rcParams.update(mpl_rc_params)
    names = list(results.keys())

    metrics_keys = [
        "rmse_3d_full",
        "rmse_3d_obs",
        "rmse_3d_away",
        "rmse_xco2_full",
        "rmse_xco2_obs",
        "rmse_xco2_away",
        "r2",
        "spread_skill",
        "roughness_lat",
        "roughness_lon",
        "sample_spread",
        "crps_mean",
        "calibration_error",
    ]
    labels = [
        "RMSE 3D\n(full)",
        "RMSE 3D\n(obs)",
        "RMSE 3D\n(away)",
        "RMSE XCO2\n(full)",
        "RMSE XCO2\n(obs)",
        "RMSE XCO2\n(away)",
        "R\u00b2",
        "Spread/\nSkill",
        "Roughness\n(lat)",
        "Roughness\n(lon)",
        "Sample\nSpread",
        "CRPS\n(mean)",
        "Calibration\nError",
    ]

    n_metrics = len(metrics_keys)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    x = np.arange(len(names))

    for i, (key, label) in enumerate(zip(metrics_keys, labels)):
        values = []
        for n in names:
            v = getattr(results[n].metrics, key, np.nan)
            values.append(v if not (isinstance(v, float) and np.isnan(v)) else 0)
        bars = axes[i].bar(x, values, color=plt.cm.tab10(x / max(len(names), 1)))
        axes[i].set_xticks(x)
        axes[i].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=6, rotation=45, ha="right")
        axes[i].set_title(label, fontsize=11)
        axes[i].grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=6
            )

    for i in range(n_metrics, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    save_figure(fig, out_dir, "conditioning_metrics", imgformats=imgformats)


def plot_ensemble_diagnostics(results, out_dir, imgformats=None):
    """Per-method ensemble diagnostics: rank histogram, calibration, spread map.

    Parameters
    ----------
    results : dict[str, OSSEResult]
    out_dir : str or Path
    imgformats : list[str]
    """
    if imgformats is None:
        imgformats = ["png", "pdf"]

    plt.rcParams.update(mpl_rc_params)
    n_exp = len(results)

    fig, axes = plt.subplots(n_exp, 3, figsize=(15, 4 * n_exp), gridspec_kw={"hspace": 0.4, "wspace": 0.3})
    if n_exp == 1:
        axes = axes[np.newaxis, :]

    for row, (name, res) in enumerate(results.items()):
        # Col 0: Rank histogram
        ax = axes[row, 0]
        if res.rank_hist is not None:
            n_bins = len(res.rank_hist)
            ax.bar(np.arange(n_bins), res.rank_hist, color="steelblue", edgecolor="white", linewidth=0.5)
            ax.axhline(1.0 / n_bins, color="red", linestyle="--", linewidth=1, label="Uniform")
            ax.set_xlabel("Rank")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center", color="gray")
        ax.set_title(f"{name}: Rank Histogram", fontsize=9)

        # Col 1: Calibration diagram
        ax = axes[row, 1]
        if res.calibration_data is not None:
            nominal = res.calibration_data["nominal"]
            observed = res.calibration_data["observed"]
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect")
            ax.scatter(nominal, observed, s=20, color="steelblue", zorder=3)
            ax.plot(nominal, observed, color="steelblue", linewidth=1)
            ax.set_xlabel("Nominal quantile")
            ax.set_ylabel("Observed fraction")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.legend(fontsize=7)
            cal_err = res.calibration_data.get("calibration_error", np.nan)
            ax.text(0.05, 0.9, f"CE={cal_err:.3f}", transform=ax.transAxes, fontsize=8)
        else:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center", color="gray")
        ax.set_title(f"{name}: Calibration", fontsize=9)

        # Col 2: Spread map (ensemble std, column-mean)
        ax = axes[row, 2]
        spread = res.samples.std(axis=0).mean(axis=-1)  # [nlat, nlon]
        im = ax.imshow(spread, origin="lower", cmap="inferno", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{name}: Ensemble Spread", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    save_figure(fig, out_dir, "ensemble_diagnostics", imgformats=imgformats)


def plot_xco2_maps(results, out_dir, imgformats=None):
    """XCO2 column maps: rows=methods, cols=[GT XCO2 | Pred XCO2 | Error | Obs overlay].

    Now redundant with plot_conditioning_comparison(level_idx=-1), which shows
    XCO2 by default. Kept for backward compatibility.

    Skips methods where pressure_weights/ak are None.
    """
    if imgformats is None:
        imgformats = ["png", "pdf"]

    valid = {k: v for k, v in results.items() if v.pressure_weights is not None and v.ak is not None}
    if not valid:
        return

    plt.rcParams.update(mpl_rc_params)
    n_exp = len(valid)
    projection = DEFAULT_PROJECTION()

    fig, axes = plt.subplots(
        n_exp,
        4,
        figsize=(20, 3 * n_exp),
        subplot_kw={"projection": projection},
        gridspec_kw={"wspace": 0.02, "hspace": 0.25},
    )
    if n_exp == 1:
        axes = axes[np.newaxis, :]

    first_res = next(iter(valid.values()))
    lat, lon = _get_latlon(first_res)
    xco2_gt = compute_xco2_column(first_res.gt, first_res.pressure_weights, first_res.ak)
    vmin = float(np.nanpercentile(xco2_gt, 2))
    vmax = float(np.nanpercentile(xco2_gt, 98))

    for row, (name, res) in enumerate(valid.items()):
        xco2_pred = compute_xco2_column(res.ensemble_mean, res.pressure_weights, res.ak)
        xco2_err = xco2_pred - xco2_gt  # signed error

        plot_map(
            axes[row, 0],
            xco2_gt,
            lat,
            lon,
            cmap=DEFAULT_CMAP_FIELD,
            vmin=vmin,
            vmax=vmax,
            title="GT XCO2" if row == 0 else "",
        )
        plot_map(
            axes[row, 1],
            xco2_pred,
            lat,
            lon,
            cmap=DEFAULT_CMAP_FIELD,
            vmin=vmin,
            vmax=vmax,
            title="Pred XCO2" if row == 0 else "",
        )
        plot_map(
            axes[row, 2],
            xco2_err,
            lat,
            lon,
            cmap=DEFAULT_CMAP_ERROR,
            symmetric=True,
            title="XCO2 Error" if row == 0 else "",
        )

        if res.mask_2d is not None:
            obs_display = np.where(res.mask_2d, xco2_gt, np.nan)
            plot_map(
                axes[row, 3],
                obs_display,
                lat,
                lon,
                cmap=DEFAULT_CMAP_FIELD,
                vmin=vmin,
                vmax=vmax,
                title="Obs Overlay" if row == 0 else "",
            )
        else:
            axes[row, 3].set_global()
            axes[row, 3].coastlines(linewidth=0.4)

        axes[row, 0].text(
            -0.08, 0.5, name.replace("_", "\n"), transform=axes[row, 0].transAxes, fontsize=8, va="center", ha="right"
        )

    fig.suptitle("XCO2 Column Comparison", fontsize=12, y=1.01)
    fig.subplots_adjust(left=0.08)
    save_figure(fig, out_dir, "xco2_maps", imgformats=imgformats)


def plot_zonal_mean(results, out_dir, imgformats=None):
    """Zonal mean lat-level cross-sections.

    Parameters
    ----------
    results : dict[str, OSSEResult]
    out_dir : str or Path
    imgformats : list[str]
    """
    if imgformats is None:
        imgformats = ["png", "pdf"]

    plt.rcParams.update(mpl_rc_params)
    n_exp = len(results)

    fig, axes = plt.subplots(1, n_exp + 1, figsize=(4 * (n_exp + 1), 5), sharey=True)
    if n_exp == 0:
        plt.close(fig)
        return

    first_res = next(iter(results.values()))
    gt_zonal = first_res.gt.mean(axis=1)  # [nlat, nlev]
    vmin = np.nanpercentile(gt_zonal, 2)
    vmax = np.nanpercentile(gt_zonal, 98)

    axes[0].imshow(gt_zonal, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title("Ground Truth", fontsize=10)
    axes[0].set_ylabel("Latitude index")
    axes[0].set_xlabel("Level")

    for i, (name, res) in enumerate(results.items()):
        pred_zonal = res.ensemble_mean.mean(axis=1)  # [nlat, nlev]
        axes[i + 1].imshow(pred_zonal, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
        axes[i + 1].set_title(name, fontsize=9)
        axes[i + 1].set_xlabel("Level")

    plt.tight_layout()
    save_figure(fig, out_dir, "zonal_mean_comparison", imgformats=imgformats)


# ---------------------------------------------------------------------------
# Registered wrappers (accept EvalResult + PlotContext)
# ---------------------------------------------------------------------------

from neural_transport.plots.base import PlotContext, register_plot  # noqa: E402


@register_plot(
    name="conditioning_comparison",
    categories=["conditioning"],
    description="Single-method conditioning comparison: GT | Observed | Pred | |Diff|",
)
def plot_conditioning_comparison_single(result, ctx: PlotContext) -> None:
    """Simplified single-method conditioning comparison."""
    meta = getattr(result, "metadata", {})
    pred = meta.get("pred")
    gt = meta.get("gt")
    if pred is None or gt is None:
        return

    pred = np.asarray(pred)
    gt = np.asarray(gt)
    mask_2d = meta.get("mask_2d")

    # Use first level or 2D
    if pred.ndim == 3:
        pred_slice = pred[:, :, 0]
        gt_slice = gt[:, :, 0]
    else:
        pred_slice = pred
        gt_slice = gt

    diff = np.abs(pred_slice - gt_slice)
    vmin = np.nanpercentile(gt_slice, 2)
    vmax = np.nanpercentile(gt_slice, 98)

    n_cols = 4 if mask_2d is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 3.5))

    axes[0].imshow(gt_slice, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title("Ground Truth")

    col = 1
    if mask_2d is not None:
        mask_2d = np.asarray(mask_2d)
        obs_display = np.where(mask_2d, gt_slice, np.nan)
        axes[col].imshow(obs_display, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
        axes[col].set_title("Observed")
        col += 1

    axes[col].imshow(pred_slice, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
    axes[col].set_title("Prediction")

    dmax = np.nanpercentile(diff, 98) if diff.size > 0 else 1.0
    axes[col + 1].imshow(diff, origin="lower", cmap="Reds", vmin=0, vmax=max(dmax, 1e-8), aspect="auto")
    axes[col + 1].set_title("|Difference|")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    ctx.savefig(fig, "conditioning_comparison")


@register_plot(
    name="error_maps",
    categories=["conditioning"],
    description="Spatial error maps (bias, RMSE, spread)",
)
def plot_error_maps_registered(result, ctx: PlotContext) -> None:
    """1xN grid of spatial error maps with colorbars."""
    maps = getattr(result, "maps", {})
    bias_map = maps.get("bias_map")
    rmse_map = maps.get("rmse_map")
    if bias_map is None and rmse_map is None:
        return

    panels = []
    if bias_map is not None:
        panels.append((np.asarray(bias_map), "Bias", "RdBu_r"))
    if rmse_map is not None:
        panels.append((np.asarray(rmse_map), "RMSE", "Reds"))
    spread_map = maps.get("spread_map")
    if spread_map is not None:
        panels.append((np.asarray(spread_map), "Spread", "inferno"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (data, title, cmap) in zip(axes, panels):
        if "Bias" in title:
            vmax = np.abs(data).max()
            im = ax.imshow(data, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        else:
            im = ax.imshow(data, origin="lower", cmap=cmap, aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    ctx.savefig(fig, "error_maps")


@register_plot(
    name="zonal_mean",
    categories=["conditioning"],
    description="Single-method zonal mean lat-height cross-section",
)
def plot_zonal_mean_single(result, ctx: PlotContext) -> None:
    """Single-method zonal mean: GT | Pred | |Diff|."""
    meta = getattr(result, "metadata", {})
    pred = meta.get("pred")
    gt = meta.get("gt")
    if pred is None or gt is None:
        return

    pred = np.asarray(pred)
    gt = np.asarray(gt)

    # Need 3D for zonal mean
    if pred.ndim < 3:
        return

    pred_zonal = pred.mean(axis=1)  # [nlat, nlev]
    gt_zonal = gt.mean(axis=1)
    diff_zonal = np.abs(pred_zonal - gt_zonal)

    vmin = np.nanpercentile(gt_zonal, 2)
    vmax = np.nanpercentile(gt_zonal, 98)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, title in zip(
        axes,
        [gt_zonal, pred_zonal, diff_zonal],
        ["GT Zonal Mean", "Pred Zonal Mean", "|Difference|"],
    ):
        if "Diff" in title:
            dmax = np.abs(data).max()
            ax.imshow(data.T, origin="lower", cmap="RdBu_r", aspect="auto", vmin=-dmax, vmax=dmax)
        else:
            ax.imshow(data.T, origin="lower", cmap="cividis", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Latitude index")
        ax.set_ylabel("Level")

    fig.tight_layout()
    ctx.savefig(fig, "zonal_mean")


# ---------------------------------------------------------------------------
# Multi-target diagnostic plots (accept zarr data or OSSEResult dicts)
# ---------------------------------------------------------------------------


def plot_obs_match_scatter(results, out_dir, level_idx=0, imgformats=None):
    """Scatter plot of prediction vs GT at observed pixels, per method.

    Perfect conditioning → points on the diagonal.

    Parameters
    ----------
    results : dict[str, OSSEResult]
    out_dir : str or Path
    level_idx : int
        Vertical level for comparison (-1 for column mean).
    imgformats : list[str]
    """
    if imgformats is None:
        imgformats = ["png", "pdf"]
    plt.rcParams.update(mpl_rc_params)

    n_exp = len(results)
    fig, axes = plt.subplots(1, n_exp, figsize=(4 * n_exp, 4), squeeze=False)

    for col, (name, res) in enumerate(results.items()):
        ax = axes[0, col]
        ens_mean = res.ensemble_mean
        gt = res.gt
        mask = res.mask_2d

        if mask is None or mask.sum() == 0:
            ax.text(0.5, 0.5, "No obs", transform=ax.transAxes, ha="center", va="center", color="gray")
            ax.set_title(name, fontsize=9)
            continue

        if level_idx == -1:
            pred_vals = ens_mean.mean(axis=-1)[mask]
            gt_vals = gt.mean(axis=-1)[mask]
        else:
            pred_vals = ens_mean[:, :, level_idx][mask]
            gt_vals = gt[:, :, level_idx][mask]

        ax.scatter(gt_vals, pred_vals, s=3, alpha=0.5, color="C0")
        lims = [min(gt_vals.min(), pred_vals.min()), max(gt_vals.max(), pred_vals.max())]
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="Perfect")
        ax.set_xlabel("GT at obs")
        ax.set_ylabel("Pred at obs")
        ax.set_title(name, fontsize=9)
        ax.set_aspect("equal")
        rmse_obs = np.sqrt(np.mean((pred_vals - gt_vals) ** 2))
        ax.text(0.05, 0.92, f"RMSE={rmse_obs:.3f}", transform=ax.transAxes, fontsize=7)

    fig.suptitle("Obs Match: Prediction vs GT at Observed Pixels", fontsize=11)
    plt.tight_layout()
    save_figure(fig, out_dir, "obs_match_scatter", imgformats=imgformats)


def plot_spread_at_unobs(results, out_dir, level_idx=0, imgformats=None):
    """Spread map showing ensemble std ONLY at unobserved locations, per method.

    Parameters
    ----------
    results : dict[str, OSSEResult]
    out_dir : str or Path
    level_idx : int
        Vertical level (-1 for column mean).
    imgformats : list[str]
    """
    if imgformats is None:
        imgformats = ["png", "pdf"]
    plt.rcParams.update(mpl_rc_params)

    n_exp = len(results)
    fig, axes = plt.subplots(1, n_exp, figsize=(4 * n_exp, 3.5), squeeze=False)

    spreads = []
    for name, res in results.items():
        if level_idx == -1:
            spread = res.samples.std(axis=0).mean(axis=-1)
        else:
            spread = res.samples[:, :, :, level_idx].std(axis=0)
        if res.mask_2d is not None:
            spread_unobs = np.where(~res.mask_2d, spread, np.nan)
        else:
            spread_unobs = spread
        spreads.append(spread_unobs)

    vmax = max(np.nanpercentile(s, 98) for s in spreads if np.any(np.isfinite(s)))

    for col, (name, spread) in enumerate(zip(results.keys(), spreads)):
        ax = axes[0, col]
        im = ax.imshow(spread, origin="lower", cmap="inferno", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Ensemble Spread (unobs only)", shrink=0.8)
    fig.suptitle("Spread at Unobserved Locations", fontsize=11)
    plt.tight_layout()
    save_figure(fig, out_dir, "spread_at_unobs", imgformats=imgformats)


def plot_per_target_panel(
    samples_per_target,
    gt_per_target,
    mask_per_target,
    obs_values_per_target,
    out_dir,
    method_name="",
    n_targets_show=4,
    n_samples_show=5,
    level_idx=-1,
    lat=None,
    lon=None,
    pressure_weights=None,
    ak=None,
    imgformats=None,
):
    """Per-target sample gallery: GT | Observed | Samples | Ens mean | Error.

    Uses Robinson projection with Spectral_r colormap. When level_idx=-1 and
    pressure_weights/ak are provided, shows total column XCO2.

    Parameters
    ----------
    samples_per_target : dict[int, np.ndarray]
        Mapping target_pos → samples array [n_samples, nlat, nlon, nlev].
    gt_per_target : dict[int, np.ndarray]
        Mapping target_pos → GT array [nlat, nlon, nlev].
    mask_per_target : dict[int, np.ndarray]
        Mapping target_pos → obs mask [nlat, nlon] bool.
    obs_values_per_target : dict[int, np.ndarray]
        Mapping target_pos → obs values [nlat, nlon].
    out_dir : str or Path
    method_name : str
    n_targets_show : int
    n_samples_show : int
    level_idx : int, -1 for XCO2/column mean
    lat, lon : np.ndarray, optional
    pressure_weights, ak : np.ndarray, optional — for XCO2 computation
    imgformats : list[str]
    """
    if imgformats is None:
        imgformats = ["png", "pdf"]
    plt.rcParams.update(mpl_rc_params)

    targets = sorted(samples_per_target.keys())[:n_targets_show]
    n_rows = len(targets)
    n_cols = 3 + n_samples_show + 1  # GT, Obs, Samples..., Ens.Mean, Error

    projection = DEFAULT_PROJECTION()

    # Generate default lat/lon if not provided
    first_gt = gt_per_target[targets[0]]
    nlat, nlon = first_gt.shape[:2]
    if lat is None:
        lat = np.linspace(-90 + 90 / nlat, 90 - 90 / nlat, nlat)
    if lon is None:
        lon = np.linspace(0, 360 - 360 / nlon, nlon)

    def _to_2d(field_3d, pw=pressure_weights, a=ak):
        if level_idx == -1:
            return _to_xco2(field_3d, pw, a)
        return field_3d[:, :, level_idx]

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec

    # Pre-compute global field and error ranges across all targets
    all_gt_slices = [_to_2d(gt_per_target[t]) for t in targets]
    all_ens_means = [_to_2d(samples_per_target[t].mean(axis=0)) for t in targets]
    all_vals = np.concatenate([s.ravel() for s in all_gt_slices + all_ens_means])
    vmin = float(np.nanpercentile(all_vals, 2))
    vmax = float(np.nanpercentile(all_vals, 98))

    err_absmax = 0.0
    for gt_s, ens_s in zip(all_gt_slices, all_ens_means):
        err_absmax = max(err_absmax, float(np.nanpercentile(np.abs(ens_s - gt_s), 98)))
    err_absmax = max(err_absmax, 1e-8)

    # Layout: n_rows map rows + 1 thin colorbar row
    fig = plt.figure(figsize=(4 * n_cols, 2.5 * n_rows + 0.6))
    gs = GridSpec(n_rows + 1, n_cols, figure=fig, height_ratios=[1] * n_rows + [0.04], wspace=0.02, hspace=0.15)

    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c], projection=projection)

    for row, t_pos in enumerate(targets):
        gt_slice = all_gt_slices[row]
        ens_mean = all_ens_means[row]
        samples = samples_per_target[t_pos]
        mask = mask_per_target.get(t_pos)
        sample_slices = [_to_2d(s) for s in samples[:n_samples_show]]

        plot_map(
            axes[row, 0],
            gt_slice,
            lat,
            lon,
            cmap=DEFAULT_CMAP_FIELD,
            vmin=vmin,
            vmax=vmax,
            title="GT" if row == 0 else "",
        )
        axes[row, 0].text(
            -0.05,
            0.5,
            f"Target {t_pos}",
            transform=axes[row, 0].transAxes,
            fontsize=7,
            va="center",
            ha="right",
            rotation=90,
        )

        obs_display = np.where(mask, gt_slice, np.nan) if mask is not None else np.full_like(gt_slice, np.nan)
        plot_map(
            axes[row, 1],
            obs_display,
            lat,
            lon,
            cmap=DEFAULT_CMAP_FIELD,
            vmin=vmin,
            vmax=vmax,
            title="Observed" if row == 0 else "",
        )

        for s_i, s_slice in enumerate(sample_slices):
            plot_map(
                axes[row, 2 + s_i],
                s_slice,
                lat,
                lon,
                cmap=DEFAULT_CMAP_FIELD,
                vmin=vmin,
                vmax=vmax,
                title=f"Sample {s_i}" if row == 0 else "",
            )

        plot_map(
            axes[row, -2],
            ens_mean,
            lat,
            lon,
            cmap=DEFAULT_CMAP_FIELD,
            vmin=vmin,
            vmax=vmax,
            title="Ens. Mean" if row == 0 else "",
        )

        diff = ens_mean - gt_slice
        plot_map(
            axes[row, -1],
            diff,
            lat,
            lon,
            cmap=DEFAULT_CMAP_ERROR,
            vmin=-err_absmax,
            vmax=err_absmax,
            title="Error" if row == 0 else "",
        )

    col_label = "XCO2" if level_idx == -1 else f"Level {level_idx}"
    unit = "ppm" if level_idx == -1 else "mixing ratio"
    title_str = f"Per-Target Gallery — {col_label}"
    if method_name:
        title_str += f" — {method_name}"
    fig.suptitle(title_str, fontsize=11)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.95)

    # Colorbars in the thin bottom row
    sm_field = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=DEFAULT_CMAP_FIELD)
    sm_error = ScalarMappable(norm=Normalize(vmin=-err_absmax, vmax=err_absmax), cmap=DEFAULT_CMAP_ERROR)

    ax_cb_f = fig.add_subplot(gs[n_rows, 0 : n_cols - 1])
    ax_cb_f.set_axis_off()
    cb_f = fig.colorbar(sm_field, ax=ax_cb_f, orientation="horizontal", fraction=1.0, pad=0.1, aspect=30, shrink=0.5)
    cb_f.set_label(f"{col_label} [{unit}]", fontsize=8)
    cb_f.ax.tick_params(labelsize=7)

    ax_cb_e = fig.add_subplot(gs[n_rows, n_cols - 1])
    ax_cb_e.set_axis_off()
    cb_e = fig.colorbar(sm_error, ax=ax_cb_e, orientation="horizontal", fraction=1.0, pad=0.1, aspect=8, shrink=0.9)
    cb_e.set_label(f"Error [{unit}]", fontsize=8)
    cb_e.ax.tick_params(labelsize=7)

    save_figure(fig, out_dir, f"per_target_panel{'_' + method_name if method_name else ''}", imgformats=imgformats)


# ── Fine-scale detail metric plots ───────────────────────────────────


def plot_power_spectra(method_spectra, out_dir, imgformats=None):
    """Plot power spectra comparison across methods.

    Parameters
    ----------
    method_spectra : dict[str, tuple[np.ndarray, np.ndarray]]
        Mapping method_name → (wavenumbers, power). Must include "gt".
    out_dir : str or Path
    imgformats : list[str]
    """

    if imgformats is None:
        imgformats = ["png", "pdf"]
    plt.rcParams.update(mpl_rc_params)

    fig, ax = plt.subplots(figsize=(8, 5))

    if "gt" in method_spectra:
        wn, ps = method_spectra["gt"]
        ax.loglog(wn, ps, "k-", linewidth=2, label="Ground Truth", zorder=10)

    colors = plt.cm.tab10(np.linspace(0, 1, len(method_spectra)))
    for i, (name, (wn, ps)) in enumerate(method_spectra.items()):
        if name == "gt":
            continue
        ax.loglog(wn, ps, color=colors[i], linewidth=1.2, alpha=0.8, label=name)

    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Power")
    ax.set_title("Power Spectrum: Ground Truth vs Methods")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, out_dir, "power_spectra", imgformats=imgformats)


def plot_detail_metrics_bars(detail_metrics, methods, out_dir, imgformats=None):
    """Bar charts for fine-scale detail metrics.

    Parameters
    ----------
    detail_metrics : dict[str, dict[str, float]]
        Mapping method_name → dict with grad_ratio, spectral_div, high_freq_power_ratio.
    methods : list[str]
        Ordered list of method names to plot.
    out_dir : str or Path
    imgformats : list[str]
    """
    if imgformats is None:
        imgformats = ["png", "pdf"]
    plt.rcParams.update(mpl_rc_params)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(methods))
    width = 0.6

    # 1. Gradient ratio
    grad_ratios = [detail_metrics[m]["grad_ratio"] for m in methods]
    colors = ["#2ca02c" if 0.8 <= r <= 1.2 else "#d62728" for r in grad_ratios]
    axes[0].bar(x, grad_ratios, width, color=colors, alpha=0.8)
    axes[0].axhline(1.0, color="k", linestyle="--", linewidth=0.8, label="GT level")
    axes[0].set_ylabel("Gradient Ratio (pred/GT)")
    axes[0].set_title("Spatial Sharpness")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    axes[0].legend(fontsize=7)

    # 2. Spectral divergence
    spec_divs = [detail_metrics[m]["spectral_div"] for m in methods]
    axes[1].bar(x, spec_divs, width, color="steelblue", alpha=0.8)
    axes[1].set_ylabel("Spectral Divergence")
    axes[1].set_title("Log-Spectral Distance (lower=better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=45, ha="right", fontsize=8)

    # 3. High-frequency power ratio
    hf_ratios = [detail_metrics[m]["high_freq_power_ratio"] for m in methods]
    colors = ["#2ca02c" if 0.5 <= r <= 2.0 else "#d62728" for r in hf_ratios]
    axes[2].bar(x, hf_ratios, width, color=colors, alpha=0.8)
    axes[2].axhline(1.0, color="k", linestyle="--", linewidth=0.8, label="GT level")
    axes[2].set_ylabel("HF Power Ratio (pred/GT)")
    axes[2].set_title("High-Frequency Detail")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    axes[2].legend(fontsize=7)

    fig.suptitle("Fine-Scale Detail Metrics", fontsize=12)
    fig.tight_layout()
    save_figure(fig, out_dir, "detail_metrics", imgformats=imgformats)
