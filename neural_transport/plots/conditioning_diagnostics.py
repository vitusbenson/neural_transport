"""Publication-quality diagnostic plots for conditioning evaluation.

Adapted from compare_conditioning_osse.py (experiment 08) and extended
with ensemble diagnostics (rank histogram, calibration, spread maps).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neural_transport.inference.metrics import (
    compute_xco2_column,
)
from neural_transport.plots.utilities.plot_utils import mpl_rc_params, save_figure


def plot_conditioning_comparison(results, out_dir, level_idx=0, max_samples=3, imgformats=None):
    """Grid plot: rows=methods, cols=[GT | Observed | Ens.Mean | |Error| | Samples].

    Parameters
    ----------
    results : dict[str, OSSEResult]
    out_dir : str or Path
    level_idx : int, vertical level to display (column-mean if -1)
    max_samples : int, max individual samples to show
    imgformats : list[str]
    """
    if imgformats is None:
        imgformats = ["png", "pdf"]

    plt.rcParams.update(mpl_rc_params)
    n_exp = len(results)
    n_cols = 4 + max_samples  # GT, Observed, Ens.Mean, |Diff|, Samples

    fig, axes = plt.subplots(
        n_exp,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_exp),
        gridspec_kw={"wspace": 0.05, "hspace": 0.35},
    )
    if n_exp == 1:
        axes = axes[np.newaxis, :]

    # Use first result's GT (all should be the same)
    first_result = next(iter(results.values()))
    gt = first_result.gt
    if level_idx == -1:
        gt_slice = gt.mean(axis=-1)
    else:
        gt_slice = gt[:, :, level_idx]

    for row, (name, res) in enumerate(results.items()):
        ens_mean = res.ensemble_mean
        if level_idx == -1:
            ens_slice = ens_mean.mean(axis=-1)
        else:
            ens_slice = ens_mean[:, :, level_idx]

        diff = np.abs(ens_slice - gt_slice)

        vmin = np.nanpercentile(gt_slice, 2)
        vmax = np.nanpercentile(gt_slice, 98)

        # Col 0: Ground truth
        axes[row, 0].imshow(gt_slice, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
        axes[row, 0].set_title("Ground Truth" if row == 0 else "", fontsize=10)

        # Col 1: Observed locations
        if res.mask_2d is not None:
            obs_display = np.where(res.mask_2d, gt_slice, np.nan)
            axes[row, 1].imshow(obs_display, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
        else:
            axes[row, 1].text(
                0.5,
                0.5,
                "No obs",
                transform=axes[row, 1].transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )
        axes[row, 1].set_title("Observed" if row == 0 else "", fontsize=10)

        # Col 2: Ensemble mean
        axes[row, 2].imshow(ens_slice, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
        axes[row, 2].set_title("Ens. Mean" if row == 0 else "", fontsize=10)

        # Col 3: |Difference|
        dmax = np.nanpercentile(diff, 98) if diff.size > 0 else 1.0
        axes[row, 3].imshow(diff, origin="lower", cmap="Reds", vmin=0, vmax=max(dmax, 1e-8), aspect="auto")
        axes[row, 3].set_title("|Difference|" if row == 0 else "", fontsize=10)

        # Cols 4+: Individual samples
        n_show = min(max_samples, res.samples.shape[0])
        for i in range(n_show):
            ax = axes[row, 4 + i]
            if level_idx == -1:
                sample_slice = res.samples[i].mean(axis=-1)
            else:
                sample_slice = res.samples[i, :, :, level_idx]
            ax.imshow(sample_slice, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
            if row == 0:
                ax.set_title(f"Sample {i}", fontsize=10)

        # Row label with metrics
        label = name.replace("_", "\n")
        m = res.metrics
        metrics_str = f"RMSE={m.rmse_3d_full:.3f}\nR2={m.r2:.3f}\nspread={m.sample_spread:.2f}"
        axes[row, 0].set_ylabel(f"{label}\n\n{metrics_str}", fontsize=8, rotation=0, labelpad=100, va="center")

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

        for i in range(n_show + 4, n_cols):
            axes[row, i].axis("off")

    fig.suptitle("OSSE Conditioning Comparison", fontsize=14, y=1.01)
    plt.tight_layout()
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
    """XCO2 column maps: rows=methods, cols=[GT XCO2 | Pred XCO2 | |Error| | Obs overlay].

    Skips methods where pressure_weights/ak are None.

    Parameters
    ----------
    results : dict[str, OSSEResult]
    out_dir : str or Path
    imgformats : list[str]
    """
    if imgformats is None:
        imgformats = ["png", "pdf"]

    # Filter to methods with pressure_weights and ak
    valid = {k: v for k, v in results.items() if v.pressure_weights is not None and v.ak is not None}
    if not valid:
        return

    plt.rcParams.update(mpl_rc_params)
    n_exp = len(valid)

    fig, axes = plt.subplots(n_exp, 4, figsize=(20, 4 * n_exp), gridspec_kw={"wspace": 0.1, "hspace": 0.35})
    if n_exp == 1:
        axes = axes[np.newaxis, :]

    first_res = next(iter(valid.values()))
    xco2_gt = compute_xco2_column(first_res.gt, first_res.pressure_weights, first_res.ak)
    vmin = np.nanpercentile(xco2_gt, 2)
    vmax = np.nanpercentile(xco2_gt, 98)

    for row, (name, res) in enumerate(valid.items()):
        xco2_pred = compute_xco2_column(res.ensemble_mean, res.pressure_weights, res.ak)
        xco2_err = np.abs(xco2_pred - xco2_gt)

        # Col 0: GT XCO2
        axes[row, 0].imshow(xco2_gt, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
        axes[row, 0].set_title("GT XCO2" if row == 0 else "", fontsize=10)

        # Col 1: Predicted XCO2
        axes[row, 1].imshow(xco2_pred, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
        axes[row, 1].set_title("Pred XCO2" if row == 0 else "", fontsize=10)

        # Col 2: |Error|
        emax = np.nanpercentile(xco2_err, 98) if xco2_err.size > 0 else 1.0
        axes[row, 2].imshow(xco2_err, origin="lower", cmap="Reds", vmin=0, vmax=max(emax, 1e-8), aspect="auto")
        axes[row, 2].set_title("|XCO2 Error|" if row == 0 else "", fontsize=10)

        # Col 3: Obs overlay
        if res.mask_2d is not None:
            obs_display = np.where(res.mask_2d, xco2_gt, np.nan)
            axes[row, 3].imshow(obs_display, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
        else:
            axes[row, 3].text(
                0.5, 0.5, "No obs", transform=axes[row, 3].transAxes, ha="center", va="center", color="gray"
            )
        axes[row, 3].set_title("Obs Overlay" if row == 0 else "", fontsize=10)

        # Row label
        axes[row, 0].set_ylabel(name.replace("_", "\n"), fontsize=9, rotation=0, labelpad=60, va="center")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("XCO2 Column Comparison", fontsize=14, y=1.01)
    plt.tight_layout()
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
    level_idx=0,
    imgformats=None,
):
    """Per-target sample gallery: GT | Masked obs | Samples | Ens mean | |Error|.

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
    level_idx : int
    imgformats : list[str]
    """
    if imgformats is None:
        imgformats = ["png", "pdf"]
    plt.rcParams.update(mpl_rc_params)

    targets = sorted(samples_per_target.keys())[:n_targets_show]
    n_rows = len(targets)
    n_cols = 3 + n_samples_show + 1  # GT, Obs, Ens.Mean, Samples..., |Error|

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3 * n_cols, 3 * n_rows),
        gridspec_kw={"wspace": 0.05, "hspace": 0.3},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, t_pos in enumerate(targets):
        gt = gt_per_target[t_pos]
        samples = samples_per_target[t_pos]
        mask = mask_per_target.get(t_pos)

        if level_idx == -1:
            gt_slice = gt.mean(axis=-1)
            ens_mean = samples.mean(axis=0).mean(axis=-1)
            sample_slices = [s.mean(axis=-1) for s in samples[:n_samples_show]]
        else:
            gt_slice = gt[:, :, level_idx]
            ens_mean = samples.mean(axis=0)[:, :, level_idx]
            sample_slices = [s[:, :, level_idx] for s in samples[:n_samples_show]]

        vmin = np.nanpercentile(gt_slice, 2)
        vmax = np.nanpercentile(gt_slice, 98)

        # Col 0: GT
        axes[row, 0].imshow(gt_slice, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
        if row == 0:
            axes[row, 0].set_title("GT", fontsize=9)
        axes[row, 0].set_ylabel(f"Target {t_pos}", fontsize=8)

        # Col 1: Observed (masked)
        if mask is not None:
            obs_display = np.where(mask, gt_slice, np.nan)
        else:
            obs_display = np.full_like(gt_slice, np.nan)
        axes[row, 1].imshow(obs_display, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
        if row == 0:
            axes[row, 1].set_title("Observed", fontsize=9)

        # Cols 2..2+n_samples: Individual samples
        for s_i, s_slice in enumerate(sample_slices):
            axes[row, 2 + s_i].imshow(s_slice, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
            if row == 0:
                axes[row, 2 + s_i].set_title(f"Sample {s_i}", fontsize=9)

        # Col -2: Ensemble mean
        axes[row, -2].imshow(ens_mean, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
        if row == 0:
            axes[row, -2].set_title("Ens. Mean", fontsize=9)

        # Col -1: |Error|
        diff = np.abs(ens_mean - gt_slice)
        dmax = np.nanpercentile(diff, 98) if diff.size > 0 else 1.0
        axes[row, -1].imshow(diff, origin="lower", cmap="Reds", vmin=0, vmax=max(dmax, 1e-8), aspect="auto")
        if row == 0:
            axes[row, -1].set_title("|Error|", fontsize=9)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    title = "Per-Target Sample Gallery"
    if method_name:
        title += f" — {method_name}"
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    save_figure(fig, out_dir, f"per_target_panel{'_' + method_name if method_name else ''}", imgformats=imgformats)
