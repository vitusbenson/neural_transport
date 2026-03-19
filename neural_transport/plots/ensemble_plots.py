"""Registered ensemble diagnostic plots: rank histogram, calibration, spread maps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from neural_transport.plots.base import PlotContext, register_plot


@register_plot(
    name="rank_histogram",
    categories=["ensemble"],
    description="Rank histogram for ensemble calibration",
)
def plot_rank_histogram(result, ctx: PlotContext) -> None:
    """Bar chart of rank histogram with uniform reference line."""
    rh = getattr(result, "diagnostics", {}).get("rank_histogram")
    if rh is None:
        return

    rh = np.asarray(rh)
    n_bins = len(rh)

    with ctx.figure("rank_histogram") as fig:
        ax = fig.add_subplot(111)
        ax.bar(
            np.arange(n_bins),
            rh,
            color="steelblue",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axhline(
            1.0 / n_bins,
            color="red",
            linestyle="--",
            linewidth=1,
            label="Uniform",
        )
        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        ax.set_title("Rank Histogram")
        ax.legend(fontsize=7)


@register_plot(
    name="calibration",
    categories=["ensemble"],
    description="Calibration diagram (nominal vs observed quantiles)",
)
def plot_calibration(result, ctx: PlotContext) -> None:
    """Scatter + perfect diagonal with calibration error annotation."""
    cal = getattr(result, "diagnostics", {}).get("calibration")
    if cal is None:
        return

    nominal = cal.get("nominal")
    observed = cal.get("observed")
    if nominal is None or observed is None:
        return

    nominal = np.asarray(nominal)
    observed = np.asarray(observed)

    with ctx.figure("calibration") as fig:
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect")
        ax.scatter(nominal, observed, s=20, color="steelblue", zorder=3)
        ax.plot(nominal, observed, color="steelblue", linewidth=1)
        ax.set_xlabel("Nominal quantile")
        ax.set_ylabel("Observed fraction")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
        cal_err = cal.get("calibration_error", float("nan"))
        ax.text(0.05, 0.9, f"CE={cal_err:.3f}", transform=ax.transAxes, fontsize=8)
        ax.set_title("Calibration Diagram")


@register_plot(
    name="spread_maps",
    categories=["ensemble"],
    description="Ensemble spread heatmap",
)
def plot_spread_maps(result, ctx: PlotContext) -> None:
    """Heatmap of ensemble spread (std) with colorbar."""
    spread = getattr(result, "maps", {}).get("spread_map")
    if spread is None:
        return

    spread = np.asarray(spread)

    with ctx.figure("spread_maps") as fig:
        ax = fig.add_subplot(111)
        im = ax.imshow(spread, origin="lower", cmap="inferno", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Ensemble Spread")
        ax.set_xticks([])
        ax.set_yticks([])
