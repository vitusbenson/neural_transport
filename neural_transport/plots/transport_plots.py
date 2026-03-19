"""Registered transport-specific plots: metric curves and obspack stations."""

from __future__ import annotations

import numpy as np

from neural_transport.plots.base import PlotContext, register_plot


@register_plot(
    name="metric_curves",
    categories=["transport"],
    description="Metric values vs lead time",
)
def plot_metric_curves(result, ctx: PlotContext) -> None:
    """Line plot of metric values vs lead time."""
    curves = getattr(result, "metadata", {}).get("metric_curves")
    if curves is None:
        return

    with ctx.figure("metric_curves") as fig:
        ax = fig.add_subplot(111)
        for metric_name, values in curves.items():
            values = np.asarray(values)
            ax.plot(np.arange(len(values)), values, label=metric_name)
        ax.set_xlabel("Lead time")
        ax.set_ylabel("Metric value")
        ax.set_title("Metrics vs Lead Time")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)


@register_plot(
    name="obspack_stations",
    categories=["transport"],
    description="ObsPack station time series",
)
def plot_obspack_stations(result, ctx: PlotContext) -> None:
    """Station-level time series from ObsPack data."""
    obspack = getattr(result, "metadata", {}).get("obspack_data")
    if obspack is None:
        return

    with ctx.figure("obspack_stations") as fig:
        ax = fig.add_subplot(111)
        for station_name, data in obspack.items():
            obs = np.asarray(data.get("obs", []))
            pred = np.asarray(data.get("pred", []))
            if len(obs) > 0:
                ax.plot(obs, label=f"{station_name} obs", linestyle="--", alpha=0.7)
            if len(pred) > 0:
                ax.plot(pred, label=f"{station_name} pred")
        ax.set_xlabel("Time step")
        ax.set_ylabel("CO2")
        ax.set_title("ObsPack Station Comparison")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
