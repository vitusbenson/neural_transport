"""Plotting framework: PlotContext, registry, and run_plots dispatcher."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_transport.evaluation.suite import EvalResult

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from neural_transport.configs import PlotConfig
from neural_transport.plots.utilities.plot_utils import mpl_rc_params, save_figure

# ---------------------------------------------------------------------------
# Plot registry
# ---------------------------------------------------------------------------

PLOT_REGISTRY: dict[str, dict[str, Any]] = {}

PLOT_CATEGORIES: dict[str, list[str]] = {
    "always": ["field_maps", "xco2_maps", "lat_height"],
    "ensemble": ["spread_maps", "rank_histogram", "calibration"],
    "distributional": [
        "marginals",
        "power_spectrum",
        "qq_plot",
        "sample_grid",
        "spatial_patterns",
        "lat_height_comparison",
        "distributional_summary",
    ],
    "conditioning": ["conditioning_comparison", "error_maps", "zonal_mean"],
    "transport": ["metric_curves", "obspack_stations"],
    "ablation": ["sweep_plots", "summary_bars", "pareto_front"],
    "animation": ["field_animation"],
}


def register_plot(name: str, categories: list[str], description: str = "") -> Callable:
    """Decorator that registers a plot function in PLOT_REGISTRY."""

    def decorator(func: Callable) -> Callable:
        PLOT_REGISTRY[name] = {
            "func": func,
            "categories": categories,
            "description": description,
        }
        return func

    return decorator


# ---------------------------------------------------------------------------
# PlotContext
# ---------------------------------------------------------------------------


class PlotContext:
    """Shared context for creating, styling, and saving figures."""

    def __init__(self, config: PlotConfig | None = None) -> None:
        self.config = config or PlotConfig()
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update(mpl_rc_params)

    def savefig(self, fig: Figure, name: str) -> None:
        """Save figure using the shared save_figure utility."""
        save_figure(
            fig,
            out_dir=self.save_dir,
            filename=name,
            imgformats=self.config.imgformats,
            dpi=self.config.dpi,
        )

    def subplot_grid(self, nrows: int, ncols: int, **kwargs: Any) -> tuple[Figure, Any]:
        """Create a subplot grid with figsize scaled by config.figsize_scale."""
        base_w, base_h = 4.0, 3.5
        scale = self.config.figsize_scale
        figsize = kwargs.pop("figsize", (ncols * base_w * scale, nrows * base_h * scale))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
        return fig, axes

    @contextlib.contextmanager
    def figure(self, name: str, **kwargs: Any) -> Generator[Figure, None, None]:
        """Context manager that creates a figure and auto-saves on exit."""
        fig = plt.figure(**kwargs)
        try:
            yield fig
        finally:
            self.savefig(fig, name)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def run_plots(result: EvalResult, ctx: PlotContext, categories: list[str] | None = None) -> list[str]:
    """Dispatch registered plots for the requested categories.

    If *categories* is None, auto-infer from EvalResult content:
    - always included: "always"
    - add "ensemble" if result.ensemble is set
    - add "distributional" if result.distributional is set
    """
    if categories is None:
        categories = ["always"]
        if getattr(result, "ensemble", None) is not None:
            categories.append("ensemble")
        if getattr(result, "distributional", None) is not None:
            categories.append("distributional")
        meta = getattr(result, "metadata", {})
        maps = getattr(result, "maps", {})
        if maps.get("bias_map") is not None or meta.get("mask_2d") is not None:
            categories.append("conditioning")
        if meta.get("experiment_type") == "transport":
            categories.append("transport")
        if meta.get("gt_fields") is not None and meta.get("gen_fields") is not None:
            if "distributional" not in categories:
                categories.append("distributional")

    # Collect plot names for the requested categories
    requested: set[str] = set()
    for cat in categories:
        requested.update(PLOT_CATEGORIES.get(cat, []))

    called: list[str] = []
    for name, entry in PLOT_REGISTRY.items():
        if name in requested:
            entry["func"](result, ctx)
            called.append(name)

    return called
