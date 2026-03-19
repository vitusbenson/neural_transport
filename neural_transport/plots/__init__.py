"""Plotting framework: registry, context, and dispatching."""

import neural_transport.plots.field_plots  # noqa: F401 — triggers @register_plot decorators
from neural_transport.plots.base import (
    PLOT_CATEGORIES,
    PLOT_REGISTRY,
    PlotContext,
    register_plot,
    run_plots,
)

__all__ = [
    "PLOT_CATEGORIES",
    "PLOT_REGISTRY",
    "PlotContext",
    "register_plot",
    "run_plots",
]
