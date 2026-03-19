"""Plotting framework: registry, context, and dispatching."""

import neural_transport.plots.animation  # noqa: F401
import neural_transport.plots.conditioning_diagnostics  # noqa: F401
import neural_transport.plots.distributional_plots  # noqa: F401
import neural_transport.plots.ensemble_plots  # noqa: F401
import neural_transport.plots.field_plots  # noqa: F401 — triggers @register_plot decorators
import neural_transport.plots.transport_plots  # noqa: F401
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
