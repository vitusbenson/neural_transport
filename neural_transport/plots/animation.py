"""Animation wrapper: field_animation registered plot."""

from __future__ import annotations

from neural_transport.plots.base import PlotContext, register_plot


@register_plot(
    name="field_animation",
    categories=["animation"],
    description="Animated prediction vs target fields",
)
def plot_field_animation(result, ctx: PlotContext) -> None:
    """Animate predictions vs targets using xmovie (lazy import)."""
    preds = getattr(result, "metadata", {}).get("pred_xr")
    targs = getattr(result, "metadata", {}).get("targ_xr")
    if preds is None or targs is None:
        return

    from neural_transport.plots.plot_results import animate_predictions

    animate_predictions(preds, targs, str(ctx.save_dir))
