"""Always-on field plots: CO2 maps, XCO2 maps, lat-height cross-sections."""

from __future__ import annotations

import numpy as np

from neural_transport.plots.base import PlotContext, register_plot

try:
    import cartopy.crs as ccrs

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pressure_weighted_column(fields, level_values):
    """Compute pressure-weighted column average (XCO2).

    Args:
        fields: [..., nlev] mixing ratio fields
        level_values: 1D array of pressure midpoints in hPa

    Returns:
        [...] pressure-weighted column mean
    """
    if level_values is None:
        return fields.mean(axis=-1)

    level_values = np.asarray(level_values, dtype=np.float64)
    nlev = len(level_values)

    boundaries = np.zeros(nlev + 1)
    boundaries[0] = level_values[0] + (level_values[0] - level_values[1]) / 2
    boundaries[-1] = 0.0
    for k in range(1, nlev):
        boundaries[k] = (level_values[k - 1] + level_values[k]) / 2

    dp = np.abs(np.diff(boundaries))
    dp_shape = (1,) * (fields.ndim - 1) + (nlev,)
    dp_broadcast = dp.reshape(dp_shape)
    return (fields * dp_broadcast).sum(axis=-1) / dp.sum()


def _get_field_data(result):
    """Extract pred/gt/lat/lon/level_values from result.metadata.

    Returns None if required keys are missing.
    """
    meta = getattr(result, "metadata", {})
    pred = meta.get("pred")
    gt = meta.get("gt")
    if pred is None or gt is None:
        return None
    lat = meta.get("lat")
    lon = meta.get("lon")
    level_values = meta.get("level_values")
    return {
        "pred": np.asarray(pred),
        "gt": np.asarray(gt),
        "lat": np.asarray(lat) if lat is not None else None,
        "lon": np.asarray(lon) if lon is not None else None,
        "level_values": level_values,
    }


def _plot_map_row(axes, gt_2d, pred_2d, title_prefix, lat=None, lon=None):
    """Plot GT | Pred | |Diff| in a row of 3 axes."""
    diff = np.abs(pred_2d - gt_2d)
    panels = [
        (gt_2d, f"{title_prefix} GT"),
        (pred_2d, f"{title_prefix} Pred"),
        (diff, f"{title_prefix} |Diff|"),
    ]
    for ax, (data, title) in zip(axes, panels):
        if HAS_CARTOPY and hasattr(ax, "coastlines"):
            ax.coastlines(linewidth=0.5)
        if lat is not None and lon is not None:
            im = ax.pcolormesh(lon, lat, data, shading="auto")
        else:
            im = ax.imshow(data, origin="lower", aspect="auto")
        ax.set_title(title, fontsize=8)
    return im


# ---------------------------------------------------------------------------
# Registered plot functions
# ---------------------------------------------------------------------------


@register_plot(
    name="field_maps",
    categories=["always"],
    description="CO2 maps at target pressure levels",
)
def plot_field_maps(result, ctx: PlotContext) -> None:
    """CO2 maps at individual pressure levels: GT | Pred | |Diff|."""
    data = _get_field_data(result)
    if data is None:
        return

    pred, gt = data["pred"], data["gt"]
    lat, lon = data["lat"], data["lon"]
    level_values = data["level_values"]

    # If 2D fields (no level dimension), show single row
    if pred.ndim == 2:
        subplot_kw = {"projection": ccrs.Robinson()} if HAS_CARTOPY else {}
        fig, axes = ctx.subplot_grid(1, 3, subplot_kw=subplot_kw)
        _plot_map_row(axes, gt, pred, "CO2", lat=lat, lon=lon)
        fig.tight_layout()
        ctx.savefig(fig, "field_maps")
        return

    # 3D: [nlat, nlon, nlev]
    nlev = pred.shape[-1]
    labels = (
        [f"{int(v)} hPa" for v in level_values] if level_values is not None else [f"Level {i}" for i in range(nlev)]
    )

    subplot_kw = {"projection": ccrs.Robinson()} if HAS_CARTOPY else {}
    fig, axes = ctx.subplot_grid(nlev, 3, subplot_kw=subplot_kw)
    if nlev == 1:
        axes = axes[np.newaxis, :]

    for k in range(nlev):
        _plot_map_row(axes[k], gt[:, :, k], pred[:, :, k], labels[k], lat=lat, lon=lon)

    fig.tight_layout()
    ctx.savefig(fig, "field_maps")


@register_plot(
    name="xco2_maps",
    categories=["always"],
    description="Column-averaged XCO2 maps",
)
def plot_xco2_maps(result, ctx: PlotContext) -> None:
    """Column-averaged XCO2: GT | Pred | Diff."""
    data = _get_field_data(result)
    if data is None:
        return

    pred, gt = data["pred"], data["gt"]
    lat, lon = data["lat"], data["lon"]
    level_values = data["level_values"]

    # Column average
    if pred.ndim == 3 and level_values is not None:
        pred_xco2 = _pressure_weighted_column(pred, level_values)
        gt_xco2 = _pressure_weighted_column(gt, level_values)
    else:
        # 2D or no level_values: just use as-is or mean over last dim
        pred_xco2 = pred.mean(axis=-1) if pred.ndim == 3 else pred
        gt_xco2 = gt.mean(axis=-1) if gt.ndim == 3 else gt

    subplot_kw = {"projection": ccrs.Robinson()} if HAS_CARTOPY else {}
    fig, axes = ctx.subplot_grid(1, 3, subplot_kw=subplot_kw)
    _plot_map_row(axes, gt_xco2, pred_xco2, "XCO2", lat=lat, lon=lon)
    fig.tight_layout()
    ctx.savefig(fig, "xco2_maps")


@register_plot(
    name="lat_height",
    categories=["always"],
    description="Zonal-mean latitude-height cross-section",
)
def plot_lat_height(result, ctx: PlotContext) -> None:
    """Zonal-mean latitude-height: GT | Pred | Diff."""
    data = _get_field_data(result)
    if data is None:
        return

    pred, gt = data["pred"], data["gt"]
    lat = data["lat"]
    level_values = data["level_values"]

    # Need 3D fields for lat-height
    if pred.ndim < 3:
        return

    # Zonal mean: average over longitude (axis=1)
    pred_zm = pred.mean(axis=1)  # [nlat, nlev]
    gt_zm = gt.mean(axis=1)  # [nlat, nlev]
    diff_zm = np.abs(pred_zm - gt_zm)

    fig, axes = ctx.subplot_grid(1, 3)
    panels = [
        (gt_zm, "GT zonal mean"),
        (pred_zm, "Pred zonal mean"),
        (diff_zm, "|Diff| zonal mean"),
    ]

    for ax, (field, title) in zip(axes, panels):
        if lat is not None and level_values is not None:
            ax.pcolormesh(lat, level_values, field.T, shading="auto")
            ax.invert_yaxis()
            ax.set_xlabel("Latitude")
            ax.set_ylabel("Pressure (hPa)")
        else:
            ax.imshow(field.T, origin="lower", aspect="auto")
        ax.set_title(title, fontsize=8)

    fig.tight_layout()
    ctx.savefig(fig, "lat_height")
