"""Utility functions for plotting."""

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from neural_transport.datamodule import CarbonDataModule

# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_PROJECTION = ccrs.Robinson
DEFAULT_DATA_CRS = ccrs.PlateCarree()
DEFAULT_CMAP_FIELD = "Spectral_r"
DEFAULT_CMAP_ERROR = "RdBu_r"
DEFAULT_CMAP_SPREAD = "YlOrRd"
DEFAULT_DPI = 150
DEFAULT_IMGFORMATS = ["png", "pdf"]


def save_figure(fig, out_dir, filename, imgformats=None, dpi=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    if imgformats is None:
        imgformats = DEFAULT_IMGFORMATS
    if dpi is None:
        dpi = DEFAULT_DPI
    for fmt in imgformats:
        fig.savefig(out_dir / f"{filename}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


mpl_rc_params = {
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.titlesize": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8,
}


# ── Map plotting helpers ─────────────────────────────────────────────


def create_map_axes(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    projection: ccrs.Projection | None = None,
    **gridspec_kw,
) -> tuple[plt.Figure, np.ndarray]:
    """Create a figure with map axes using the given projection.

    Returns (fig, axes) where axes is always a 2D numpy array.
    """
    if projection is None:
        projection = DEFAULT_PROJECTION()
    if figsize is None:
        figsize = (4.5 * ncols, 2.5 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        subplot_kw={"projection": projection},
        gridspec_kw=gridspec_kw,
    )
    # Always return 2D array
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1 and nrows > 1:
        axes = axes.T
    return fig, axes


def plot_map(
    ax: plt.Axes,
    data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    cmap: str = DEFAULT_CMAP_FIELD,
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    title: str = "",
    colorbar: bool = False,
    cb_label: str = "",
) -> plt.cm.ScalarMappable:
    """Plot a 2D field on a map axis with Robinson projection.

    Args:
        ax: A GeoAxes with projection already set.
        data: [nlat, nlon] array.
        lat: [nlat] latitude values.
        lon: [nlon] longitude values.
        cmap: Colormap name.
        vmin, vmax: Color limits. If None, auto-computed from data.
        symmetric: If True, center colorbar at 0 (for error/difference maps).
        title: Axis title.
        colorbar: If True, add a colorbar.
        cb_label: Colorbar label.

    Returns:
        The pcolormesh mappable (for shared colorbars).
    """
    if vmin is None:
        vmin = float(np.nanpercentile(data, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(data, 98))

    if symmetric:
        absmax = max(abs(vmin), abs(vmax))
        vmin, vmax = -absmax, absmax

    im = ax.pcolormesh(
        lon,
        lat,
        data,
        transform=DEFAULT_DATA_CRS,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.3")
    if title:
        ax.set_title(title, fontsize=9)

    if colorbar:
        cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8, aspect=30)
        if cb_label:
            cb.set_label(cb_label, fontsize=7)
        cb.ax.tick_params(labelsize=6)

    return im


def decorate_earth(ax, terrain=False, grid=False, land=False, ocean=False, borders=False, lakes=False, rivers=False):
    """Add optional geographic features to an axis."""
    ax.coastlines()

    if grid:
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="dimgray", alpha=0.4, zorder=2)
        gl.xlabel_style = {"size": 8, "color": "dimgray"}
        gl.ylabel_style = {"size": 8, "color": "dimgray"}
        gl.bottom_labels = False
        gl.right_labels = False

    if terrain:
        ax.stock_img()
    if land:
        ax.add_feature(cfeature.LAND)
    if ocean:
        ax.add_feature(cfeature.OCEAN)
    if borders:
        ax.add_feature(cfeature.BORDERS, linestyle=':')
    if lakes:
        ax.add_feature(cfeature.LAKES)
    if rivers:
        ax.add_feature(cfeature.RIVERS)


PROJECTION_MAP = {
    "PlateCarree": ccrs.PlateCarree,
    "Robinson": ccrs.Robinson,
    "Mollweide": ccrs.Mollweide,
    "Aitoff": ccrs.Aitoff,
    "InterruptedGoodeHomolosine": ccrs.InterruptedGoodeHomolosine,
    "Orthographic": ccrs.Orthographic,
    "Mercator": ccrs.Mercator,
    "LambertCylindrical": ccrs.LambertCylindrical,
}


def parse_projections(names):
    """Convert list of projection names to ccrs projection instances."""
    if not names:
        return [ccrs.PlateCarree()]
    projections = []
    for name in names:
        if name not in PROJECTION_MAP:
            raise ValueError(f"Unknown projection: {name}. Available: {', '.join(PROJECTION_MAP.keys())}")
        projections.append(PROJECTION_MAP[name]())
    return projections


def normalize_tests(tests: torch.Tensor):
    """
    Normalize tensor [B, N, C] across spatial dimension N.

    Returns normalized tensor and the per-sample mean and std.
    """
    mean = tests.mean(dim=1, keepdim=True)
    std = tests.std(dim=1, keepdim=True)
    normalized = (tests - mean) / std
    return normalized, mean, std


def normalize_minmax(arr):
    """
    Normalize array using min-max scaling along the feature/spatial dimension.
    """
    if isinstance(arr, torch.Tensor):
        # Assume arr shape [B, N, C]
        min_val = arr.min(dim=1, keepdim=True).values
        max_val = arr.max(dim=1, keepdim=True).values
        normalized = (arr - min_val) / (max_val - min_val)
    elif isinstance(arr, xr.DataArray):
        # Assume dims include lat and lon
        min_val = arr.min(dim=("lat", "lon"))
        max_val = arr.max(dim=("lat", "lon"))
        min_val, _ = xr.broadcast(min_val, arr)
        max_val, _ = xr.broadcast(max_val, arr)
        normalized = (arr - min_val) / (max_val - min_val)
    else:
        raise TypeError("Input must be torch.Tensor or xarray.DataArray")
    return normalized


def load_carbontracker_tests(data_path=None) -> torch.Tensor:
    """Load and return CarbonTracker CO₂ ground truth tensor.

    Args:
        data_path: Path to CarbonTracker data directory. Must be provided.
    """
    if data_path is None:
        raise ValueError("data_path must be provided. Pass the path to the CarbonTracker data directory.")

    data_kwargs = dict(
        data_path=data_path,
        dataset="carbontracker",
        grid="latlon5.625",
        vertical_levels="l10",
        freq="6h",
        n_timesteps=1,
        batch_size_train=64,
        batch_size_pred=32,
        num_workers=32,
        val_rollout_n_timesteps=None,
        target_vars=["co2massmix"],
        compute=False,
    )
    dset = CarbonDataModule(**data_kwargs)
    dset.setup("fit")
    dl_val = dset.val_dataloader()
    tests = next(iter(dl_val))
    tests = {k: v.squeeze(1) if isinstance(v, torch.Tensor) else v for k, v in tests.items()}

    return tests["co2massmix"]  # Shape: [B, N, C]
