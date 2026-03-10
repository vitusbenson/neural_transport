"""Utility functions for plotting."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import torch
import xarray as xr

from neural_transport.datamodule import CarbonDataModule


def save_figure(fig, out_dir, filename, imgformats=["svg", "png", "pdf"], dpi=300):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
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
