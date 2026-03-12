"""Animate trajectory of predicted CO₂ samples (noise → target)."""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from matplotlib.animation import FuncAnimation

from neural_transport.plots.utilities.plot_utils import (
    decorate_earth,
    normalize_minmax,
    PROJECTION_MAP,
    parse_projections,
    save_animation,
)
from neural_transport.plots.utilities.cmaps import get_cmap_list

sns.set_theme()
sns.color_palette("crest", as_cmap=True)


def animate_trajectory(
    trajectory: xr.DataArray,
    sample_idx: int = 0,
    level_idx: int | None = 0,
    projection: ccrs.Projection | None = None,
    terrain: bool = False,
    grid: bool = True,
    land: bool = False,
    ocean: bool = False,
    borders: bool = False,
    lakes: bool = False,
    rivers: bool = False,
    cmap: str = "bone_r",
    figsize: tuple[int, int] | None = None,
    bias_hidden: bool = False,
    title: str = "Trajectory (noise → target)",
    fps: int = 6,
):
    """Animate the time evolution of a single trajectory sample."""

    da_sample = trajectory.isel(time=0, sample=sample_idx)

    if level_idx is None:
        da_sample = da_sample.mean(dim="level")
        label_bar = "XCO₂ (normalized)"
    else:
        da_sample = da_sample.isel(level=level_idx)
        label_bar = "CO₂ (normalized)"

    if bias_hidden:
        da_sample = normalize_minmax(da_sample)

    if projection is None:
        projection = ccrs.PlateCarree()

    lat, lon = da_sample.sizes["lat"], da_sample.sizes["lon"]

    if figsize is None:
        panel_width = 6
        panel_height = panel_width * lat / lon
        figsize = (panel_width, panel_height)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = plt.axes(projection=projection)

    data_to_plot = da_sample.values
    if data_to_plot.ndim == 3:
        data_to_plot = data_to_plot if level_idx is not None else data_to_plot.mean(axis=-1)

    vmin, vmax = np.nanmin(data_to_plot), np.nanmax(data_to_plot)

    im = ax.pcolormesh(
        da_sample["lon"],
        da_sample["lat"],
        da_sample.isel(trajectory_steps=0).values,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        rasterized=True,
    )

    if not isinstance(projection, ccrs.PlateCarree):
        decorate_earth(
            ax,
            terrain=terrain,
            grid=grid,
            land=land,
            ocean=ocean,
            borders=borders,
            lakes=lakes,
            rivers=rivers,
        )
    elif grid:
        ax.gridlines(draw_labels=False)

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    step_text = ax.text(
        0.05,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
    )

    # Horizontal colorbar
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.12, shrink=0.6)
    cbar.set_label(label_bar, fontsize=12, labelpad=5)

    fig.suptitle(title, fontsize=16, fontweight="bold")

    nsteps = da_sample.sizes["trajectory_steps"]

    def update(frame):
        map_data = da_sample.isel(trajectory_steps=frame).values
        im.set_array(map_data.ravel())
        t = frame / (nsteps - 1)
        step_text.set_text(f"t={t:.2f}")
        return im, step_text

    anim = FuncAnimation(
        fig,
        update,
        frames=nsteps,
        interval=1000 / fps,
        blit=False,
    )

    return anim


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Animate CO₂ trajectory.")
    parser.add_argument(
        "--samples_path",
        type=str,
        required=True,
        help="Path to .zarr or .nc file containing trajectory predictions.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for saved animation.",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of the sample to animate.",
    )
    parser.add_argument(
        "--level_idx",
        type=str,
        default=None,
        help="Vertical level index to plot (or 'None' for mean).",
    )
    parser.add_argument(
        "--use_ipcc",
        action="store_true",
        help="Use IPCC colormap.",
    )
    parser.add_argument(
        "--use_selected",
        action="store_true",
        help="Use selected colormap.",
    )
    parser.add_argument(
        "--bias_hidden",
        action="store_true",
        help="Use min-max normalization.",
    )
    parser.add_argument(
        "--projections",
        nargs="*",
        default=["PlateCarree"],
        help=f"List of projections. Available: {', '.join(PROJECTION_MAP.keys())}",
    )

    args = parser.parse_args()

    level_idx = None if args.level_idx in ("None", "none", "", None) else int(args.level_idx)

    # Load predictions
    path = Path(args.samples_path)
    samples = xr.open_zarr(path) if path.suffix == ".zarr" else xr.open_dataset(path)

    cmaps = get_cmap_list(args.use_ipcc, args.use_selected)
    cmap = cmaps[0]

    projections = parse_projections(args.projections)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for proj in projections:
        anim = animate_trajectory(
            trajectory=samples.trajectory,
            sample_idx=args.sample_idx,
            level_idx=level_idx,
            projection=proj,
            terrain=False,
            grid=True,
            land=False,
            ocean=False,
            borders=False,
            lakes=False,
            rivers=False,
            cmap=cmap,
            bias_hidden=args.bias_hidden,
            title="Trajectory (noise → target)",
        )
        proj_name = proj.__class__.__name__
        save_animation(anim, args.out_dir, f"trajectory_animation_{args.sample_idx}_{proj_name}", formats=["mp4"])