"""Plot trajectory time series of predicted CO₂ samples."""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from matplotlib import gridspec

from neural_transport.plots.utilities.cmaps import get_cmap_list
from neural_transport.plots.utilities.plot_utils import (
    PROJECTION_MAP,
    decorate_earth,
    normalize_minmax,
    parse_projections,
    save_figure,
)

sns.set_theme()
sns.color_palette("crest", as_cmap=True)


def plot_trajectory_timeseries(
    trajectory: xr.DataArray,
    sample_idx: int = 0,
    level_idx: int | None = 0,
    projection: ccrs.Projection | None = None,
    terrain: bool = False,
    grid: bool = False,
    land: bool = False,
    ocean: bool = False,
    borders: bool = False,
    lakes: bool = False,
    rivers: bool = False,
    cmap: str = "bone_r",
    figsize: tuple[int, int] | None = None,
    bias_hidden: bool = False,
    title: str = "Trajectory Time Series",
) -> plt.Figure:
    """Plot the time series of a single trajectory sample."""
    da_sample = trajectory.isel(sample=sample_idx)

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

    nsteps = da_sample.sizes["time"]
    lat, lon = da_sample.sizes["lat"], da_sample.sizes["lon"]

    ncols = nsteps // 2 if nsteps % 2 == 0 else (nsteps + 1) // 2
    nrows = 2 if nsteps > ncols else 1

    if figsize is None:
        panel_width = 3
        panel_height = panel_width * lat / lon
        figsize = (panel_width * ncols, panel_height * nrows)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.05, hspace=0.15)

    data_to_plot = da_sample.values
    if data_to_plot.ndim == 3:
        data_to_plot = data_to_plot if level_idx is not None else data_to_plot.mean(axis=-1)
    vmin, vmax = np.nanmin(data_to_plot), np.nanmax(data_to_plot)

    axes = []
    for i in range(nsteps):
        row = i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col], projection=projection)
        axes.append(ax)

        map_data = da_sample.isel(time=i).values

        im = ax.pcolormesh(
            da_sample["lon"],
            da_sample["lat"],
            map_data,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            rasterized=True,
        )

        if not isinstance(projection, ccrs.PlateCarree):
            decorate_earth(
                ax, terrain=terrain, grid=grid, land=land, ocean=ocean, borders=borders, lakes=lakes, rivers=rivers
            )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.text(
            0.05,
            0.95,
            f"t={i}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        )

    # Hide unused axes
    for i in range(nsteps, nrows * ncols):
        axes[i].set_visible(False)

    # Horizontal colorbar
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.05])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar_ax.set_xlabel(label_bar, fontsize=12, labelpad=10)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.05)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CO₂ trajectory time series.")
    parser.add_argument(
        "--samples_path", type=str, required=True, help="Path to .zarr or .nc file containing trajectory predictions."
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for saved plots.")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index of the sample to plot.")
    parser.add_argument(
        "--level_idx", type=str, default=None, help="Vertical level index to plot (or 'None' for mean)."
    )
    parser.add_argument("--use_ipcc", action="store_true", help="Use IPCC colormap.")
    parser.add_argument("--use_selected", action="store_true", help="Use selected colormap.")
    parser.add_argument("--bias_hidden", action="store_true", help="Use min-max normalization.")
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

    fig = plot_trajectory_timeseries(
        trajectory=samples.trajectory,
        sample_idx=args.sample_idx,
        level_idx=args.level_idx,
        projection=projections[0],
        terrain=False,
        grid=True,
        land=False,
        ocean=False,
        borders=False,
        lakes=False,
        rivers=False,
        cmap=cmap,
        bias_hidden=args.bias_hidden,
        title="Trajectory Time Series",
    )

    save_figure(fig, args.out_dir, "trajectory_timeseries", imgformats=["pdf"], dpi=300)
