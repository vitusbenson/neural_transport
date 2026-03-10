"""Plot CO₂ trajectory samples on map projections."""

import argparse
import contextlib
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

from neural_transport.plots.utilities.cmaps import get_cmap_list
from neural_transport.plots.utilities.plot_utils import PROJECTION_MAP, decorate_earth, parse_projections, save_figure

sns.set_theme()  # Optional seaborn style
sns.color_palette("crest", as_cmap=True)


def plot_samples_projection(
    traj: xr.DataArray,
    projection: ccrs.Projection | None = None,
    terrain: bool = False,
    grid: bool = False,
    land: bool = False,
    ocean: bool = False,
    borders: bool = False,
    lakes: bool = False,
    rivers: bool = False,
    n_samples: int = 2,
    ncol: int = 2,
    sample_indices: list[int] | None = None,
    level_idx: int = 0,
    cmaps: list[str] | None = None,
    seed: int = 7,
    figsize: tuple | None = None,
    title: str = "Sample Projections",
) -> plt.Figure:
    """Plot selected/random samples at the last time step on a global projection."""
    for dim in ("sample", "time", "lat", "lon"):
        if dim not in traj.dims:
            raise ValueError(f"traj must have '{dim}' dimension")

    if sample_indices is not None:
        sample_indices = list(sample_indices)[:n_samples]
        if len(sample_indices) < n_samples:
            raise ValueError("sample_indices shorter than n_samples")
    else:
        rng = np.random.default_rng(seed)
        n_available = traj.sizes["sample"]
        if n_samples > n_available:
            raise ValueError("n_samples > available samples")
        sample_indices = list(rng.choice(n_available, size=n_samples, replace=False))

    if cmaps is None:
        cmaps = ["bone_r"] * n_samples
    elif len(cmaps) < n_samples:
        cmaps = cmaps + ["bone_r"] * (n_samples - len(cmaps))

    if projection is None:
        projections = [ccrs.PlateCarree()] * n_samples
    elif isinstance(projection, list):
        if len(projection) < n_samples:
            projections = projection + [projection[-1]] * (n_samples - len(projection))
        else:
            projections = projection[:n_samples]
    else:
        projections = [projection] * n_samples

    last_time = traj.sizes["time"] - 1

    nrow = int(np.ceil(n_samples / ncol))
    aspect = traj.sizes["lat"] / traj.sizes["lon"]
    panel_width = 8.0
    panel_height = panel_width * aspect
    if figsize is None:
        figsize = (panel_width * ncol, (panel_height + 0.6) * nrow)

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    axes_flat = axes.ravel()

    for i, (sample_idx, cmap, proj) in enumerate(zip(sample_indices, cmaps, projections, strict=False)):
        ax = fig.add_subplot(axes_flat[i].get_subplotspec(), projection=proj)
        axes_flat[i].remove()  # remove placeholder

        da_sample = traj.isel(sample=sample_idx, level=level_idx, time=last_time)
        with contextlib.suppress(Exception):
            da_sample = da_sample.compute()

        decorate_earth(
            ax, terrain=terrain, grid=grid, land=land, ocean=ocean, borders=borders, lakes=lakes, rivers=rivers
        )

        mapable = da_sample.plot(
            ax=ax, cmap=cmap, add_colorbar=False, add_labels=False, transform=ccrs.PlateCarree(), rasterized=True
        )

        ax.set_title(f"Sample {sample_idx}, {proj.__class__.__name__}", fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

        cbar = fig.colorbar(mapable, ax=ax, fraction=0.025, pad=0.04)
        cbar.ax.set_ylabel("CO₂ (normalized)", rotation=270, labelpad=15)

    for j in range(len(sample_indices), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CO₂ trajectory samples on map projections.")
    parser.add_argument("--samples_path", type=str, required=True, help="Path to .zarr or .nc file containing samples.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for saved plots.")
    parser.add_argument(
        "--use_selected", action="store_true", help="Use the full cmap_selected list instead of default_cmap."
    )
    parser.add_argument("--use_ipcc", action="store_true", help="Use IPCC colormaps instead of default or selected.")
    parser.add_argument("--n_samples", type=int, default=4, help="Number of trajectory samples to plot.")
    parser.add_argument("--level_idx", type=int, default=0, help="Level index to plot.")
    parser.add_argument("--ncol", type=int, default=2, help="Number of columns in subplot grid.")
    parser.add_argument(
        "--projections",
        nargs="*",
        default=["PlateCarree"],
        help=f"List of projections. Available: {', '.join(PROJECTION_MAP.keys())}",
    )
    args = parser.parse_args()

    path = Path(args.samples_path)
    samples = xr.open_zarr(path) if path.suffix == ".zarr" else xr.open_dataset(path)

    cmaps = get_cmap_list(args.use_ipcc, args.use_selected)
    cmaps = cmaps[: args.n_samples]

    projections = parse_projections(args.projections)

    fig = plot_samples_projection(
        samples.trajectory,
        projection=projections,
        terrain=False,
        grid=True,
        land=False,
        ocean=False,
        borders=False,
        lakes=False,
        rivers=False,
        n_samples=args.n_samples,
        ncol=args.ncol,
        level_idx=args.level_idx,
        cmaps=cmaps,
        title="CO₂ Projection Samples",
    )

    save_figure(fig, args.out_dir, "samples_projection", imgformats=["pdf"])
