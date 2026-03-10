"""Plot predicted vs. ground truth CO₂ samples to guess."""

import argparse
import contextlib
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import xarray as xr
from matplotlib import gridspec

from neural_transport.plots.utilities.cmaps import get_cmap_list
from neural_transport.plots.utilities.plot_utils import (
    PROJECTION_MAP,
    decorate_earth,
    load_carbontracker_tests,
    normalize_minmax,
    normalize_tests,
    parse_projections,
    save_figure,
)

sns.set_theme()
sns.color_palette("crest", as_cmap=True)


def plot_samples_with_comparison(
    traj: xr.DataArray,
    tests: torch.Tensor,
    projection: ccrs.Projection | None = ccrs.Robinson(),
    terrain: bool = False,
    grid: bool = True,
    land: bool = False,
    ocean: bool = False,
    borders: bool = False,
    lakes: bool = False,
    rivers: bool = False,
    n_samples: int = 1,
    sample_indices: list[int] | None = None,
    level_idx: int = 0,
    cmap: str = "bone_r",
    seed: int = 42,
    figsize: tuple[int, int] | None = None,
    bias_hidden: bool = False,
    title: str = "Which one is a generated sample, which one is ground truth?",
) -> plt.Figure:
    """Plot selected/random trajectory predictions against normalized ground truth."""
    if bias_hidden:
        tests_norm = normalize_minmax(tests)
        traj_norm = normalize_minmax(traj)
    else:
        tests_norm, _, _ = normalize_tests(tests)
        traj_norm = traj

    B, N, C = tests_norm.shape
    lat, lon = traj.sizes["lat"], traj.sizes["lon"]
    if N != lat * lon:
        raise ValueError(f"Expected N={lat * lon}, got N={N}")

    last_time = traj.sizes["time"] - 1

    if sample_indices is not None:
        sample_indices = list(sample_indices)[:n_samples]
    else:
        rng = np.random.default_rng(seed)
        sample_indices = list(rng.choice(min(traj.sizes["sample"], B), size=n_samples, replace=False))

    nrow, ncol = n_samples, 2
    aspect = lat / lon
    panel_width = 5.75
    panel_height = panel_width * aspect
    if figsize is None:
        figsize = (panel_width * ncol + 1.5, (panel_height + 0.6) * nrow)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrow, ncol, figure=fig, width_ratios=[1, 1], wspace=0.15, hspace=0.25)

    for i, sample_idx in enumerate(sample_indices):
        ax_pred = fig.add_subplot(gs[i, 0], projection=projection or None)
        ax_true = fig.add_subplot(gs[i, 1], projection=projection or None)

        da_sample = traj_norm.isel(sample=sample_idx, level=level_idx, time=last_time)
        with contextlib.suppress(Exception):
            da_sample = da_sample.compute()

        gt_field = tests_norm[sample_idx, :, level_idx].cpu().numpy().reshape(lat, lon)
        gt_da = xr.DataArray(
            gt_field, dims=("lat", "lon"), coords={"lat": traj["lat"], "lon": traj["lon"]}, name="co2massmix"
        )

        vmin = min(da_sample.min().item(), gt_field.min())
        vmax = max(da_sample.max().item(), gt_field.max())

        # --- Prediction ---
        if projection:
            decorate_earth(
                ax_pred, terrain=terrain, grid=grid, land=land, ocean=ocean, borders=borders, lakes=lakes, rivers=rivers
            )
            map_pred = da_sample.plot(
                ax=ax_pred,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
                add_labels=False,
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )
        else:
            map_pred = da_sample.plot(ax=ax_pred, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False, add_labels=False)
            ax_pred.set_aspect('equal', adjustable='box')

        ax_pred.set_xlabel("")
        ax_pred.set_ylabel("")

        # --- Ground truth ---
        if projection:
            decorate_earth(
                ax_true, terrain=terrain, grid=grid, land=land, ocean=ocean, borders=borders, lakes=lakes, rivers=rivers
            )
            gt_da.plot(
                ax=ax_true,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
                add_labels=False,
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )
        else:
            gt_da.plot(ax=ax_true, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False, add_labels=False)
            ax_true.set_aspect("equal", adjustable="box")

        ax_true.set_xlabel("")
        ax_true.set_ylabel("")

        # --- Colorbar for each sample pair ---
        cbar_ax = fig.add_axes([0.92, ax_pred.get_position().y0, 0.012, ax_pred.get_position().height])
        fig.colorbar(map_pred, cax=cbar_ax, label="CO₂ (normalized)")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.05)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare predicted vs. ground truth CO₂ samples.")
    parser.add_argument(
        "--samples_path", type=str, required=True, help="Path to .zarr or .nc file containing trajectory predictions."
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to CarbonTracker data directory.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for saved plots.")
    parser.add_argument(
        "--projections",
        nargs="*",
        default=["Robinson"],
        help=f"List of projections. Available: {', '.join(PROJECTION_MAP.keys())}",
    )
    parser.add_argument("--grid", action="store_true", help="Draw gridlines on maps.")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to compare.")
    parser.add_argument("--level_idx", type=int, default=0, help="Vertical level index to plot.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection.")
    parser.add_argument(
        "--use_selected", action="store_true", help="Use the full cmap_selected list instead of default_cmap."
    )
    parser.add_argument("--use_ipcc", action="store_true", help="Use IPCC colormaps instead of default or selected.")
    parser.add_argument(
        "--bias_hidden", action="store_true", help="Use min-max normalization instead of bias-variance normalization."
    )
    args = parser.parse_args()

    path = Path(args.samples_path)
    samples = xr.open_zarr(path) if path.suffix == ".zarr" else xr.open_dataset(path)
    co2tests = load_carbontracker_tests(data_path=args.data_path)

    cmaps = get_cmap_list(args.use_ipcc, args.use_selected)
    cmaps = cmaps[: args.n_samples]

    projections = parse_projections(args.projections)

    for proj in projections:
        fig = plot_samples_with_comparison(
            traj=samples.trajectory,
            tests=co2tests,
            projection=proj,
            grid=args.grid,
            n_samples=args.n_samples,
            level_idx=args.level_idx,
            cmap=cmaps[0],
            seed=args.seed,
            bias_hidden=args.bias_hidden,
            title="Which one is a generated sample, which one is ground truth?",
        )
        proj_name = proj.__class__.__name__
        save_figure(
            fig,
            args.out_dir,
            f"samples_comparison_{'minmax_' if args.bias_hidden else ''}{proj_name}",
            imgformats=["pdf"],
            dpi=300,
        )
