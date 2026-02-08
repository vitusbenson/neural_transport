"""Plot predicted vs. ground truth CO₂ samples."""

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

from neural_transport.plots.utilities.plot_utils import (
    decorate_earth, load_carbontracker_tests, normalize_minmax,
    normalize_tests, save_figure, PROJECTION_MAP, parse_projections
)
from neural_transport.plots.utilities.cmaps import get_cmap_list

sns.set_theme()
sns.color_palette("crest", as_cmap=True)


def plot_samples_and_ground_truth(
    traj: xr.DataArray,
    tests: torch.Tensor,
    projection: ccrs.Projection | None=ccrs.Robinson(),
    terrain: bool=False,
    grid: bool=True,
    land: bool=False, ocean: bool=False, borders: bool=False, lakes: bool=False, rivers: bool=False,
    n_samples: int=2,
    sample_indices: list[int] | None=None,
    level_idx: int | None=None,
    cmap: str="bone_r",
    seed: int=42,
    bias_hidden: bool=False,
    title: str="Generated Samples vs. Ground Truth",
) -> plt.Figure:
    """Plot trajectory predictions against ground truth samples."""
    if bias_hidden:
        tests_norm = normalize_minmax(tests)
        traj_norm = normalize_minmax(traj)
    else:
        tests_norm, _, _ = normalize_tests(tests)
        traj_norm = traj

    B, N, C = tests_norm.shape
    lat, lon = traj.sizes["lat"], traj.sizes["lon"]
    if N != lat * lon:
        raise ValueError(f"Expected N={lat*lon}, got N={N}")

    last_time = traj.sizes["time"] - 1

    if sample_indices is not None:
        sample_indices = list(sample_indices)[:n_samples]
    else:
        rng = np.random.default_rng(seed)
        n_available = min(traj.sizes["sample"], B)
        n_draw = min(n_samples, n_available)
        sample_indices = list(rng.choice(n_available, size=n_draw, replace=False))

    nrow, ncol = n_samples, 2
    aspect = lat / lon
    panel_width = 5.75
    panel_height = panel_width * aspect
    figsize = (panel_width * ncol + 1.5, (panel_height + 0.6) * nrow)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrow, ncol, figure=fig, width_ratios=[1, 1], wspace=0.15, hspace=0.25)

    axes = [[fig.add_subplot(gs[i, j], projection=projection if projection else None) for j in range(2)]
            for i in range(nrow)]
    mapable = None

    for i, sample_idx in enumerate(sample_indices):
        ax_pred, ax_true = axes[i]

        # Prediction
        da_sample = traj_norm.isel(sample=sample_idx, time=last_time)
        with contextlib.suppress(Exception):
            da_sample = da_sample.compute()

        if level_idx is None:
            da_sample = da_sample.mean(dim="level")
            label_bar = "XCO₂ (normalized)"
        else:
            da_sample = da_sample.isel(level=level_idx)
            label_bar = "CO₂ (normalized)"

        # Ground truth
        if level_idx is None:
            gt_field = tests_norm[sample_idx, :, :].mean(axis=1).cpu().numpy().reshape(lat, lon)
        else:
            gt_field = tests_norm[sample_idx, :, level_idx].cpu().numpy().reshape(lat, lon)
        gt_da = xr.DataArray(gt_field, dims=("lat", "lon"),
                             coords={"lat": traj["lat"], "lon": traj["lon"]})

        # --- Plot prediction ---
        if projection:
            decorate_earth(ax_pred, terrain=terrain, grid=grid, land=land, ocean=ocean,
                           borders=borders, lakes=lakes, rivers=rivers)
            map_pred = da_sample.plot(ax=ax_pred, cmap=cmap, add_colorbar=False, add_labels=False,
                                      transform=ccrs.PlateCarree(), rasterized=True)
        else:
            map_pred = da_sample.plot(ax=ax_pred, cmap=cmap, add_colorbar=False, add_labels=False, rasterized=True)
            ax_pred.set_aspect("equal", adjustable="box")
        ax_pred.set_title(f"Sample {sample_idx}", fontsize=11)
        ax_pred.set_xlabel("")
        ax_pred.set_ylabel("")

        # --- Plot ground truth ---
        if projection:
            decorate_earth(ax_true, terrain=terrain, grid=grid, land=land, ocean=ocean,
                           borders=borders, lakes=lakes, rivers=rivers)
            gt_da.plot(ax=ax_true, cmap=cmap, add_colorbar=False, add_labels=False,
                                  transform=ccrs.PlateCarree(), rasterized=True)
        else:
            gt_da.plot(ax=ax_true, cmap=cmap, add_colorbar=False, add_labels=False, rasterized=True)
            ax_true.set_aspect("equal", adjustable="box")
        ax_true.set_xlabel("")
        ax_true.set_ylabel("")

        mapable = map_pred

    # Column titles
    axes[0][0].text(0.5, 1.25, "Prediction", fontsize=13, fontweight="bold",
                    ha="center", transform=axes[0][0].transAxes)
    axes[0][1].text(0.5, 1.25, "Ground Truth", fontsize=13, fontweight="bold",
                    ha="center", transform=axes[0][1].transAxes)

    # Shared horizontal colorbar
    cbar = fig.colorbar(mapable, ax=[ax for row in axes for ax in row],
                        orientation="horizontal", fraction=0.04, pad=0.08, aspect=25)
    cbar.ax.set_xlabel(label_bar, labelpad=10, fontsize=12)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.05)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare predicted vs. ground truth CO₂ samples.")
    parser.add_argument("--samples_path", type=str, required=True,
                        help="Path to .zarr or .nc file containing trajectory predictions.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to CarbonTracker data directory.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for saved plots.")
    parser.add_argument("--projections", nargs="*", default=["Robinson"],
                        help=f"List of projections. Available: {', '.join(PROJECTION_MAP.keys())}")
    parser.add_argument("--grid", action="store_true", help="Draw gridlines on maps.")
    parser.add_argument("--n_samples", type=int, default=2, help="Number of samples to compare.")
    parser.add_argument("--level_idx", type=int, default=None, help="Vertical level index to plot.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection.")
    parser.add_argument("--use_selected", action="store_true", help="Use the full cmap_selected list instead of default_cmap.")
    parser.add_argument("--use_ipcc", action="store_true", help="Use IPCC colormaps instead of default or selected.")
    parser.add_argument("--bias_hidden", action="store_true", help="Use min-max normalization instead of bias-variance normalization.")
    args = parser.parse_args()

    # Load predictions and ground truth
    path = Path(args.samples_path)
    samples = xr.open_zarr(path) if path.suffix == ".zarr" else xr.open_dataset(path)
    co2tests = load_carbontracker_tests(data_path=args.data_path)

    # Colormap selection
    cmaps = get_cmap_list(args.use_ipcc, args.use_selected)
    cmaps = cmaps[:args.n_samples]

    projections = parse_projections(args.projections)

    for proj in projections:
        fig = plot_samples_and_ground_truth(
            traj=samples.trajectory,
            tests=co2tests,
            projection=proj,
            grid=args.grid,
            n_samples=args.n_samples,
            level_idx=args.level_idx,
            cmap=cmaps[0],
            seed=args.seed,
            bias_hidden=args.bias_hidden,
            title="Generated Samples vs. Ground Truth",
        )
        proj_name = proj.__class__.__name__
        suffix = "minmax_" if args.bias_hidden else ""
        save_figure(fig, args.out_dir, f"samples_comparison_gt_{suffix}{proj_name}", imgformats=["pdf"], dpi=300)
