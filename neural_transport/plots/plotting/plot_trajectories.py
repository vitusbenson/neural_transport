"""Plot CO₂ trajectory samples."""

import argparse
import contextlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

from neural_transport.plots.utilities.plot_utils import save_figure
from neural_transport.plots.utilities.cmaps import get_cmap_list

sns.set_theme()  # Optional, for consistent style
sns.color_palette("crest", as_cmap=True)


def plot_trajectories(
    traj: xr.DataArray,
    n_samples: int = 2,
    different_samples: bool = True,
    sample_indices: list[int] | None=None,
    level_idx: int = 0,
    time_indices: list[int] | None=None,
    cmaps: list[str] | None=None,
    seed: int = 42,
    figsize: tuple[int, int] | None=None,
    title: str = "Sample Trajectories",
) -> plt.Figure:
    """Plot trajectories for selected/random samples."""
    for dim in ("sample", "time", "lat", "lon"):
        if dim not in traj.dims:
            raise ValueError(f"traj must have '{dim}' dimension")

    if time_indices is None:
        time_indices = list(range(10))

    # --- choose sample indices ---
    if different_samples:
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
    else:
        sample_indices = [64] * n_samples

    # --- colormaps ---
    nrow = len(sample_indices)
    ncol = len(time_indices)
    if cmaps is None:
        cmaps = ["bone_r"] * nrow
    elif len(cmaps) < nrow:
        cmaps = cmaps + ["bone_r"] * (nrow - len(cmaps))

    # --- figure size ---
    aspect = traj.sizes["lat"] / traj.sizes["lon"]
    panel_width = 3.0
    panel_height = panel_width * aspect
    if figsize is None:
        figsize = (panel_width * ncol, (panel_height + 0.5) * nrow)

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    last_maps = []

    for i, (sample_idx, cmap) in enumerate(zip(sample_indices, cmaps)):
        traj_sample = traj.isel(sample=sample_idx, level=level_idx)
        with contextlib.suppress(Exception):
            traj_sample = traj_sample.compute()

        for j, time_idx in enumerate(time_indices):
            ax = axes[i, j]
            traj_slice = traj_sample.isel(time=time_idx)
            mappable = traj_slice.plot(ax=ax, cmap=cmap, add_colorbar=False, add_labels=False, rasterized=True)

            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([0, 120, 240])
            ax.set_yticks([-45, 0, 45])

            if j > 0:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            ax.text(
                0.05, 0.93, f"t={time_idx}",
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5),
            )

        last_maps.append(mappable)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    for i, (sample_idx, mappable) in enumerate(zip(sample_indices, last_maps)):
        row_axes = axes[i, :]
        row_bottom = row_axes[0].get_position().y0
        row_top = row_axes[-1].get_position().y1
        row_height = row_top - row_bottom

        cbar_ax = fig.add_axes([0.92, row_bottom, 0.007, row_height])
        fig.colorbar(mappable, cax=cbar_ax, label="CO₂")

    if title:
        fig.suptitle(title, fontsize=20, fontweight="bold")

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CO₂ trajectory samples.")
    parser.add_argument("--samples_path", type=str, required=True, help="Path to .zarr or .nc file containing samples.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for saved plots.")
    parser.add_argument("--use_selected", action="store_true", help="Use the full cmap_selected list instead of default_cmap.")
    parser.add_argument("--use_ipcc", action="store_true", help="Use IPCC colormaps instead of default or selected.")
    parser.add_argument("--n_samples", type=int, default=2, help="Number of trajectory samples to plot.")
    parser.add_argument("--level_idx", type=int, default=0, help="Level index to plot.")
    parser.add_argument("--time_indices", type=int, nargs="+", default=[0, 5, 7, 9], help="Time indices to plot.")
    args = parser.parse_args()

    # Load dataset
    if args.samples_path.endswith(".zarr"):
        samples = xr.open_zarr(args.samples_path)
    else:
        samples = xr.open_dataset(args.samples_path)

    fig = plot_trajectories(
        traj=samples.trajectory,
        n_samples=args.n_samples,
        sample_indices=None,
        level_idx=args.level_idx,
        time_indices=args.time_indices,
        cmaps=get_cmap_list(args.use_ipcc, args.use_selected)
    )

    save_figure(fig, args.out_dir, "trajectories", imgformats=["pdf"])
