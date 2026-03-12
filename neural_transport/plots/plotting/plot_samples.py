"""Plot selected/random trajectory samples at the last time step."""

import argparse
import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

from neural_transport.plots.utilities.plot_utils import save_figure
from neural_transport.plots.utilities.cmaps import get_cmap_list

sns.set_theme()  # Optional seaborn style
sns.color_palette("crest", as_cmap=True)


def plot_samples(
    traj: xr.DataArray,
    n_samples: int = 2,
    ncol: int = 2,
    sample_indices: list[int] | None=None,
    level_idx: int = 0,
    cmaps: list[str] | None=None,
    seed: int = 42,
    figsize: tuple[int, int] | None=None,
    title: str = "Samples",
) -> plt.Figure:
    """Plot selected/random samples at the last integration step."""
    for dim in ("trajectory_steps", "sample", "time", "lat", "lon"):
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

    last_step = traj.sizes["trajectory_steps"] - 1

    nrow = int(np.ceil(n_samples / ncol))
    aspect = traj.sizes["lat"] / traj.sizes["lon"]
    panel_width = 6.0
    panel_height = panel_width * aspect
    if figsize is None:
        figsize = (panel_width * ncol, (panel_height + 0.6) * nrow)

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for i, (sample_idx, cmap) in enumerate(zip(sample_indices, cmaps)):
        ax = axes_flat[i]
        da_sample = traj.isel(time=0, sample=sample_idx, trajectory_steps=last_step, level=level_idx)
        with contextlib.suppress(Exception):
            da_sample = da_sample.compute()

        mappable = da_sample.plot(ax=ax, cmap=cmap, add_colorbar=False, add_labels=False, rasterized=True)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Sample {sample_idx}", fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.025, pad=0.04)
        cbar.ax.set_ylabel("CO₂ (normalized)", rotation=270, labelpad=15)

    for j in range(len(sample_indices), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    return fig, axes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot trajectory samples at the last time step.")
    parser.add_argument("--samples_path", type=str, required=True, help="Path to .zarr or .nc file containing samples.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for saved plots.")
    parser.add_argument("--use_selected", action="store_true", help="Use the full cmap_selected list instead of default_cmap.")
    parser.add_argument("--use_ipcc", action="store_true", help="Use IPCC colormaps instead of default or selected.")
    parser.add_argument("--use_ipcc_one", action="store_true", help="Use a single IPCC colormap.")
    parser.add_argument("--n_samples", type=int, default=2, help="Number of samples to plot.")
    parser.add_argument("--level_idx", type=int, default=0, help="Level index to plot.")
    parser.add_argument("--ncol", type=int, default=2, help="Number of columns in subplot grid.")
    args = parser.parse_args()

    path = Path(args.samples_path)
    samples = xr.open_zarr(path) if path.suffix == ".zarr" else xr.open_dataset(path)

    fig, _ = plot_samples(
        traj=samples.trajectory,
        n_samples=args.n_samples,
        sample_indices=list(range(args.n_samples)),
        ncol=args.ncol,
        level_idx=args.level_idx,
        cmaps=get_cmap_list(args.use_ipcc, args.use_ipcc_one, args.use_selected, n_samples=args.n_samples),
        title="Samples",
    )

    save_figure(fig, args.out_dir, "samples", imgformats=["pdf"])
