"""Plot CO₂ fields with multiple colormaps."""

import argparse
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

from neural_transport.plots.utilities.cmaps import get_cmap_list
from neural_transport.plots.utilities.plot_utils import mpl_rc_params, save_figure

sns.set_theme()  # Optional, for consistent style
sns.color_palette("crest", as_cmap=True)


def plot_cmaps(samples: xr.Dataset, cmap_list: list[str] | None = None, n_col: int = 2) -> plt.Figure:
    """Plot a set of colormaps applied to the same CO₂ field."""
    cmap_list = cmap_list or get_cmap_list()
    n_colors = len(cmap_list)
    n_rows = math.ceil(n_colors / n_col)

    da = samples["co2massmix"].isel(sample=2, level=0).compute()

    with mpl.rc_context(mpl_rc_params):
        fig, axes = plt.subplots(n_rows, n_col, figsize=(12, 3 * n_rows))
        axes = axes.ravel()

        for i, ax in enumerate(axes):
            if i < n_colors:
                cmap = cmap_list[i]
                da.plot(x="lon", y="lat", ax=ax, cmap=cmap, rasterized=True)
                ax.set_aspect("equal")
                cmap_name = getattr(cmap, "name", str(cmap))
                ax.set_title(f"{cmap_name}, surface CO₂")
                ax.set_xlabel("")
                ax.set_ylabel("")
            else:
                ax.axis("off")

        fig.tight_layout()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CO₂ field with various colormaps.")
    parser.add_argument("--samples_path", type=str, required=True, help="Path to .zarr or .nc file containing samples.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for saved plots.")
    parser.add_argument(
        "--use_selected", action="store_true", help="Use the full cmap_selected list instead of default_cmap."
    )
    parser.add_argument("--use_ipcc", action="store_true", help="Use IPCC colormaps instead of default or selected.")
    args = parser.parse_args()

    # Load sample dataset
    if args.samples_path.endswith(".zarr"):
        samples = xr.open_zarr(args.samples_path)
    else:
        samples = xr.open_dataset(args.samples_path)

    fig = plot_cmaps(samples, cmap_list=get_cmap_list(args.use_ipcc, args.use_selected))

    save_figure(fig, args.out_dir, "colormap_comparison", imgformats=["pdf"])
