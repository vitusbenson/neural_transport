import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import xarray as xr
import xrft
from torch.utils.tensorboard.writer import figure_to_image

from neural_transport.tools.conversion import *
from neural_transport.tools.xarray_helper import tensor_to_xarray

mplstyle.use("fast")
warnings.filterwarnings("ignore", category=UserWarning)


def create_val_step_dataset(
    preds, batch, dataset="egg4", grid="latlon1", vertical_levels="l10"
):
    to_xarray = partial(
        tensor_to_xarray, dataset=dataset, grid=grid, vertical_levels=vertical_levels
    )

    return xr.Dataset(
        {f"{k}_pred": to_xarray(v) for k, v in preds.items() if v[0].numel() > 200}
        | {k: to_xarray(v) for k, v in batch.items() if v[0].numel() > 200}
    )


def plot_icon_grid(ds, var, vmin=None, vmax=None, nstep=None):
    vmin = ds[var].min().compute() if vmin is None else vmin
    vmax = ds[var].max().compute() if vmax is None else vmax
    nstep = 31 if nstep is None else nstep

    levels = np.linspace(vmin, vmax, nstep)

    fig, ax = plt.subplots(
        figsize=(10, 10), subplot_kw=dict(projection=ccrs.Robinson())
    )
    cmap = plt.get_cmap("Spectral_r", len(levels))
    ax.set_global()

    ax.gridlines(draw_labels=True, linewidth=0.5, color="dimgray", alpha=0.4, zorder=2)

    ax.coastlines(linewidth=0.5, zorder=2)
    cnf = ax.tricontourf(
        np.degrees(ds.clon),
        np.degrees(ds.clat),
        ds[var],
        levels=levels,
        vmin=vmin,
        vmax=vmax,
        extend="both",
        cmap=cmap,
        zorder=0,
        transform=ccrs.PlateCarree(),
    )

    cbar_ax = fig.add_axes([0.2, 0.25, 0.6, 0.015], autoscalex_on=True)  # -- x,y,w,h
    cbar = fig.colorbar(cnf, cax=cbar_ax, orientation="horizontal")
    plt.setp(cbar.ax.get_xticklabels()[::2], visible=False)
    cbar.set_label(var)

    return fig


def plot_atmospheric_layer_icon(ds, sample_idx, layer_idx, vari_idx=0, time_idx=0):
    ds = ds.isel(batch=sample_idx, level=layer_idx, vari=vari_idx, time=time_idx)

    with mpl.rc_context(
        {
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
        }
    ):
        fig, axs = plt.subplots(
            2, 2, figsize=(11, 5), subplot_kw=dict(projection=ccrs.Robinson())
        )

        vmin = ds.targ_t1.quantile(0.05)  # (ds.targ_t1.min()//5)*5-5
        vmax = ds.targ_t1.quantile(0.95)  # (ds.targ_t1.max()//5)*5+5
        if vmax <= vmin:
            vmax = vmin + 0.01
        nstep = 51

        levels = np.linspace(vmin, vmax, nstep)

        cmap = plt.get_cmap("Spectral_r", len(levels))

        targ_kwargs = dict(
            levels=levels,
            vmin=vmin,
            vmax=vmax,
            extend="both",
            cmap=cmap,
            zorder=0,
            transform=ccrs.PlateCarree(),
            # transform_first=True,
        )

        ax = axs[0, 0]
        ax.set_global()

        gl = ax.gridlines(
            draw_labels=True, linewidth=0.5, color="dimgray", alpha=0.4, zorder=2
        )
        gl.xlabel_style = {"size": 8, "color": "dimgray"}
        gl.ylabel_style = {"size": 8, "color": "dimgray"}
        gl.bottom_labels = False
        gl.right_labels = False

        ax.coastlines(linewidth=0.5, zorder=2)
        cnf = ax.tricontourf(
            np.degrees(ds.clon), np.degrees(ds.clat), ds.targ_t1, **targ_kwargs
        )
        ax.set_title("Target")

        ax = axs[0, 1]
        ax.set_global()

        gl = ax.gridlines(
            draw_labels=True, linewidth=0.5, color="dimgray", alpha=0.4, zorder=2
        )
        gl.xlabel_style = {"size": 8, "color": "dimgray"}
        gl.ylabel_style = {"size": 8, "color": "dimgray"}
        gl.bottom_labels = False
        gl.right_labels = False

        ax.coastlines(linewidth=0.5, zorder=2)
        cnf = ax.tricontourf(
            np.degrees(ds.clon), np.degrees(ds.clat), ds.pred_t1, **targ_kwargs
        )
        ax.set_title("Prediction")

        cbar_conc = plt.colorbar(
            cnf, ax=axs[0, :], shrink=0.9
        )  # , orientation='horizontal')

        max_delta = abs(ds.targ_t1 - ds.targ_t).max()
        max_delta = min(
            (max_delta * 1.1 if max_delta < 1 else (max_delta // 1) + 1), 10
        )
        if max_delta < 1e-6:
            max_delta = 1e-6

        levels = np.linspace(-max_delta, max_delta, nstep)

        cmap = plt.get_cmap("RdBu_r", len(levels))

        delta_kwargs = dict(
            levels=levels,
            vmin=-max_delta,
            vmax=max_delta,
            extend="both",
            cmap=cmap,
            zorder=0,
            transform=ccrs.PlateCarree(),
            # transform_first=True,
        )

        ax = axs[1, 0]
        ax.set_global()

        gl = ax.gridlines(
            draw_labels=True, linewidth=0.5, color="dimgray", alpha=0.4, zorder=2
        )
        gl.xlabel_style = {"size": 8, "color": "dimgray"}
        gl.ylabel_style = {"size": 8, "color": "dimgray"}
        gl.bottom_labels = False
        gl.right_labels = False

        ax.coastlines(linewidth=0.5, zorder=2)
        cnf = ax.tricontourf(
            np.degrees(ds.clon),
            np.degrees(ds.clat),
            (ds.targ_t1 - ds.targ_t),
            **delta_kwargs,
        )
        ax.set_title("Delta Target")

        ax = axs[1, 1]
        ax.set_global()

        gl = ax.gridlines(
            draw_labels=True, linewidth=0.5, color="dimgray", alpha=0.4, zorder=2
        )
        gl.xlabel_style = {"size": 8, "color": "dimgray"}
        gl.ylabel_style = {"size": 8, "color": "dimgray"}
        gl.bottom_labels = False
        gl.right_labels = False

        ax.coastlines(linewidth=0.5, zorder=2)
        cnf = ax.tricontourf(
            np.degrees(ds.clon),
            np.degrees(ds.clat),
            (ds.pred_t1 - ds.targ_t),
            **delta_kwargs,
        )
        ax.set_title("Delta Predicted")

        cbar_delta = plt.colorbar(
            cnf, ax=axs[1, :], shrink=0.9
        )  # , orientation='horizontal')

    return fig


def plot_atmospheric_layer(ds, sample_idx, layer_idx, vari_idx=0, time_idx=0):
    ds = ds.isel(batch=sample_idx, level=layer_idx, vari=vari_idx, time=time_idx)

    with mpl.rc_context(
        {
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
        }
    ):
        fig, axs = plt.subplots(
            2, 2, dpi=300, subplot_kw=dict(projection=ccrs.Robinson()), figsize=(8, 5)
        )

        vmin = ds.targ_t1.quantile(0.05)  # (ds.targ_t1.min()//5)*5-5
        vmax = ds.targ_t1.quantile(0.95)  # (ds.targ_t1.max()//5)*5+5
        if vmax <= vmin:
            vmax = vmin + 0.01
        nstep = 51

        levels = np.linspace(vmin, vmax, nstep)

        cmap = plt.get_cmap("Spectral_r", len(levels))

        targ_kwargs = dict(
            levels=levels,
            vmin=vmin,
            vmax=vmax,
            extend="both",
            cmap=cmap,
            zorder=0,
            add_colorbar=False,
            transform=ccrs.PlateCarree(),
        )

        # targ_kwargs = dict(cbar_kwargs=dict(shrink=0.5, label = "[ppm]"), vmin = (ds.targ_t1.min()//5)*5-5, vmax = (ds.targ_t1.max()//5)*5+5)

        cnf = ds.targ_t1.plot(ax=axs[0, 0], **targ_kwargs)
        axs[0, 0].set_title("Target")

        ds.pred_t1.plot(ax=axs[0, 1], **targ_kwargs)
        axs[0, 1].set_title("Prediction")

        cbar_conc = plt.colorbar(cnf, ax=axs[0, :], shrink=0.9)

        max_delta = abs(ds.targ_t1 - ds.targ_t).max()
        max_delta = min(
            (max_delta * 1.1 if max_delta < 1 else (max_delta // 1) + 1), 10
        )
        if max_delta < 1e-6:
            max_delta = 1e-6

        levels = np.linspace(-max_delta, max_delta, nstep)

        cmap = plt.get_cmap("RdBu_r", len(levels))

        delta_kwargs = dict(
            levels=levels,
            vmin=-max_delta,
            vmax=max_delta,
            extend="both",
            cmap=cmap,
            zorder=0,
            add_colorbar=False,
            transform=ccrs.PlateCarree(),
        )

        cnf = (ds.targ_t1 - ds.targ_t).plot(ax=axs[1, 0], **delta_kwargs)
        axs[1, 0].set_title("Delta Target")

        (ds.pred_t1 - ds.targ_t).plot(ax=axs[1, 1], **delta_kwargs)
        axs[1, 1].set_title("Delta Predicted")

        cbar_delta = plt.colorbar(cnf, ax=axs[1, :], shrink=0.9)

        for ax in axs.flatten():
            ax.set_global()
            gl = ax.gridlines(
                draw_labels=True, linewidth=0.5, color="dimgray", alpha=0.4, zorder=2
            )
            gl.xlabel_style = {"size": 8, "color": "dimgray"}
            gl.ylabel_style = {"size": 8, "color": "dimgray"}
            gl.bottom_labels = False
            gl.right_labels = False
            ax.coastlines(linewidth=0.5, zorder=2)
        # fig.tight_layout()

    return fig


def plot_zonal_mean(ds, sample_idx, vari_idx=0, time_idx=0):
    with mpl.rc_context(
        {
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
        }
    ):
        fig, axs = plt.subplots(2, 2, dpi=300, figsize=(8, 5))

        ds = ds.isel(batch=sample_idx, vari=vari_idx, time=time_idx)

        targ_kwargs = dict(
            x="lat",
            y="level",
            cbar_kwargs=dict(shrink=0.7, label="[ppm]"),
            vmin=(ds.targ_t1.mean("lon").min() // 5) * 5 - 5,
            vmax=(ds.targ_t1.mean("lon").max() // 5) * 5 + 5,
        )

        ds.targ_t1.mean("lon").plot(ax=axs[0, 0], **targ_kwargs)
        axs[0, 0].set_title("Target")

        ds.pred_t1.mean("lon").plot(ax=axs[0, 1], **targ_kwargs)
        axs[0, 1].set_title("Prediction")

        max_delta = abs((ds.targ_t1 - ds.targ_t).mean("lon")).max()
        max_delta = min(
            (max_delta * 1.1 if max_delta < 1 else (max_delta // 1) + 1), 10
        )
        delta_kwargs = dict(
            x="lat",
            y="level",
            cbar_kwargs=dict(shrink=0.7, label="[ppm]"),
            norm=colors.SymLogNorm(linthresh=0.01),
            vmin=-max_delta,
            vmax=max_delta,
            cmap="RdBu_r",
        )

        (ds.targ_t1 - ds.targ_t).mean("lon").plot(ax=axs[1, 0], **delta_kwargs)
        axs[1, 0].set_title("Delta Target")

        (ds.pred_t1 - ds.targ_t).mean("lon").plot(ax=axs[1, 1], **delta_kwargs)
        axs[1, 1].set_title("Delta Predicted")

        fig.tight_layout()

    return fig


def plot_zonal_spectrum(ds, layer_idx, time_idx=0, vari_idx=0):
    ds = ds.isel(level=layer_idx, time=time_idx, vari=vari_idx)

    Fpred = xrft.fft(ds.pred_t1, dim="lon", real_dim="lon")
    Ftarg = xrft.fft(ds.targ_t1, dim="lon", real_dim="lon")

    with mpl.rc_context(
        {
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
        }
    ):
        fig = plt.figure(dpi=300, figsize=(5, 4))
        ax = fig.add_subplot(111)
        xr.concat(
            [abs(Ftarg).mean(["lat", "batch"]), abs(Fpred).mean(["lat", "batch"])],
            dim=["targ", "pred"],
        ).plot(yscale="log", hue="concat_dim", ax=ax)
        ax.set_title("Zonal Power Spectrum")

        fig.tight_layout()

    return fig


def add_figure_to_logger(ds, sample_idx, layer_idx, name, grid="latlon5.625"):
    try:
        if grid.startswith("latlon"):
            return name, figure_to_image(
                plot_atmospheric_layer(ds, sample_idx=sample_idx, layer_idx=layer_idx),
                close=True,
            )
        else:
            return name, figure_to_image(
                plot_atmospheric_layer_icon(
                    ds, sample_idx=sample_idx, layer_idx=layer_idx
                ),
                close=True,
            )
    except Exception as e:
        print(f"Could not plot {name}: {e}")
        return name, None


def progress_indicator(future):
    print(".", end="", flush=True)


def plots_val_step(
    logger_experiment,
    current_epoch,
    preds,
    batch,
    variables=["co2molemix"],
    layer_idxs=[0, 1, 9, 18],
    batch_idx=0,
    n_samples=8,
    dataset="egg4",
    grid="latlon1",
    vertical_levels="l10",
    max_workers=32,
    plot_every_n_epochs=1,
):
    ds = create_val_step_dataset(
        preds, batch, dataset=dataset, grid=grid, vertical_levels=vertical_levels
    )

    for molecule in ["ch4", "co2"]:
        if f"{molecule}molemix" in variables:
            ds[f"{molecule}molemix"] = massmix_to_molemix(ds[f"{molecule}massmix"])
            ds[f"{molecule}molemix_next"] = massmix_to_molemix(
                ds[f"{molecule}massmix_next"]
            )
            ds[f"{molecule}molemix_pred"] = massmix_to_molemix(
                ds[f"{molecule}massmix_pred"]
            )

    try:
        outpath = Path(logger_experiment.log_dir) / "val_ds" / f"ds_{current_epoch}.nc"
        outpath.parent.mkdir(exist_ok=True, parents=True)
        if outpath.exists():
            outpath.unlink()
        ds.to_netcdf(outpath)
    except TypeError:
        print("Not saving Val Set")

    if current_epoch % plot_every_n_epochs != 0:
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for variable in variables:
            for layer_idx in layer_idxs:
                for sample_idx in range(min(len(ds.batch), n_samples)):
                    curr_ds = xr.Dataset(
                        {
                            "targ_t": ds[f"{variable}"],
                            "targ_t1": ds[f"{variable}_next"],
                            "pred_t1": ds[f"{variable}_pred"],
                        }
                    )
                    futures.append(
                        executor.submit(
                            add_figure_to_logger,
                            curr_ds,
                            sample_idx,
                            layer_idx,
                            f"{variable} Layer {layer_idx} sample {sample_idx} batch {batch_idx}",
                            grid,
                        )
                    )

        for future in futures:
            name, img = future.result()
            if img is not None:
                logger_experiment.add_image(name, img, current_epoch)
            
