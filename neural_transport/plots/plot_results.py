from pathlib import Path

import cartopy.crs as ccrs
from matplotlib import gridspec
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import xrft
import xskillscore
from xmovie import Movie
from xmovie.core import convert_gif

from neural_transport.inference.analyse import freq_mean
from neural_transport.tools.conversion import (
    compute_xco2_via_ak,
    density_to_massmix,
    massmix_to_molemix,
    zonal_wavenumber_to_wavelength,
    km_per_gridcell,
)

from sklearn.decomposition import PCA

import torch

mpl_rc_params = {
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.titlesize": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8,
}


def plot_value_over_leadtime(
    da, ylabel="", ylim=[0, 1], thresh_value=None, figsize=(8, 5), freq="QS", **kwargs
):

    daf = (
        freq_mean(da, freq=freq).rename(time="days")
        # if freq else da.assign_coords().rename(time="days")
    )

    with mpl.rc_context(mpl_rc_params):
        sns.set_palette("Spectral", n_colors=len(da.level))
        daf.plot(hue="level", figsize=figsize)
        sns.move_legend(plt.gca(), loc="center left", bbox_to_anchor=(1, 0.5))
        if thresh_value is not None:
            plt.axhline(y=thresh_value, ls="--", color="black", zorder=0)

            invalid_days = (
                daf.min("level")
                .compute()
                .where(lambda x: x < thresh_value, drop=True)
                .days.values
            )
            min_days = (
                daf.days.values[-1] if len(invalid_days) == 0 else invalid_days[0]
            )
            plt.axvline(x=min_days, color="black", zorder=0, lw=0.5)
        else:
            min_days = 0

        plt.xticks([min_days if min_days < 20 else 0, 20, 40, 60, 80])
        plt.ylim(*ylim)
        plt.xlabel("Lead time [days]")
        plt.ylabel(ylabel)

        plt.tight_layout()
        fig = plt.gcf()

    return fig


def rmse(pred, targ, weights, dims=["lat", "lon"]):
    return ((pred - targ) ** 2 * weights).mean(dims) ** 0.5


def mae(pred, targ, weights, dims=["lat", "lon"]):
    return (np.abs(pred - targ) * weights).mean(dims)


def bias(pred, targ, weights, dims=["lat", "lon"]):
    return (pred * weights).mean(dims) - (targ * weights).mean(dims)


def r2(pred, targ, weights, dims=["lat", "lon"]):
    return (
        xskillscore.pearson_r(
            pred,
            targ,
            dim=dims,
            weights=weights.isel(**{d: 0 for d in weights.dims if d not in dims}),
        )
        ** 2
    )


def nse(pred, targ, weights, dims=["lat", "lon"]):
    return xskillscore.r2(
        pred,
        targ,
        dim=dims,
        weights=weights.isel(**{d: 0 for d in weights.dims if d not in dims}),
    )


def rel_mean(pred, targ, weights, dims=["lat", "lon"]):
    return (pred * weights).mean(dims) / (targ * weights).mean(dims)


def rel_std(pred, targ, weights, dims=["lat", "lon"]):
    return (pred * weights).std(dims) / (targ * weights).std(dims)


METRICS = dict(
    rmse=rmse, mae=mae, bias=bias, r2=r2, nse=nse, rel_mean=rel_mean, rel_std=rel_std
)
METRIC_LABELS = dict(
    rmse="RMSE",
    mae="MAE",
    bias="Bias",
    r2=r"$R^2$",
    nse="NSE",
    rel_mean="Pred Mean / Targ Mean",
    rel_std="Pred Std / Targ Std",
)
METRIC_LIMITS = dict(r2=[0, 1.1], nse=[-1, 1.1])


def get_metric_limits(metric, metric_pred):
    return METRIC_LIMITS.get(
        metric,
        (
            [-np.abs(metric_pred).quantile(0.98), np.abs(metric_pred).quantile(0.98)]
            if metric in ["bias"]
            else (
                [
                    1 - np.abs(1 - metric_pred).quantile(0.98),
                    1 + np.abs(1 - metric_pred).quantile(0.98),
                ]
                if metric in ["rel_mean", "rel_std"]
                else [0, metric_pred.quantile(0.98)]
            )
        ),
    )


METRIC_THRESH = dict(r2=0.9, nse=0.5, rel_mean=1.0, rel_std=0.9)


def compute_metric_over_samples(metric_func, pred, targ, weights, dims=None):
    """
    Computes a metric over a generative forecast (sample dimension) for single timestep.
    Returns the mean metric across samples.
    """
    if dims is None:
        dims = [d for d in ["lat", "lon", "time"] if d in pred.dims]
    
    targ_aligned = targ
    if "time" not in pred.dims and "time" in targ.dims:
        targ_aligned = targ.isel(time=0, drop=True)
        weights = weights.isel(time=0, drop=True) if "time" in weights.dims else weights
        dims = [d for d in dims if d != "time"]
    
    metrics = []
    for s in range(pred.sizes["sample"]):
        metrics.append(
            metric_func(
                pred.isel(sample=s).compute(),
                targ_aligned.compute(),
                weights,
                dims=dims,
            )
        )

    metric_mean = xr.concat(metrics, dim="sample").mean("sample")
    return metric_mean


def plot_metric_over_leadtime(pred, targ, metric, figsize=(8, 5), freq="QS"):
    weights = np.cos(np.deg2rad(targ.lat.compute()))
    _, weights = xr.broadcast(targ, weights)

    metric_func = METRICS[metric]
    
    metric_pred = metric_func(pred.compute(), targ.compute(), weights)

    ylim = get_metric_limits(metric, metric_pred)

    fig = plot_value_over_leadtime(
        metric_pred,
        ylabel=METRIC_LABELS[metric],
        ylim=ylim,
        thresh_value=METRIC_THRESH.get(metric, None),
        figsize=figsize,
        freq=freq,
    )

    return fig


def plot_value_over_space(da, clabel="", figsize=(8, 4), **kwargs):
    das = da.mean([d for d in da.dims if d not in ["lat", "lon"]])

    with mpl.rc_context(mpl_rc_params):
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(projection=ccrs.Robinson())

        das.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cbar_kwargs=dict(label=clabel, shrink=0.8),
            **kwargs,
        )
        ax.set_global()
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.5, color="dimgray", alpha=0.4, zorder=2
        )
        gl.xlabel_style = {"size": 8, "color": "dimgray"}
        gl.ylabel_style = {"size": 8, "color": "dimgray"}
        gl.bottom_labels = False
        gl.right_labels = False
        ax.coastlines(linewidth=0.5, zorder=2)

        plt.tight_layout()

    return fig


METRIC_CMAPS = dict(
    rmse="Spectral_r",
    mae="Spectral_r",
    bias="RdBu_r",
    r2="Spectral",
    nse="Spectral",
    rel_mean="RdBu_r",
    rel_std="RdBu_r",
)


def plot_metric_over_space(pred, targ, metric, figsize=(8, 4)):
    weights = np.cos(np.deg2rad(targ.lat.compute()))
    _, weights = xr.broadcast(targ, weights)

    metric_func = METRICS[metric]

    reduce_dims = []
    if "time" in pred.dims:
        reduce_dims.append("time")
    
    if "sample" in pred.dims:
        metric_pred = compute_metric_over_samples(metric_func, pred, targ, weights, dims=reduce_dims)
    else:
        metric_pred = metric_func(pred.compute(), targ.compute(), weights, dims=reduce_dims)

    ylim = get_metric_limits(metric, metric_pred)

    fig = plot_value_over_space(
        metric_pred,
        clabel=METRIC_LABELS[metric],
        cmap=METRIC_CMAPS[metric],
        vmin=ylim[0],
        vmax=ylim[1],
        figsize=figsize,
    )

    return fig


def plot_value_over_latheight(da, clabel="", figsize=(8, 4), **kwargs):
    das = da.mean([d for d in da.dims if d not in ["lat", "level"]])

    with mpl.rc_context(mpl_rc_params):
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot()

        old_level = das.level.values
        das["level"] = range(len(old_level))

        das.plot(
            x="lat",
            y="level",
            ax=ax,
            cbar_kwargs=dict(label=clabel, shrink=0.8),
            **kwargs,
        )

        plt.yticks(ticks=range(len(old_level)), labels=old_level.astype("int"))

        plt.tight_layout()

    return fig


def plot_metric_over_latheight(pred, targ, metric, figsize=(8, 4)):
    weights = np.cos(np.deg2rad(targ.lat.compute()))
    _, weights = xr.broadcast(targ, weights)

    metric_func = METRICS[metric]

    if "sample" in pred.dims:
        metric_pred = compute_metric_over_samples(metric_func, pred, targ, weights, dims=["time"])
    else:
        metric_pred = metric_func(pred.compute(), targ.compute(), weights, dims=["time"])

    ylim = get_metric_limits(metric, metric_pred)

    fig = plot_value_over_latheight(
        metric_pred,
        clabel=METRIC_LABELS[metric],
        figsize=figsize,
        cmap=METRIC_CMAPS[metric],
        vmin=ylim[0],
        vmax=ylim[1],
    )

    return fig


def get_zonal_spectrum(pred, targ):
    pred = pred.compute()
    targ = targ.compute()

    Fpred = xrft.fft(pred, dim="lon", real_dim="lon")
    Ftarg = xrft.fft(targ, dim="lon", real_dim="lon")

    Specpred = abs(Fpred).mean(["lat", "level"])
    Spectarg = abs(Ftarg).mean(["lat", "level"])

    return Specpred, Spectarg


def plot_zonal_spectrum_line(pred, targ, figsize=(8, 5), **kwargs):
    Specpred, Spectarg = get_zonal_spectrum(pred, targ)

    mean_dims_pred = [d for d in ["time", "sample"] if d in Specpred.dims]
    mean_dims_targ = [d for d in ["time", "sample"] if d in Spectarg.dims]
    Specpred_mean = Specpred.mean(mean_dims_pred) if mean_dims_pred else Specpred
    Spectarg_mean = Spectarg.mean(mean_dims_targ) if mean_dims_targ else Spectarg

    with mpl.rc_context(mpl_rc_params):
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot()

        xr.concat(
            [Specpred_mean, Spectarg_mean], dim=["Prediction", "Target"]
        ).rename({"concat_dim": "Variable"}).plot(yscale="log", hue="Variable", ax=ax)
        ax.set_title("Zonal Power Spectrum")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Power")

        plt.tight_layout()

    return fig


def get_zonal_spectrum_physical(pred, targ):
    pred = pred.compute()
    targ = targ.compute()

    # Mean latitude and spacing
    dx, circ_at_lat = km_per_gridcell(pred)

    # Assign lon coordinate in kilometers for xrft
    pred = pred.assign_coords(lon=pred["lon"] * dx)
    targ = targ.assign_coords(lon=targ["lon"] * dx)

    # Properly scaled FFT
    Fpred = xrft.fft(pred, dim="lon", true_phase=True, true_amplitude=True)
    Ftarg = xrft.fft(targ, dim="lon", true_phase=True, true_amplitude=True)
    Fpred = Fpred.isel(freq_lon=(Fpred["freq_lon"] > 0))
    Ftarg = Ftarg.isel(freq_lon=(Ftarg["freq_lon"] > 0))

    Specpred = abs(Fpred).mean(["lat", "level"])
    Spectarg = abs(Ftarg).mean(["lat", "level"])

    freq_km = Specpred["freq_lon"].values
    k_circ = freq_km * circ_at_lat
    Specpred = Specpred.assign_coords(k_circ=("freq_lon", k_circ))
    Spectarg = Spectarg.assign_coords(k_circ=("freq_lon", k_circ))

    return Specpred, Spectarg


def plot_zonal_spectrum_line_physical(pred, targ, figsize=(8, 5), **kwargs):
    """
    Plot the zonal power spectrum of prediction and target.
    The bottom x-axis shows frequency, the top x-axis shows wavelength in km.

    Parameters
    ----------
    pred, targ : xarray.DataArray
        Prediction and target fields with 'lon' dimension.
    figsize : tuple
        Figure size.
    """
    Specpred, Spectarg = get_zonal_spectrum_physical(pred, targ)

    mean_dims_pred = [d for d in ["time", "sample"] if d in Specpred.dims]
    mean_dims_targ = [d for d in ["time", "sample"] if d in Spectarg.dims]
    Specpred_mean = Specpred.mean(mean_dims_pred) if mean_dims_pred else Specpred
    Spectarg_mean = Spectarg.mean(mean_dims_targ) if mean_dims_targ else Spectarg

    with mpl.rc_context(mpl_rc_params):
        fig, ax = plt.subplots(figsize=figsize)

        # Plot
        xr.concat([Specpred_mean, Spectarg_mean], dim=["Prediction", "Target"]) \
          .rename({"concat_dim": "Variable"}) \
          .plot(x="k_circ", yscale="log", hue="Variable", ax=ax)

        ax.set_title("Zonal Power Spectrum")
        ax.set_xlabel("Zonal Wavenumber [cycles per Earth circumference]")
        ax.set_ylabel("Power")

        secax = ax.secondary_xaxis('top')
        secax.set_xlabel("Wavelength [km]")

        # Compute wavelength for frequencies (skip zero)
        lat_mean = float(pred["lat"].mean().values) if "lat" in pred.coords else 0.0
        bottom_ticks = ax.get_xticks()
        freq_nonzero = bottom_ticks[bottom_ticks > 0]
        wavelength_nonzero = zonal_wavenumber_to_wavelength(freq_nonzero, lat_mean)
       
        # Set nicely spaced ticks
        secax.set_xticks(freq_nonzero)
        # secax.set_xticklabels([f"{km:.2e}" for km in wavelength_nonzero]) # scientific notation
        secax.set_xticklabels([f"{km:.0f}" for km in wavelength_nonzero]) # non-scientific notation

        plt.tight_layout()

    return fig


def plot_zonal_spectrum_heatmap(pred, targ, figsize=(8, 5), freq="QS", **kwargs):
    Specpred, Spectarg = get_zonal_spectrum(pred, targ)

    rename_dict = {"concat_dim": "Variable", "freq_lon": "Frequency"}

    average_time = False
    if "time" not in Specpred.dims:
        average_time = True
    else:
        rename_dict["time"] = "Lead time [days]"
    
    Specpredf = freq_mean(Specpred, freq=freq)
    Spectargf = freq_mean(Spectarg, freq=freq, average_time=average_time)

    result = xr.concat([Specpredf, Spectargf], dim=["Prediction", "Target"])
    result = result.rename(rename_dict)

    with mpl.rc_context(mpl_rc_params):
        if "sample" in result.dims:
            result = result.mean("sample")
        if result.sizes.get("Lead time [days]", 0) > 1:
            result.plot(
                cmap="Spectral",
                norm=mpl.colors.LogNorm(),
                col="Variable",
                figsize=figsize,
                cbar_kwargs={"label": "Power"},
            )
        else:
            result.plot.line(
                hue="Variable",
                figsize=figsize,
            )
        fig = plt.gcf()
        plt.suptitle("Zonal Power Spectrum")

    return fig


def get_pred_targ_from_varname(preds, targs, varname):
    if varname.endswith("molemix"):
        if varname not in targs:
            if varname.replace("molemix", "massmix") not in targs:
                targs[varname] = massmix_to_molemix(
                    density_to_massmix(
                        targs[varname.replace("molemix", "density")],
                        targs["airdensity"],
                        ppm=True,
                    )
                )
            else:
                targs[varname] = massmix_to_molemix(
                    targs[varname.replace("molemix", "massmix")]
                )
            targs[varname].attrs = dict(units="ppm", long_name=varname)
        if varname not in preds:
            if varname.replace("molemix", "massmix") not in preds:
                preds[varname] = massmix_to_molemix(
                    density_to_massmix(
                        preds[varname.replace("molemix", "density")],
                        targs["airdensity"],
                        ppm=True,
                    )
                )
            else:
                preds[varname] = massmix_to_molemix(
                    preds[varname.replace("molemix", "massmix")]
                )

    elif varname.endswith("massmix") and varname not in targs:
        targs[varname] = density_to_massmix(
            targs[varname.replace("massmix", "density")]
        )

    pred = preds[varname].compute()
    targ = targs[varname].compute()

    if "sample" not in pred.dims:
        pred["time"] = targ["time"]
    
    pred["level"] = targ["level"]
    return pred, targ


def plot_metrics(
    preds,
    targs,
    out_dir,
    over_leadtime=True,
    over_space=True,
    over_latheight=True,
    zonal_spectrum=True,
    varnames=["co2molemix"],
    metrics=["rmse", "mae", "bias", "rel_mean", "rel_std"], # "r2", "nse",
    imgformats=["svg", "png", "pdf"],
):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    plt_fcts = []
    if over_leadtime and "sample" not in preds.dims:
        plt_fcts.append(["over_leadtime", plot_metric_over_leadtime])
    if over_space:
        plt_fcts.append(["over_space", plot_metric_over_space])
    if over_latheight:
        plt_fcts.append(["over_latheight", plot_metric_over_latheight])

    for varname in varnames:
        pred, targ = get_pred_targ_from_varname(preds, targs, varname)

        for metric in metrics:
            for plottype, plt_fct in plt_fcts:
                plt_fct(pred, targ, metric)

                for imgformat in imgformats:
                    plt.savefig(
                        out_dir / f"{varname}_{metric}_{plottype}.{imgformat}", dpi=300
                    )

                plt.close()

        if zonal_spectrum:
            plot_zonal_spectrum_line(pred, targ)

            for imgformat in imgformats:
                plt.savefig(
                    out_dir / f"{varname}_zonal_spectrum_line.{imgformat}", dpi=300
                )
            plt.close()

            plot_zonal_spectrum_line_physical(pred, targ)

            for imgformat in imgformats:
                plt.savefig(
                    out_dir / f"{varname}_zonal_spectrum_line_physical.{imgformat}", dpi=300
                )
            plt.close()

            plot_zonal_spectrum_heatmap(pred, targ)

            for imgformat in imgformats:
                plt.savefig(
                    out_dir / f"{varname}_zonal_spectrum_heatmap.{imgformat}", dpi=300
                )
            plt.close()

    return


def plot_3d_variable(da, fig, tt, *args, **kwargs):
    targ = da.sel(vari="targ").isel(time=tt)
    pred = da.sel(vari="pred").isel(time=tt)

    with mpl.rc_context(mpl_rc_params):
        axs = fig.subplots(
            3,
            3,
            subplot_kw=dict(projection=ccrs.Robinson()),
            gridspec_kw={"wspace": 0.05, "hspace": 0.05},
        )

        levels = np.linspace(
            kwargs.get("vmin", targ.min()),
            kwargs.get("vmax", targ.max()),
            kwargs.get("nstep", 21),
        )
        cmap = plt.get_cmap("Spectral_r", len(levels))

        vari_kwargs = dict(
            levels=levels,
            vmin=kwargs.get("vmin", targ.min()),
            vmax=kwargs.get("vmax", targ.max()),
            cmap=cmap,
            zorder=0,
            transform=ccrs.PlateCarree(),
        )

        delta_levels = np.linspace(
            -kwargs.get("max_delta", 1),
            kwargs.get("max_delta", 1),
            kwargs.get("nstep", 21),
        )
        delta_cmap = plt.get_cmap("RdBu_r", len(delta_levels))
        delta_kwargs = dict(
            levels=delta_levels,
            vmin=-kwargs.get("max_delta", 1),
            vmax=kwargs.get("max_delta", 1),
            cmap=delta_cmap,
            zorder=0,
            transform=ccrs.PlateCarree(),
        )

        for i, curr_da in enumerate([targ, pred, targ - pred]):
            for j, level in enumerate(kwargs.get("levels", [1, 8, 15])):
                ax = axs[i, j]

                cnf = curr_da.isel(level=level).plot(
                    ax=ax,
                    add_colorbar=False,
                    **(vari_kwargs if i < 2 else delta_kwargs),
                )

                if j == len(kwargs.get("levels", [1, 8, 15])) - 1:
                    if i == 1:
                        plt.colorbar(
                            cnf,
                            ax=axs[:2, :],
                            shrink=0.7,
                            label=kwargs.get("clabel", ""),
                        )
                    elif i == 2:
                        plt.colorbar(
                            cnf,
                            ax=axs[2, :],
                            shrink=0.9,
                            label=kwargs.get("clabel_delta", ""),
                        )

                ax.set_global()

                gl = ax.gridlines(
                    draw_labels=True,
                    linewidth=0.5,
                    color="dimgray",
                    alpha=0.4,
                    zorder=2,
                )
                gl.xlabel_style = {"size": 6, "color": "dimgray"}
                gl.ylabel_style = {"size": 6, "color": "dimgray"}
                gl.bottom_labels = False
                gl.right_labels = False
                if j > 0:
                    gl.left_labels = False

                ax.coastlines(linewidth=0.5, zorder=2)

                if i == 2:
                    ax.text(
                        0.5,
                        -0.05,
                        f"{da.level.values[level]:.0f} hPa",
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        size=10,
                    )
                    ax.set_title("")
                elif (i == 0) and (j == 1):
                    ax.set_title(str(da.time.isel(time=tt).values)[:10])
                else:
                    ax.set_title("")

                if j == 0:
                    ax.text(
                        -0.1,
                        1.02,
                        ["Ground Truth", "Prediction", "Difference"][i],
                        transform=ax.transAxes,
                        size=8,
                        weight="bold",
                    )

    return None, None


def animate_3d_variable(pred, targ, outpath, plot_kwargs=dict(), num_workers=32):
    da = (
        xr.Dataset({"pred": pred, "targ": targ})
        .to_array("vari")
        .chunk({"time": 1, "lat": -1, "lon": -1, "level": -1, "vari": -1})
        .fillna(0.0)
    )

    mov = Movie(da, plot_3d_variable, pixelwidth=1920, pixelheight=960, **plot_kwargs)

    mov.save(
        str(outpath),
        remove_frames=True,
        remove_movie=False,
        progress=True,
        overwrite_existing=True,
        framerate=8,
        gif_framerate=8,
        parallel=True,
        parallel_compute_kwargs=dict(scheduler="processes", num_workers=num_workers),
        verbose=False,
    )
    plt.close("all")

    convert_gif(
        str(outpath),
        gpath=str(outpath).replace(".mp4", ".gif"),
        resolution=[640, 320],
        gif_palette=False,
        verbose=False,
        remove_movie=False,
        gif_framerate=8,
    )


def animate_predictions(
    preds,
    targs,
    out_dir,
    varnames=["co2molemix"],
    postfix="3d_anim",
    levels=[0, 3, 5],
    num_workers=32,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for varname in varnames:
        pred, targ = get_pred_targ_from_varname(preds, targs, varname)

        # !!!Caution!!! this is a quick fix producing random movie snippets of different samples, not a continuous time frame, though if noise is generated in a path, it shows this path in latent space
        if "sample" in pred.dims and "time" in targ.dims:
            if "time" in pred.dims:
                pred = pred.isel(sample=np.random.randint(0, pred.sizes["sample"]), drop=True)
            else:
                pred = pred.rename({"sample": "time"})
            min_len = min(pred.sizes["time"], targ.sizes["time"])
            pred = pred.isel(time=slice(0, min_len))
            targ = targ.isel(time=slice(0, min_len))
            pred = pred.assign_coords(time=targ.time)

        vmin = targ.isel(level=levels).quantile(0.02).compute().item()
        vmax = targ.isel(level=levels).quantile(0.98).compute().item()
        max_delta = (
            np.abs(targ - pred).isel(level=levels).quantile(0.95).compute().item()
        )

        nstep = 101

        unit = targ.attrs.get("units", "")
        long_name = targ.attrs.get("long_name", varname)
        clabel = f"{long_name} [{unit}]"
        clabel_delta = f"Delta [{unit}]"

        plot_kwargs = dict(
            clabel=clabel,
            clabel_delta=clabel_delta,
            vmin=vmin,
            vmax=vmax,
            max_delta=max_delta,
            nstep=nstep,
            levels=levels,
        )

        animate_3d_variable(
            pred,
            targ,
            out_dir / f"{varname}_{postfix}.mp4",
            plot_kwargs=plot_kwargs,
            num_workers=num_workers,
        )

    return


ALL_OBSPACK_TYPES = [
    "surface-insitu",
    "aircraft-pfp",
    "aircraft-insitu",
    "surface-flask",
    "shipboard-insitu",
    "aircraft-flask",
    "aircore",
    "surface-pfp",
    "tower-insitu",
    "shipboard-flask",
]


def plot_obspack_stations(
    obs,
    metadata,
    out_dir,
    compare_obs=None,
    ids=None,
    stations=["mlo", "izo", "zep", "spo"],
    types=ALL_OBSPACK_TYPES,
    quality=["representative"],
    levels="default",
    freq="QS",
    imgformats=["svg", "png", "pdf"],
):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    if "co2molemix" not in obs:
        obs["co2molemix"] = massmix_to_molemix(obs.co2massmix)
    if compare_obs is not None:
        if "co2molemix" not in compare_obs:
            if "co2massmix" not in compare_obs:
                compare_obs["co2massmix"] = density_to_massmix(
                    compare_obs.co2density, compare_obs.airdensity, ppm=True
                )
            compare_obs["co2molemix"] = massmix_to_molemix(compare_obs.co2massmix)

    metadata["default_level"] = metadata.level == metadata.groupby(
        ["station", "quality", "type"]
    )["level"].transform("max")
    if stations == "all":
        stations = metadata.station.unique()
    if ids is not None:
        subset = metadata[metadata.id.isin(ids)]
    else:
        subset = metadata[
            (metadata.station.isin(stations))
            & (metadata.type.isin(types))
            & (metadata.quality.isin(quality))
        ]
        if levels == "default":
            subset = subset[subset.default_level]
        elif isinstance(levels, list):
            subset = subset[subset.level.isin(levels)]

    filenames = pd.Series(obs.obs_filename.max("time"))

    matching_indices = {}
    for idx, filename in enumerate(filenames):
        matching_indices[filename] = idx

    for _, row in subset.iterrows():
        station_id = row["id"]

        if station_id not in matching_indices:
            print(f"Warning: Station {station_id} not found in observations")
            continue

        i = matching_indices[station_id]

        with mpl.rc_context(mpl_rc_params):
            plt.figure(figsize=(8, 5))
            ax = plt.subplot()

            if compare_obs is not None:
                compare_obs.co2molemix.isel(cell=i).plot(
                    ax=ax, label="Inversion", color="tab:green", lw=0.75
                )

            obs.co2molemix.isel(cell=i).plot(
                ax=ax, label="Predicted", color="tab:orange", lw=0.75
            )

            obs.obs_co2molemix.isel(cell=i).plot(
                ax=ax, color="black", lw=0.75, alpha=0.85, label="Observed", marker="x"
            )

            ax.set_xlabel("")
            ax.set_title(f"{row['site_name']}, Level {row['level']}")
            ax.set_ylabel("CO2 molemix [ppm]")

            for date in pd.date_range(
                start=obs.time[0].item(), end=obs.time[-1].item(), freq=freq
            ):
                ax.axvline(x=date, color="grey", alpha=0.5, ls="--", lw=0.5, zorder=0)
            plt.legend()
            plt.tight_layout()

            for imgformat in imgformats:
                plt.savefig(out_dir / f"obspack_{row['id']}.{imgformat}", dpi=300)
            plt.close()


def plot_analyze_noise_path(noises, angles, label="$\\theta$"):
    """
    Analyze a sequence of high-dimensional noise tensors.
    Returns matplotlib figures for norm, cosine similarity, and PCA projection.
    """
    noises_flat = [x.flatten().cpu() for x in noises]
    norms = torch.tensor([x.norm().item() for x in noises_flat])
    cosine_sims = [
        torch.nn.functional.cosine_similarity(noises_flat[0], x, dim=0).item()
        for x in noises_flat
    ]

    if isinstance(angles, torch.Tensor):
        xvals = angles.cpu().numpy()
        is_numeric = True
    elif all(isinstance(a, (int, float)) for a in angles):
        xvals = np.array(angles)
        is_numeric = True
    else:
        xvals = np.arange(len(angles))
        is_numeric = False

    figs = []

    # Norm vs θ
    fig1, ax1 = plt.subplots(figsize=(6, 3), constrained_layout=True)
    ax1.plot(xvals, norms, marker="o")
    ax1.set_ylabel("Norm of noise vector")
    ax1.set_xlabel(label)
    ax1.set_title(f"Norm of x({label}) along path")
    if not is_numeric:
        ax1.set_xticks(xvals)
        ax1.set_xticklabels(angles, rotation=45)
    ax1.grid(True)
    figs.append(fig1)

    # Cosine similarity vs θ
    fig2, ax2 = plt.subplots(figsize=(6, 3), constrained_layout=True)
    ax2.plot(xvals, cosine_sims, marker="o")
    ax2.set_ylabel("Cosine similarity with start")
    ax2.set_xlabel(label)
    ax2.set_title("Cosine similarity with starting noise")
    if not is_numeric:
        ax2.set_xticks(xvals)
        ax2.set_xticklabels(angles, rotation=45)
    ax2.grid(True)
    figs.append(fig2)

    # PCA projection
    X = torch.stack(noises_flat).numpy()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig3, ax3 = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax3.plot(X_pca[:, 0], X_pca[:, 1], marker="o")
    for i, a in enumerate(angles):
        if i % 10 == 0 or i == len(angles) - 1 or len(angles) <= 10:
            if is_numeric:
                ax3.text(X_pca[i, 0], X_pca[i, 1], f"{a:.2f}")
            else:
                ax3.text(X_pca[i, 0], X_pca[i, 1], str(a))
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_title("Noise path projected via PCA")
    figs.append(fig3)

    return figs


def plot_pairwise_cosine_similarity(noises, labels=None):
    """
    Compute and plot the pairwise cosine similarity matrix between noise samples.
    """
    
    X = torch.stack([x.flatten() for x in noises])
    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim = (X_norm @ X_norm.T).cpu().numpy()

    # Default labels if none provided
    if labels is None:
        labels = [str(i) for i in range(len(noises))]

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(cos_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="cosine similarity")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Pairwise cosine similarity of noise vectors")

    return fig


def plot_noise_diagnostics(noises, angles, out_dir, label="$\\theta$", imgformats=["svg", "png", "pdf"]):
    """
    Save diagnostics for noise interpolation or flow matching analysis.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    figs = plot_analyze_noise_path(noises, angles, label=label)
    names = ["norm_vs_angle", "cosine_vs_angle", "pca_projection"]

    if label == "Index pair":
        fig_4 = plot_pairwise_cosine_similarity(noises, labels=angles)
        figs.append(fig_4)
        names.append("pairwise_cosine_similarity")

    for name, fig in zip(names, figs):
        for fmt in imgformats:
            fig.savefig(out_dir / f"noise_{name}.{fmt}", dpi=300)
        plt.close(fig)

    return


def plot_panel(ax, data, title, cmap="cividis", vmin=None, vmax=None, aspect_ratio=2.0, bold=False):
    """Helper to plot one panel with consistent style."""
    im = ax.imshow(
        data[::-1, :],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect=aspect_ratio / 2,
    )
    ax.set_title(title, fontsize=13 if bold else 11, fontweight="bold" if bold else "normal")
    ax.axis("off")
    return im


def plot_obs_mask_and_samples(
    batch,
    preds_var,
    varname="co2massmix",
    nlat=32,
    nlon=64,
    max_samples=6,
):
    """
    Plot observed values, masked observations, and several generated samples.
    Layout: 2x4 grid
      [0,0] = Ground Truth
      [1,0] = Masked Observations
      [0,1..3], [1,1..3] = Generated Samples (up to 6)
    """
    b, t, c = 0, 0, 0

    obs_values = batch["obs_values"][b, t, :, c].detach().cpu().numpy().reshape(nlat, nlon)
    obs_mask = batch["obs_mask"][b, t, :, c].detach().cpu().numpy().reshape(nlat, nlon)
    target_vals = batch[varname][b, t, :, c].detach().cpu().numpy().reshape(nlat, nlon)
    masked_obs = np.where(obs_mask, obs_values, np.nan)

    # Generated samples [sample, lat, lon, level, (time)]
    samples = preds_var
    if "level" in samples.dims:
        samples = samples.isel(level=0)
    if "time" in samples.dims:
        samples = samples.isel(time=0)
    samples_np = samples.values  # shape: [sample, lat, lon]
    n_samples = min(samples_np.shape[0], max_samples)

    # Global color limits
    vmin = np.nanmin([np.nanmin(target_vals), np.nanmin(samples_np[:n_samples, ...])])
    vmax = np.nanmax([np.nanmax(target_vals), np.nanmax(samples_np[:n_samples, ...])])
    obs_min, obs_max = np.nanmin(masked_obs), np.nanmax(masked_obs)
    targ_min, targ_max = np.nanmin(target_vals), np.nanmax(target_vals)

    # Figure setup
    aspect_ratio = nlon / nlat
    base_size = 3.0
    fig_width = 4 * base_size * (aspect_ratio / 2)
    fig_height = 2 * base_size
    fig, axs = plt.subplots(2, 4, figsize=(fig_width, fig_height))
    axs = axs.reshape(2, 4)

    # Panels ([0,0]: Ground truth, [1,0]: Masked obs, [0,1..3] and [1,1..3]: samples)
    im = plot_panel(axs[0, 0], target_vals, "Ground Truth", vmin=targ_min, vmax=targ_max, aspect_ratio=aspect_ratio, bold=True)
    plot_panel(axs[1, 0], np.ma.masked_invalid(masked_obs), "Masked Observations",
               vmin=obs_min, vmax=obs_max, aspect_ratio=aspect_ratio, bold=True)

    for i in range(n_samples):
        row = 0 if i < 3 else 1
        col = (i % 3) + 1
        plot_panel(axs[row, col], samples_np[i, :, :], f"Sample {i}",
                   vmin=vmin, vmax=vmax, aspect_ratio=aspect_ratio)

    fig.text(0.56, 0.9, "Generated Samples", fontsize=14, fontweight="bold", ha="center", va="top")

    for row in range(2):
        for col in range(4):
            if not axs[row, col].images:
                axs[row, col].axis("off")

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical", label="CO₂ [ppm]")

    fig.suptitle("Observations vs Generated Samples", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    return fig

def plot_obs_mask_and_samples_x(
    batch: dict,
    preds_var: xr.DataArray,
    varname: str = "co2massmix",
    nlat: int = 32,
    nlon: int = 64,
    max_samples: int = 6,
    b: int = 0,
    t: int = 0
):
    """
    Plot observed values, masked observations, and several generated samples.
    Layout: 2x4 grid
      [0,0] = Ground Truth
      [1,0] = Masked Observations
      [0,1..3], [1,1..3] = Generated Samples (up to 6)
    """
    b, t, c = 0, 0, 0

    obs_values = batch["obs_values"][b, t, :, c].detach().cpu().numpy().reshape(nlat, nlon)
    obs_mask = batch["obs_mask"][b, t, :, c].detach().cpu().numpy().reshape(nlat, nlon)
    masked_obs = np.where(obs_mask, obs_values, np.nan)

    # Generated samples [sample, lat, lon, level, (time)]
    samples = preds_var
    if "time" in samples.dims:
        samples = samples.isel(time=t)

    if "xco2_averaging_kernel" in batch:
        ak = batch["xco2_averaging_kernel"][b, t, :, :].detach().cpu().numpy()  # [N, C]
        xco2_prior = batch["xco2_apriori"][b, t, :].detach().cpu().numpy().squeeze(-1)  # [N]
        co2_profile_prior = batch["co2_profile_apriori"][b, t, :, :].detach().cpu().numpy()  # [N, C]
        vals = batch[varname][b, t, :, :].detach().cpu().numpy()  # [N, C]
        target_vals = compute_xco2_via_ak(vals, ak, xco2_prior, co2_profile_prior).reshape(nlat, nlon)
        if "level" in samples.dims:
            ak_reshaped = ak.reshape(nlat, nlon, -1)  # [lat, lon, level]
            xco2_prior_reshaped = xco2_prior.reshape(nlat, nlon)  # [lat, lon]
            co2_profile_prior_reshaped = co2_profile_prior.reshape(nlat, nlon, -1)  # [lat, lon, level]
            samples_np = samples.values  # [sample, lat, lon, level]
            samples_list = []
            for i in range(min(samples_np.shape[0], max_samples)):
                samples_list.append(compute_xco2_via_ak(samples_np[i], ak_reshaped, xco2_prior_reshaped, co2_profile_prior_reshaped))
            samples = xr.DataArray(
                np.array(samples_list),
                dims=["sample", "lat", "lon"]
            )
    else:
        target_vals = batch[varname][b, t, :, :].mean(dim=-1).detach().cpu().numpy().reshape(nlat, nlon)
        if "level" in samples.dims:
            samples = samples.mean(dim="level")

    samples_np = samples.values  # shape: [sample, lat, lon]
    n_samples = min(samples_np.shape[0], max_samples)

    # Global color limits
    vmin = np.nanmin([np.nanmin(target_vals), np.nanmin(samples_np[:n_samples, ...])])
    vmax = np.nanmax([np.nanmax(target_vals), np.nanmax(samples_np[:n_samples, ...])])
    if np.all(np.isnan(masked_obs)):
        obs_min, obs_max = vmin, vmax
    else:
        obs_min, obs_max = np.nanmin(masked_obs), np.nanmax(masked_obs)
    targ_min, targ_max = np.nanmin(target_vals), np.nanmax(target_vals)

    # Figure setup
    aspect_ratio = nlon / nlat
    base_size = 3.0
    fig_width = 4 * base_size * (aspect_ratio / 2)
    fig_height = 2 * base_size
    fig, axs = plt.subplots(2, 4, figsize=(fig_width, fig_height))
    axs = axs.reshape(2, 4)

    # Panels ([0,0]: Ground truth, [1,0]: Masked obs, [0,1..3] and [1,1..3]: samples)
    im = plot_panel(axs[0, 0], target_vals, "Ground Truth", vmin=targ_min, vmax=targ_max, aspect_ratio=aspect_ratio, bold=True)
    plot_panel(axs[1, 0], np.ma.masked_invalid(masked_obs), "Masked Observations",
               vmin=obs_min, vmax=obs_max, aspect_ratio=aspect_ratio, bold=True)

    for i in range(n_samples):
        row = 0 if i < 3 else 1
        col = (i % 3) + 1
        plot_panel(axs[row, col], samples_np[i, :, :], f"Sample {i}",
                   vmin=vmin, vmax=vmax, aspect_ratio=aspect_ratio)

    fig.text(0.56, 0.9, "Generated Samples", fontsize=14, fontweight="bold", ha="center", va="top")

    for row in range(2):
        for col in range(4):
            if not axs[row, col].images:
                axs[row, col].axis("off")

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical", label="XCO₂ [ppm]")

    fig.suptitle("Observations vs Generated Samples", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    return fig


def plot_mask_pattern_on_samples(
    batch,
    preds_var,
    nlat=32,
    nlon=64,
    max_samples=6,
):
    """
    Plot generated samples with mask pattern overlaid as contours.
    This helps debug if mask pattern artifacts appear in generated samples.
    """
    b, t, c = 0, 0, 0

    # Get mask pattern
    if "obs_mask_original" in batch:
        obs_mask = batch["obs_mask_original"][b, t, :, c].detach().cpu().numpy().reshape(nlat, nlon)
    else:
        obs_mask = batch["obs_mask"][b, t, :, c].detach().cpu().numpy().reshape(nlat, nlon)

    # Generated samples [sample, lat, lon, level, (time)]
    samples = preds_var
    if "time" in samples.dims:
        samples = samples.isel(time=t)

    if "xco2_averaging_kernel" in batch:
        ak = batch["xco2_averaging_kernel"][b, t, :, :].detach().cpu().numpy()  # [N, C]
        if "level" in samples.dims:
            ak_reshaped = ak.reshape(nlat, nlon, -1)  # [lat, lon, level]
            xco2_prior_reshaped = batch["xco2_apriori"][b, t, :].detach().cpu().numpy().reshape(nlat, nlon)  # [lat, lon]
            co2_profile_prior_reshaped = batch["co2_profile_apriori"][b, t, :, :].detach().cpu().numpy().reshape(nlat, nlon, -1)  # [lat, lon, level]
            samples_np = samples.values  # [sample, lat, lon, level]
            samples_list = []
            for i in range(min(samples_np.shape[0], max_samples)):
                samples_list.append(compute_xco2_via_ak(samples_np[i], ak_reshaped, xco2_prior_reshaped, co2_profile_prior_reshaped))
            samples = xr.DataArray(
                np.array(samples_list),
                dims=["sample", "lat", "lon"]
            )
    else:
        if "level" in samples.dims:
            samples = samples.mean(dim="level")

    samples_np = samples.values  # shape: [sample, lat, lon]
    n_samples = min(samples_np.shape[0], max_samples)

    # Global color limits
    vmin = np.nanmin(samples_np[:n_samples, ...])
    vmax = np.nanmax(samples_np[:n_samples, ...])

    # Figure setup
    aspect_ratio = nlon / nlat
    base_size = 3.0
    fig_width = 3 * base_size * (aspect_ratio / 2)
    fig_height = 2 * base_size
    fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))
    axs = axs.flatten()

    for i in range(n_samples):
        ax = axs[i]
        # Plot sample
        im = ax.imshow(
            samples_np[i, :, :],
            cmap="cividis",
            vmin=vmin,
            vmax=vmax,
            aspect=aspect_ratio / 2,
            origin='lower',
        )
        # Overlay mask as red contour
        ax.contour(
            obs_mask,
            levels=[0.5],
            colors='red',
            linewidths=1.5,
        )
        ax.set_title(f"Sample {i} + Mask", fontsize=11)
        ax.axis("off")

    # Hide unused subplots
    for i in range(n_samples, len(axs)):
        axs[i].axis("off")

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical", label="XCO₂ [ppm]")

    fig.suptitle("Generated Samples with Mask Pattern Overlay (red contour)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    return fig


def plot_masked_bias_samples(
    batch,
    preds_var,
    varname="co2massmix",
    nlat=32,
    nlon=64,
    max_samples=6,
):
    """
    Plot bias of generated samples relative to ground truth, only at grid cells where observations are masked.
    Layout: 2x4 grid
      [0,0] = Ground Truth
      [1,0] = Masked Observations
      [0,1..3], [1,1..3] = Generated Samples (up to 6)
    """

    b, t, c = 0, 0, 0

    obs_values = batch["obs_values"][b, t, :, c].detach().cpu().numpy().reshape(nlat, nlon)
    obs_mask = batch["obs_mask"][b, t, :, c].detach().cpu().numpy().reshape(nlat, nlon)
    masked_obs = np.where(obs_mask, obs_values, np.nan)

    # Generated samples [sample, lat, lon, level, (time)]
    samples = preds_var
    if "time" in samples.dims:
        samples = samples.isel(time=t)

    if "xco2_averaging_kernel" in batch:
        ak = batch["xco2_averaging_kernel"][b, t, :, :].detach().cpu().numpy()
        xco2_prior = batch["xco2_apriori"][b, t, :].detach().cpu().numpy().squeeze(-1)
        co2_profile_prior = batch["co2_profile_apriori"][b, t, :, :].detach().cpu().numpy()
        vals = batch[varname][b, t, :, :].detach().cpu().numpy()
        target_vals = compute_xco2_via_ak(vals, ak, xco2_prior, co2_profile_prior).reshape(nlat, nlon)
        if "level" in samples.dims:
            ak_reshaped = ak.reshape(nlat, nlon, -1)
            xco2_prior_reshaped = xco2_prior.reshape(nlat, nlon)
            co2_profile_prior_reshaped = co2_profile_prior.reshape(nlat, nlon, -1)
            samples_np = samples.values
            samples_list = []
            for i in range(min(samples_np.shape[0], max_samples)):
                samples_list.append(compute_xco2_via_ak(samples_np[i], ak_reshaped, xco2_prior_reshaped, co2_profile_prior_reshaped))
            samples = xr.DataArray(
                np.array(samples_list),
                dims=["sample", "lat", "lon"]
            )
    else:
        target_vals = batch[varname][b, t, :, :].mean(dim=-1).detach().cpu().numpy().reshape(nlat, nlon)
        if "level" in samples.dims:
            samples = samples.mean(dim="level")

    samples_np = samples.values  # shape: [sample, lat, lon]
    n_samples = min(samples_np.shape[0], max_samples)

    bias_list = []
    for i in range(n_samples):
        bias = samples_np[i] - target_vals
        bias = np.where(obs_mask, bias, np.nan)
        bias_list.append(bias)

    bias_np = np.array(bias_list)

    # Global symmetric color limits
    vmax = np.nanmax(np.abs(bias_np))
    vmin = -vmax
    targ_min, targ_max = np.nanmin(target_vals), np.nanmax(target_vals)
    if np.all(np.isnan(masked_obs)):
        obs_min, obs_max = targ_min, targ_max
    else:
        obs_min, obs_max = np.nanmin(masked_obs), np.nanmax(masked_obs)

    # Figure setup
    aspect_ratio = nlon / nlat
    base_size = 3.0
    fig_width = 4 * base_size * (aspect_ratio / 2)
    fig_height = 2 * base_size
    fig, axs = plt.subplots(2, 4, figsize=(fig_width, fig_height))
    axs = axs.reshape(2, 4)

    # Panels ([0,0]: Ground truth, [1,0]: Masked obs, [0,1..3] and [1,1..3]: samples)
    # Ground truth
    im = plot_panel(axs[0, 0], target_vals, "Ground Truth", vmin=targ_min, vmax=targ_max, aspect_ratio=aspect_ratio, bold=True)

    # Masked observations
    plot_panel(axs[1, 0], np.ma.masked_invalid(masked_obs), "Masked Observations",
        vmin=obs_min, vmax=obs_max, aspect_ratio=aspect_ratio, bold=True)

    # Bias samples
    for i in range(n_samples):
        row = 0 if i < 3 else 1
        col = (i % 3) + 1
        im2 = plot_panel(
            axs[row, col],
            bias_np[i],
            f"Bias Sample {i}",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            aspect_ratio=aspect_ratio
        )

    fig.text(
        0.56,
        0.9,
        "Generated Samples",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="top"
    )

    for row in range(2):
        for col in range(4):
            if not axs[row, col].images:
                axs[row, col].axis("off")

    plt.tight_layout(rect=[0, 0, 0.9, 1], h_pad=0.2)

    # shared colorbars
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im2, cax=cbar_ax, orientation="vertical", label="Bias [ppm]")

    bbox0 = axs[1,0].get_position()
    cbar_truth_ax = fig.add_axes([
        bbox0.x0,
        bbox0.y0 - 0.05,
        bbox0.width,
        0.02
    ])
    fig.colorbar(
        im,
        cax=cbar_truth_ax,
        orientation="horizontal",
        label="XCO₂ [ppm]"
    )

    fig.suptitle("Bias of Generated Samples (Masked Locations)", fontsize=16)

    return fig


def plot_masking_diagnostics(
        batch: dict,
        preds: xr.Dataset,
        out_dir: str,
        varnames: list = ["co2massmix"],
        nlat: int = 32,
        nlon: int = 64,
        imgformats: list = ["svg", "png", "pdf"]) -> None:
    """
    Save diagnostics for masking or flow matching analysis.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for varname in varnames:
        if varname not in preds:
            raise KeyError(f"{varname} not found in preds")

        if varname == "co2massmix":
            varname = "co2molemix"
            batch[varname] = massmix_to_molemix(batch["co2massmix"])
            preds[varname] = massmix_to_molemix(preds["co2massmix"])
        preds_var = preds[varname]

        fig = plot_obs_mask_and_samples_x(
            batch,
            preds_var,
            varname=varname,
            nlat=nlat,
            nlon=nlon,
            max_samples=6,
            )
        for fmt in imgformats:
            fig.savefig(out_dir / f"masking_{varname}.{fmt}", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig_bias = plot_masked_bias_samples(
            batch,
            preds_var,
            varname=varname,
            nlat=nlat,
            nlon=nlon,
            max_samples=6,
        )
        for fmt in imgformats:
            fig_bias.savefig(out_dir / f"masking_{varname}_bias.{fmt}", dpi=300, bbox_inches="tight")
        plt.close(fig_bias)

        # Plot mask pattern overlay on generated samples
        if "obs_mask_original" in batch:
            fig_mask = plot_mask_pattern_on_samples(
                batch,
                preds_var,
                nlat=nlat,
                nlon=nlon,
                max_samples=6,
            )
            for fmt in imgformats:
                fig_mask.savefig(out_dir / f"masking_{varname}_mask_overlay.{fmt}", dpi=300, bbox_inches="tight")
            plt.close(fig_mask)

    return


def plot_samples(
        preds,
        out_dir,
        score_path,
        tests=None,
        freq="QS",
        varnames=["co2molemix"],
        normalize=False,
        imgformats=["svg", "png", "pdf"],
        **generate_kwargs):
    """
    Wrapper for sample diagnostics plots
    """
    noise_pattern = generate_kwargs.get("noise_pattern", None)
    avg_over_levels = generate_kwargs.get("avg_over_levels", True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_path = Path(score_path)
    maps = xr.open_dataset(score_path / ("metrics_maps.nc" if freq == "QS" else f"metrics_maps_{freq}.nc"))
    df_global_scalars = pd.read_csv(score_path / ("metrics_global_scalars.csv" if freq == "QS" else f"metrics_global_scalars_{freq}.csv"))

    for varname in varnames:
        if varname not in preds:
            raise KeyError(f"{varname} not found in preds")

        preds_var = preds[varname]  # shape: [sample=100, lat=32, lon=64, level=10]

        plot_sample_mean_cdf(preds_var, out_dir,
                        tests=tests,
                        varname=varname,
                        avg_over_levels=avg_over_levels,
                        normalize=normalize,
                        center_to_test_mean=False,
                        imgformats=imgformats,)
        
        plot_sample_mean_cdf(preds_var, out_dir,
                        tests=tests,
                        varname=varname,
                        avg_over_levels=avg_over_levels,
                        normalize=normalize,
                        center_to_test_mean=False,
                        remove_low_pressure=True,
                        imgformats=imgformats,)

        plot_sample_mean_cdf(preds_var, out_dir,
                        tests=tests,
                        varname=varname,
                        avg_over_levels=avg_over_levels,
                        normalize=normalize,
                        center_to_test_mean=True,
                        imgformats=imgformats,)

        plot_sample_cdf(preds_var, out_dir,
                        tests=tests,
                        varname=varname,
                        avg_over_levels=avg_over_levels,
                        normalize=normalize,
                        center_to_test_mean=False,
                        imgformats=imgformats,)

        plot_sample_cdf(preds_var, out_dir,
                        tests=tests,
                        varname=varname,
                        avg_over_levels=avg_over_levels,
                        normalize=normalize,
                        center_to_test_mean=True,
                        imgformats=imgformats,)

        if noise_pattern is not None:
            plot_pairwise_sample_distances(preds_var, out_dir,
                                        varname=varname,
                                        avg_over_levels=avg_over_levels,
                                        imgformats=imgformats)
            nsamples = preds_var.sizes["sample"]

            if noise_pattern in ["spiral_noise", "spiral_outward_noise"]:
                angles = torch.linspace(0, 4*np.pi, nsamples)
                param_name = r"$\theta$"
            elif noise_pattern in ["geodesic_noise", "linear_noise"]:
                angles = torch.linspace(0, 1, nsamples)
                param_name = r"$\alpha$"
            elif noise_pattern == "antipodal_orthogonal_noise":
                labels = []
                for i in range(nsamples // 2):
                    labels += [f"{i+1}a", f"{i+1}b"]
                if nsamples % 2 == 1:
                    labels.append(f"{(nsamples // 2) + 1}a")
                angles = labels
                param_name = "Index pair"
            else:
                angles = torch.arange(nsamples)
                param_name = "Index"
            
            plot_noise_path_samples(preds_var, out_dir,
                                    varname=varname,
                                    noise_pattern=noise_pattern,
                                    angles=angles,
                                    param_name=param_name,
                                    level_idx=0,
                                    imgformats=imgformats)

        if tests is not None and varname not in tests:
            raise KeyError(f"{varname} not found in tests")

        tests_var = tests[varname]  # shape: [time=9, level=10, lat=32, lon=64]
        tests_var_mean = tests_var.mean(dim="time")  # shape: [level, lat, lon]

        if not avg_over_levels:
            for i, lvl in enumerate(tests_var_mean.level.values):
                plot_crps(maps,
                          df_global_scalars,
                          out_dir,
                          varname=varname,
                          level=lvl,
                          imgformats=imgformats)

                plot_scatter_preds_vs_tests(maps, df_global_scalars,
                                            preds_var.isel(level=i),
                                            tests_var_mean.isel(level=i),
                                            out_dir,
                                            varname=varname,
                                            level=lvl,
                                            imgformats=imgformats)

                plot_spread_skill(maps,
                                  out_dir,
                                  varname=varname,
                                  level=lvl,
                                  imgformats=imgformats)

                plot_error_locations(maps,
                                     out_dir,
                                     varname=varname,
                                     level=lvl,
                                     imgformats=imgformats)

        preds_var_mean = preds_var.mean(dim="level")  # shape: [sample, lat, lon]
        tests_var_mean = tests_var_mean.mean(dim="level")  # shape: [lat, lon]

        plot_crps(maps,
                  df_global_scalars,
                  out_dir,
                  varname=varname,
                  imgformats=imgformats)

        plot_scatter_preds_vs_tests(maps, df_global_scalars,
                                    preds_var_mean,
                                    tests_var_mean,
                                    out_dir,
                                    varname=varname,
                                    imgformats=imgformats)

        plot_spread_skill(maps,
                          out_dir,
                          varname=varname,
                          imgformats=imgformats)

        plot_error_locations(maps,
                             out_dir,
                             varname=varname,
                             imgformats=imgformats)


def plot_pairwise_sample_distances(preds_var, out_dir,
                                   varname="co2molemix",
                                   avg_over_levels=True,
                                   imgformats=["svg", "png", "pdf"]):
    """
    Plot pairwise L2 distances between samples for a given variable at one vertical level.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    
    # Determine levels
    if "level" in preds_var.dims and not avg_over_levels:
        levels = preds_var.level.values
    else:
        levels = [None]

    for level in levels:
        if level is not None:
            ds_slice = preds_var.sel(level=level)
        else:
            ds_slice = preds_var.mean(dim="level") if "level" in preds_var.dims else preds_var

        data = ds_slice.values  # shape: (n_samples, lat, lon)
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array (sample, lat, lon), got shape {data.shape}")

        # Flatten spatial dims and compute pairwise L2 distances
        data_flat = data.reshape(data.shape[0], -1)
        dist_matrix = np.linalg.norm(data_flat[:, None, :] - data_flat[None, :, :], axis=-1)

        with plt.rc_context({
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }):
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(dist_matrix, cmap="cividis")
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, label="L2 distance between samples")
            ax.set_title(f"Pairwise distances ({varname})" + (f" at level {level:.0f}" if level is not None else ""))
            ax.set_xlabel("Sample index")
            ax.set_ylabel("Sample index")

            n_samples = data.shape[0]
            ax.set_xticks(np.arange(0, n_samples, 2))
            ax.set_yticks(np.arange(0, n_samples, 2))

            fig.tight_layout()
            for fmt in imgformats:
                filename = f"pairwise_distances_{varname}" + (f"_level{level}" if level is not None else "") + f".{fmt}"
                fig.savefig(out_dir / filename, dpi=300)

            plt.close(fig)


def plot_noise_path_samples(
    preds_var: xr.DataArray,
    out_dir,
    varname="co2molemix",
    noise_pattern: str = "random",
    angles=None,
    param_name: str = "Index",
    level_idx: int = 0,
    projection=None,
    cmap="bone_r",
    imgformats=["svg", "png", "pdf"],
):
    """
    Plot the CO₂ samples along their noise trajectory.

    preds_var dims expected:
        (sample, lat, lon, level)
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if projection is None:
        projection = ccrs.PlateCarree()

    samples = preds_var

    if "level" in samples.dims:
        samples = samples.isel(level=level_idx)

    nsamples = samples.sizes["sample"]
    lat = samples.sizes["lat"]
    lon = samples.sizes["lon"]

    nrows = 2
    max_cols = 5
    max_panels = nrows * max_cols
    indices = np.linspace(0, nsamples - 1, min(nsamples, max_panels), dtype=int)
    ncols = int(np.ceil(len(indices) / nrows))

    panel_width = 3
    panel_height = panel_width * lat / lon
    figsize = (panel_width * ncols, panel_height * nrows)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.05, hspace=0.15)

    data = samples.values
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    axes = []

    for plot_idx, i in enumerate(indices):

        row = plot_idx // ncols
        col = plot_idx % ncols

        ax = fig.add_subplot(gs[row, col], projection=projection)
        axes.append(ax)

        map_data = samples.isel(sample=i).values

        im = ax.pcolormesh(
            samples["lon"],
            samples["lat"],
            map_data,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            rasterized=True,
        )

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        angle = angles[i] if angles is not None else i
        if isinstance(angle, (float, np.floating, torch.Tensor)):
            label = f"{param_name}={float(angle):.2f}"
        else:
            label = f"{param_name}={angle}"
        ax.text(
            0.05,
            0.95,
            label,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
        )

    for i in range(nsamples, nrows * ncols):
        axes.append(fig.add_subplot(gs[i]))
        axes[-1].set_visible(False)

    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.05])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar_ax.set_xlabel("CO₂ [ppm]")

    level_text = "surface level" if level_idx == 0 else f"level {level_idx}"
    fig.suptitle(
        f"Sampled CO₂ at {level_text} from {noise_pattern} path",
        fontsize=12, fontweight="bold",
    )

    for fmt in imgformats:
        fig.savefig(out_dir / f"{noise_pattern}_path_samples_{varname}.{fmt}", dpi=300, bbox_inches="tight")

    plt.close(fig)


def normalize_array(arr, normalize):
    return (arr - arr.mean()) / arr.std() if normalize else arr


def plot_sample_mean_cdf(preds_var, out_dir, tests=None,
                    varname="co2molemix", avg_over_levels=True,
                    normalize=False,
                    center_to_test_mean=False,
                    remove_low_pressure=False,
                    imgformats=["svg", "png", "pdf"]):
    """
    Plot cumulative distribution functions (CDFs) of normalized mean values per sample for one or multiple variables. Optionally compare to test data.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if avg_over_levels:
        preds_var = preds_var.mean(dim="level")  # shape: [(time,) sample, lat, lon]

    pred_mean = preds_var.mean(dim=["lat", "lon"])  # shape: [(time,) sample, (level)]
    test_mean = None
    if tests is not None:
        tests_var = tests[varname]  # shape: [time, level, lat, lon]
        # !!!Caution!!! this is a dirty fix especially for long time series.
        tests_var = tests_var.rename({"time": "sample"})
        if avg_over_levels:
            tests_var = tests_var.mean(dim="level")  # shape: [sample, lat, lon]
        test_mean = tests_var.mean(dim=["lat", "lon"])  # shape: [sample, (level)]


    fig, ax = plt.subplots(figsize=(8, 5))

    if not avg_over_levels:
        levels = preds_var.level.values
        if remove_low_pressure:
            levels = levels[:-1]
        colors = sns.color_palette("crest", len(levels))
        levels_plot = levels[::-1]

        for i, lvl in enumerate(levels_plot):
            color = colors[i]

            # Prediction CDF
            level_pred = pred_mean.sel(level=lvl).values.flatten()
            level_pred = normalize_array(level_pred, normalize)
            if center_to_test_mean and test_mean is not None:
                level_pred -= test_mean.sel(level=lvl).mean().values
            x_pred = np.sort(level_pred)
            y_pred = np.arange(1, len(level_pred)+1) / len(level_pred)
            ax.plot(x_pred, y_pred, marker="x", linestyle="-", alpha=0.7, color=color)

            # Test CDF
            if not center_to_test_mean and test_mean is not None:
                level_test = test_mean.sel(level=lvl).values.flatten()
                level_test = normalize_array(level_test, normalize)
                x_test = np.sort(level_test)
                y_test = np.arange(1, len(level_test)+1) / len(level_test)
                ax.plot(x_test, y_test, marker="o", linestyle="-", alpha=0.7, color=color)

        legend_handles = []
        legend_handles.append(Line2D([], [], linestyle="none", label="Dataset"))
        legend_handles.append(Line2D([0], [0], marker="x", color="black", linestyle="None", label="  Predictions"))
        if not center_to_test_mean and test_mean is not None:
            legend_handles.append(Line2D([0], [0], marker="o", color="black", linestyle="None", label="  Tests"))
        legend_handles.append(Line2D([], [], linestyle="none", label=""))
        legend_handles.append(Line2D([], [], linestyle="none", label="Level [hPa]"))
        for color, lvl in zip(colors, levels_plot):
            legend_handles.append(
                Line2D([0], [0], color=color, lw=2, label=f"  {lvl:.0f}")
            )
        legend = ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            frameon=True
        )
        for text in legend.get_texts():
            if text.get_text().strip() in ["Dataset", "Level [hPa]"]:
                text.set_weight("bold")

    else:
        pred_mean_vals = pred_mean.values
        pred_mean_vals = normalize_array(pred_mean_vals, normalize)
        if center_to_test_mean and test_mean is not None:
            pred_mean_vals -= test_mean.mean().values
        x_pred = np.sort(pred_mean_vals)
        y_pred = np.arange(1, len(pred_mean_vals)+1) / len(pred_mean_vals)
        ax.plot(x_pred, y_pred, label="Predictions", marker="x", linestyle="-")

        if not center_to_test_mean and test_mean is not None:
            test_mean_vals = test_mean.values
            test_mean_vals = normalize_array(test_mean_vals, normalize)
            x_test = np.sort(test_mean_vals)
            y_test = np.arange(1, len(test_mean_vals)+1) / len(test_mean_vals)
            ax.plot(x_test, y_test, label="Tests", marker="o", linestyle="-")
        ax.legend(title="Dataset", loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    xlabel = f"{'Normalized ' if normalize else ''}Mean {varname} {'[ppm]' if not normalize else ''}"
    title = f"CDF of {'Normalized ' if normalize else ''}Mean {varname} per Sample"
    if center_to_test_mean:
        xlabel = f"Deviation from Test Mean {varname} {'[ppm]' if not normalize else ''}"
        title = f"CDF of Predictions Relative to Test Mean ({varname})"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout(rect=[0, 0, 0.75, 1])

    for fmt in imgformats:
        fig.savefig(out_dir / f"cdf{'_centered' if center_to_test_mean else ''}{'_low_pressure_removed' if remove_low_pressure else ''}_{varname}.{fmt}", dpi=300, bbox_inches='tight')

    plt.close(fig)


def plot_sample_cdf(preds_var, out_dir, tests=None,
                    varname="co2molemix", avg_over_levels=True,
                    normalize=False,
                    center_to_test_mean=False,
                    imgformats=["svg", "png", "pdf"]):
    """
    Plot cumulative distribution functions (CDFs) of normalized values per sample for one or multiple variables. Optionally compare to test data.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if avg_over_levels:
        preds_var = preds_var.mean(dim="level")  # shape: [(time,) sample, lat, lon]

    pred_mean = preds_var  # shape: [(time,) sample, lat, lon, (level)]
    test_mean = None
    if tests is not None:
        tests_var = tests[varname]  # shape: [time, level, lat, lon]
        # !!!Caution!!! this is a dirty fix especially for long time series.
        tests_var = tests_var.rename({"time": "sample"})
        if avg_over_levels:
            tests_var = tests_var.mean(dim="level")  # shape: [sample, lat, lon]
        test_mean = tests_var  # shape: [sample, (level), lat, lon]


    fig, ax = plt.subplots(figsize=(8, 5))

    if not avg_over_levels:
        levels = preds_var.level.values
        colors = sns.color_palette("crest", len(levels))
        levels_plot = levels[::-1]

        for i, lvl in enumerate(levels_plot):
            color = colors[i]

            # Prediction CDF
            level_pred = pred_mean.sel(level=lvl).values.flatten()
            level_pred = normalize_array(level_pred, normalize)
            if center_to_test_mean and test_mean is not None:
                level_pred -= test_mean.sel(level=lvl).mean().values
            x_pred = np.sort(level_pred)
            y_pred = np.arange(1, len(level_pred)+1) / len(level_pred)
            ax.plot(x_pred, y_pred, linestyle="--", alpha=0.7, color=color)

            # Test CDF
            if not center_to_test_mean and test_mean is not None:
                level_test = test_mean.sel(level=lvl).values.flatten()
                level_test = normalize_array(level_test, normalize)
                x_test = np.sort(level_test)
                y_test = np.arange(1, len(level_test)+1) / len(level_test)
                ax.plot(x_test, y_test, linestyle="-", alpha=0.7, color=color)

        legend_handles = []
        legend_handles.append(Line2D([], [], linestyle="none", label="Dataset"))
        legend_handles.append(Line2D([0], [0], color="black", linestyle="--", label="  Predictions"))
        if not center_to_test_mean and test_mean is not None:
            legend_handles.append(Line2D([0], [0], color="black", linestyle="-", label="  Tests"))
        legend_handles.append(Line2D([], [], linestyle="none", label=""))
        legend_handles.append(Line2D([], [], linestyle="none", label="Level [hPa]"))
        for color, lvl in zip(colors, levels_plot):
            legend_handles.append(
                Line2D([0], [0], color=color, lw=2, label=f"  {lvl:.0f}")
            )
        legend = ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            frameon=True
        )
        for text in legend.get_texts():
            if text.get_text().strip() in ["Dataset", "Level [hPa]"]:
                text.set_weight("bold")

    else:
        pred_mean_vals = pred_mean.values
        pred_mean_vals = normalize_array(pred_mean_vals, normalize)
        if center_to_test_mean and test_mean is not None:
            pred_mean_vals -= test_mean.mean().values
        x_pred = np.sort(pred_mean_vals)
        y_pred = np.arange(1, len(pred_mean_vals)+1) / len(pred_mean_vals)
        ax.plot(x_pred, y_pred, label="Predictions", linestyle="--")

        if not center_to_test_mean and test_mean is not None:
            test_mean_vals = test_mean.values
            test_mean_vals = normalize_array(test_mean_vals, normalize)
            x_test = np.sort(test_mean_vals)
            y_test = np.arange(1, len(test_mean_vals)+1) / len(test_mean_vals)
            ax.plot(x_test, y_test, label="Tests", linestyle="-")
        ax.legend(title="Dataset", loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    xlabel = f"{'Normalized ' if normalize else ''} {varname} {'[ppm]' if not normalize else ''}"
    title = f"CDF of {'Normalized ' if normalize else ''} {varname} per Sample"
    if center_to_test_mean:
        xlabel = f"Deviation from Test {varname} {'[ppm]' if not normalize else ''}"
        title = f"CDF of Predictions Relative to Test ({varname})"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout(rect=[0, 0, 0.75, 1])

    for fmt in imgformats:
        fig.savefig(out_dir / f"cdf_global{'_centered' if center_to_test_mean else ''}_{varname}.{fmt}", dpi=300, bbox_inches='tight')

    plt.close(fig)


def plot_crps(maps, df_global_scalars, out_dir,
              varname="co2molemix", level=None,
              imgformats=["svg", "png", "pdf"]):
    """
    Plot CRPS maps and global mean value.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if level is not None:
        title = f"CRPS ({varname}) - level {level:.0f}"
        level_str = f"_level{level:.0f}"
        crps_map = maps[f"CRPS_map_co2molemix_level{level:.0f}"]
        crps_mean = df_global_scalars[f"CRPS_ensemble_mean_level{level:.0f}"].iloc[0]
    else:
        title = f"CRPS ({varname}) - mean over levels"
        level_str = ""
        crps_map = maps["CRPS_map_co2molemix"]
        crps_mean = df_global_scalars["CRPS_ensemble_mean"].iloc[0]

    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(projection=ccrs.PlateCarree()))
    crps_map.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="cividis", add_colorbar=True, rasterized=True)
    ax.coastlines(linewidth=0.5)
    ax.set_title(title)
    fig.text(0.5, 0.01, f"Global mean CRPS = {crps_mean:.4f}", ha="center", fontsize=10)
    for fmt in imgformats:
        fig.savefig(out_dir / f"crps_{varname}{level_str}.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_preds_vs_tests(maps, df_global_scalars,
                                preds_var, tests_var, out_dir,
                                varname="co2molemix", level=None,
                                imgformats=["svg", "png", "pdf"]):
    """
    Scatter plot: ensemble mean predictions vs ground truth.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_samples = preds_var.sizes.get("sample", 1)
    n_points = np.prod(preds_var.sizes.get("lat", 1) * preds_var.sizes.get("lon", 1))

    if level is not None:
        title=f"Predicted vs Ground Truth\n({varname}) - level {level:.0f}"
        level_str = f"_level{level:.0f}"
        ens_mean = maps[f"Mean_map_co2molemix_level{level:.0f}"].values
        slope = df_global_scalars[f"LinReg_Slope_level{level:.0f}"].iloc[0]
        intercept = df_global_scalars[f"LinReg_Intercept_level{level:.0f}"].iloc[0]
        r_value = df_global_scalars[f"LinReg_R_value_level{level:.0f}"].iloc[0]
        rmse = df_global_scalars[f"RMSE_scalar_level{level:.0f}"].iloc[0]
        bias = df_global_scalars[f"Bias_scalar_level{level:.0f}"].iloc[0]
        crps = df_global_scalars[f"CRPS_ensemble_mean_level{level:.0f}"].iloc[0]
    else:
        title=f"Predicted vs Ground Truth\n({varname}) - mean over levels"
        level_str = ""
        ens_mean = maps["Mean_map_co2molemix"].values
        slope = df_global_scalars["LinReg_Slope"].iloc[0]
        intercept = df_global_scalars["LinReg_Intercept"].iloc[0]
        r_value = df_global_scalars["LinReg_R_value"].iloc[0]
        rmse = df_global_scalars["RMSE_scalar"].iloc[0]
        bias = df_global_scalars["Bias_scalar"].iloc[0]
        crps = df_global_scalars["CRPS_ensemble_mean"].iloc[0]

    y_true   = tests_var.values
    mask = np.isfinite(y_true) & np.isfinite(ens_mean)
    y_true, ens_mean = y_true[mask], ens_mean[mask]

    lims = [min(y_true.min(), ens_mean.min()), max(y_true.max(), ens_mean.max())]
    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6,6))
    hb = ax.hexbin(y_true, ens_mean, gridsize=100, cmap="cividis", bins="log")
    plt.colorbar(hb, ax=ax, label="log(count)")

    ax.plot(lims, lims, "k--", label="1:1 line")
    ax.plot(lims, [slope*lim + intercept for lim in lims], "r-", label=f"Trend (slope={slope:.2f})")
    ax.set_xlabel("Ground Truth (ppm)")
    ax.set_ylabel("Ensemble Mean Prediction (ppm)")
    ax.set_title(title)
    ax.legend()

    textstr = (
        f"samples = {n_samples}\n"
        f"points = {n_points}\n"
        f"R² = {r_value**2:.3f}\n"
        f"RMSE = {rmse:.2f} ppm\n"
        f"Bias = {bias:.2f} ppm\n"
        f"CRPS = {crps:.2f} ppm"
    )
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            va="top", ha="left", bbox=dict(facecolor="white", alpha=0.7))

    for fmt in imgformats:
        fig.savefig(out_dir / f"scatter_{varname}{level_str}.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)

        
def plot_spread_skill(maps, out_dir,
                      varname="co2molemix", level=None,
                      imgformats=["svg", "png", "pdf"]):
    """
    Plot ensemble spread (std of predictions) vs absolute error (ensemble mean vs ground truth).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if level is not None:
        title=f"Spread-Skill ({varname}) - level {level:.0f}"
        level_str = f"_level{level:.0f}"
        ens_std  = maps[f"Spread_map_co2molemix_level{level:.0f}"].values
        abs_error = np.abs(maps[f"Bias_map_co2molemix_level{level:.0f}"].values)
    else:
        title=f"Spread-Skill ({varname}) - mean over levels"
        level_str = ""
        ens_std  = maps["Spread_map_co2molemix"].values
        abs_error = np.abs(maps["Bias_map_co2molemix"].values)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6,6))
    hb = ax.hexbin(ens_std, abs_error, gridsize=80, cmap="magma", bins="log")
    plt.colorbar(hb, ax=ax, label="log(count)")

    lims = [0, max(ens_std.max(), abs_error.max())]
    ax.plot(lims, lims, "k--", label="1:1 line (perfect calibration)")
    ax.set_xlabel("Spread (std of ensemble predictions, ppm)")
    ax.set_ylabel("Absolute Error (mean vs ground truth, ppm)")
    ax.set_title(title)
    ax.legend()

    for fmt in imgformats:
        fig.savefig(out_dir / f"spread_skill_{varname}{level_str}.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_locations(maps, out_dir,
                         varname="co2molemix", level=None,
                         vmax_bias=5.0, vmax_rmse=10.0,
                         imgformats=["svg", "png", "pdf"]):
        """
        Plot spatial bias, RMSE and ensemble spread maps.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if level is not None:
            title=f"Spatial Diagnostics: Bias, RMSE, and Ensemble Spread ({varname}) - level {level:.0f}"
            level_str = f"_level{level:.0f}"
            bias = maps[f"Bias_map_co2molemix_level{level:.0f}"].values
            rmse = maps[f"RMSE_map_co2molemix_level{level:.0f}"].values
            spread = maps[f"Spread_map_co2molemix_level{level:.0f}"].values
        else:
            title=f"Spatial Diagnostics: Bias, RMSE, and Ensemble Spread ({varname}) - mean over levels"
            level_str = ""
            bias = maps["Bias_map_co2molemix"].values
            rmse = maps["RMSE_map_co2molemix"].values
            spread = maps["Spread_map_co2molemix"].values

        fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        titles = ["Bias [ppm]", "RMSE [ppm]", "Ensemble spread ($\\sigma$) [ppm]"]
        cmaps = ["RdBu_r", "inferno", "cividis"]
        data = [bias, rmse, spread]
        vmins = [-vmax_bias, 0, 0]
        vmaxs = [vmax_bias, vmax_rmse, vmax_rmse]

        for ax, arr, title, cmap, vmin, vmax in zip(axs, data, titles, cmaps, vmins, vmaxs):
            im = ax.imshow(arr[::-1, :], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(title, fontsize=14)
            plt.colorbar(im, ax=ax, shrink=0.7)

        plt.suptitle(title, fontsize=16, fontweight="bold")

        for fmt in imgformats:
            fig.savefig(out_dir / f"error_maps_{varname}{level_str}.{fmt}", dpi=300, bbox_inches="tight")
        plt.close(fig)
