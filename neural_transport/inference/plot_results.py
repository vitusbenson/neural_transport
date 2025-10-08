from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
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
from neural_transport.tools.conversion import *

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
        if "time" in Specpredf.dims:
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
                fig = plt_fct(pred, targ, metric)

                for imgformat in imgformats:
                    plt.savefig(
                        out_dir / f"{varname}_{metric}_{plottype}.{imgformat}", dpi=300
                    )

                plt.close()

        if zonal_spectrum:
            fig = plot_zonal_spectrum_line(pred, targ)

            for imgformat in imgformats:
                plt.savefig(
                    out_dir / f"{varname}_zonal_spectrum_line.{imgformat}", dpi=300
                )
            plt.close()

            fig = plot_zonal_spectrum_heatmap(pred, targ)

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
                        cbar = plt.colorbar(
                            cnf,
                            ax=axs[:2, :],
                            shrink=0.7,
                            label=kwargs.get("clabel", ""),
                        )
                    elif i == 2:
                        delta_cbar = plt.colorbar(
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

        # !!!Caution!!! this is a quick fix producing random movie snippets of different samples, not a continuous time frame, though is noise is generated in a path, it shows this path in latent space
        if "sample" in pred.dims and "time" in targ.dims:
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

    for _, row in subset.iterrows():
        try:
            i = np.where(filenames == row["id"])[0][0]
        except:
            continue

        with mpl.rc_context(mpl_rc_params):
            fig = plt.figure(figsize=(8, 5))
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

    figs = []

    # Norm vs θ
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(angles.cpu(), norms, marker="o")
    ax1.set_ylabel("Norm of noise vector")
    ax1.set_xlabel(label)
    ax1.set_title(f"Norm of x({label}) along path")
    ax1.grid(True)
    figs.append(fig1)

    # Cosine similarity vs θ
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.plot(angles.cpu(), cosine_sims, marker="o")
    ax2.set_ylabel("Cosine similarity with start")
    ax2.set_xlabel(label)
    ax2.set_title(f"Cosine similarity with starting noise")
    ax2.grid(True)
    figs.append(fig2)

    # PCA projection
    X = torch.stack(noises_flat).numpy()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ax3.plot(X_pca[:, 0], X_pca[:, 1], marker="o")
    for i, a in enumerate(angles):
        ax3.text(X_pca[i, 0], X_pca[i, 1], f"{a:.2f}")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_title("Noise path projected via PCA")
    figs.append(fig3)

    return figs


def plot_noise_diagnostics(noises, angles, out_dir, label="$\\theta$", imgformats=["svg", "png", "pdf"]):
    """
    Save diagnostics for noise interpolation or flow matching analysis.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    figs = plot_analyze_noise_path(noises, angles, label=label)
    names = ["norm_vs_angle", "cosine_vs_angle", "pca_projection"]

    for name, fig in zip(names, figs):
        for fmt in imgformats:
            fig.savefig(out_dir / f"noise_{name}.{fmt}", dpi=300)
        plt.close(fig)

    return


def plot_samples(
        preds,
        out_dir,
        tests=None,
        varnames=["co2massmix"],
        avg_over_levels=True,
        normalize=False,
        center_to_test_mean=False,
        imgformats=["svg", "png", "pdf"],
        generate_kwargs=None):
    """
    Wrapper for sample diagnostics plots
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_sample_cdf(preds, out_dir,
                    tests=tests,
                    varnames=varnames,
                    avg_over_levels=avg_over_levels,
                    normalize=normalize,
                    center_to_test_mean=center_to_test_mean,
                    imgformats=imgformats,)

    if generate_kwargs is not None and generate_kwargs.get("noise") is not None:
        plot_pairwise_sample_distances(preds, out_dir,
                                       varnames=varnames,
                                       avg_over_levels=avg_over_levels,
                                       imgformats=imgformats)


def plot_pairwise_sample_distances(preds, out_dir,
                                   varnames=["co2massmix"], avg_over_levels=True,
                                   imgformats=["svg", "png", "pdf"]):
    """
    Plot pairwise L2 distances between samples for a given variable at one vertical level.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for varname in varnames:
        if varname not in preds:
            raise KeyError(f"{varname} not found in dataset")

        preds_var = preds[varname]

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

            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(dist_matrix, cmap="cividis")
            fig.colorbar(im, ax=ax, label="L2 distance between samples")
            ax.set_title(f"Pairwise distances ({varname}" + (f", level={level}" if level is not None else "") + ")")
            ax.set_xlabel("Sample index")
            ax.set_ylabel("Sample index")

            for fmt in imgformats:
                fig.savefig(out_dir / f"pairwise_distances_{varname}" + (f"_level{level}" if level is not None else "") + f".{fmt}", dpi=300)

            plt.close(fig)


def plot_sample_cdf(preds, out_dir, tests=None,
                    varnames=["co2massmix"], avg_over_levels=True,
                    normalize=False,
                    center_to_test_mean=False,
                    imgformats=["svg", "png", "pdf"]):
    """
    Plot cumulative distribution functions (CDFs) of normalized mean values per sample
    for one or multiple variables. Optionally compare to test data.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for varname in varnames:
        if varname not in preds:
            raise KeyError(f"{varname} not found in dataset")

        preds_var = preds[varname]

        if "level" in preds_var.dims and avg_over_levels:
            preds_var = preds_var.mean(dim="level")  # shape: (sample, lat, lon)

        pred_mean = preds_var.mean(dim=["lat", "lon"]).values  # shape: (sample)

        if tests is not None:
            tests_var = tests[varname]
            if isinstance(tests_var, torch.Tensor):
                tests_var = tests_var.cpu().numpy()
            if tests_var.ndim == 3:  # [B, lat, lon]
                dims = ("sample", "lat", "lon")
            elif tests_var.ndim == 4:  # [B, level, lat, lon]
                dims = ("sample", "level", "lat", "lon")
            else:
                raise ValueError(f"tests must be 3D or 4D, got shape {tests_var.shape}")

            test_da = xr.DataArray(tests_var, dims=dims)
            avg_dims = [d for d in ["level", "lat", "lon"] if d in test_da.dims]
            test_mean = test_da.mean(dim=avg_dims).values

            if normalize:
                test_mean = (test_mean - test_mean.mean()) / test_mean.std()
        else:
            test_mean = None

        if center_to_test_mean and test_mean is not None:
            pred_mean = pred_mean - test_mean.mean()
            test_mean = None

        if normalize:
            pred_mean = (pred_mean - pred_mean.mean()) / pred_mean.std()

        # CDF values
        x_pred = np.sort(pred_mean)
        y_pred = np.arange(1, len(pred_mean)+1) / len(pred_mean)

        fig, ax = plt.subplots(figsize=(8, 5))

        if test_mean is not None:
            x_test = np.sort(test_mean)
            y_test = np.arange(1, len(test_mean)+1) / len(test_mean)
            ax.plot(x_test, y_test, label="Tests", marker="o")
        if center_to_test_mean:
            ax.plot(x_pred, y_pred, color="red")
        else:
            ax.plot(x_pred, y_pred, label="Predictions", marker="x")

        xlabel = f"{'Normalized ' if normalize else ''}Mean {varname} {'[ppm]' if not normalize else ''}"
        title = f"CDF of {'Normalized ' if normalize else ''}Mean {varname} per Sample"
        if center_to_test_mean:
            xlabel = f"Deviation from Test Mean {varname} {'[ppm]' if not normalize else ''}"
            title = f"CDF of Predictions Relative to Test Mean ({varname})"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CDF")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        for fmt in imgformats:
            fig.savefig(out_dir / f"cdf_{varname}.{fmt}", dpi=300)

        plt.close(fig)
