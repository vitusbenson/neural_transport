import time as pytime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xskillscore
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from neural_transport.tools.conversion import *


def freq_mean(data, freq="QS"):
    if freq:
        dates = pd.date_range(
            data.time[0].values, data.time[-1].values, freq=freq, inclusive="left"
        )
    else:
        dates = [data.time[0].values]

    bins = []
    bin = 0
    for date in data.time.values:
        if date in dates:
            bin = 0
        bins.append(bin)
        bin += 1

    data["time"] = bins

    dataf = data.groupby("time", squeeze=False).mean()

    try:
        dataf["time"] = dataf.time / (
            pd.Timedelta(days=1) / data.time.diff("time").values[0]
        ).astype("timedelta64[D]")
    except:
        dataf["time"] = dataf.time / 4

    return dataf


def get_first_idx_below_threshold(data, threshold=0.8, freq="QS"):
    agg_data = freq_mean(data.copy(deep=True), freq=freq).isel(time=slice(1, None))

    idxs_below_threshold = (
        agg_data.compute().where(lambda x: x < threshold, drop=True).time.values
    )

    if len(idxs_below_threshold) == 0:
        return agg_data.time.values[-1]
    else:
        return idxs_below_threshold[0]


def compute_score_df(targs, preds, freq="QS"):

    preds["lat"] = targs["lat"]
    preds["lon"] = targs["lon"]
    preds["level"] = targs["level"]
    start = pytime.time()

    molemix_targ = (
        massmix_to_molemix(targs.co2massmix)
        .persist()
        .transpose("time", "level", "lat", "lon")
    )

    molemix_pred = (
        massmix_to_molemix(preds.co2massmix)
        .persist()
        .transpose("time", "level", "lat", "lon")
    )

    # make weights as cosine of the latitude and broadcast
    weights = np.cos(np.deg2rad(targs.lat))
    _, weights = xr.broadcast(targs, weights)

    # Remove the time dimension from weights
    weights = weights.isel(time=0)

    print(f"Data Loading {pytime.time() - start}")

    metrics = {}

    ### Metrics

    # Mass RMSE
    # R^2, NSE, RMSE, Rel RMSE, Abs Bias, Rel Abs Bias
    # 3D, per layer, per grid cell, per voxel (in time)
    # num steps before R^2 < 0.8
    # For Taylor Plot: RMSE, sigma_pred, sigma_targ, pearson corr coef
    start = pytime.time()

    targ_mass = (targs.co2massmix * targs.airmass) / 1e6
    pred_mass = (preds.co2massmix * targs.airmass) / 1e6

    targ_mass_sum = targ_mass.sum(["lat", "lon", "level"]).compute() / 3.664
    pred_mass_sum = pred_mass.sum(["lat", "lon", "level"]).compute() / 3.664

    metrics["Mass_RMSE"] = (
        (targ_mass_sum - pred_mass_sum) ** 2
    ).mean().compute().item() ** 0.5
    print(f"Mass RMSE {pytime.time() - start}")

    metrics["RelMass_RMSE"] = (
        ((targ_mass_sum - pred_mass_sum) / targ_mass_sum) ** 2
    ).mean().compute().item() ** 0.5

    rmsef = freq_mean((targ_mass_sum - pred_mass_sum) ** 2, freq="QS") ** 0.5
    relrmsef = (
        freq_mean(((targ_mass_sum - pred_mass_sum) / targ_mass_sum) ** 2, freq="QS")
        ** 0.5
    )
    for days in [7, 30, 60, 90]:
        metrics[f"Mass_RMSE_{days}d"] = (
            rmsef.isel(time=days * 4).mean().compute().item()
        )
        metrics[f"RelMass_RMSE_{days}d"] = (
            relrmsef.isel(time=days * 4).mean().compute().item()
        )

    for conc, targ, pred in [
        ("co2molemix", molemix_targ, molemix_pred),
    ]:
        start = pytime.time()
        mse = xskillscore.mse(
            targ.chunk({"lat": -1, "lon": -1, "level": -1}),
            pred.chunk({"lat": -1, "lon": -1, "level": -1}),
            dim=["lat", "lon", "level"],
            weights=weights,
        ).compute()  # ((pred - targ) ** 2).mean().compute().item()
        metrics[f"RMSE_4D_{conc}"] = mse.mean().item() ** 0.5
        print(f"RMSE 4D {pytime.time() - start}")
        print(metrics)

        start = pytime.time()
        metrics[f"StdDev_Targ_4D_{conc}"] = (
            targ.weighted(np.cos(np.deg2rad(targ.lat))).std().compute().item()
        )
        metrics[f"StdDev_Pred_4D_{conc}"] = (
            pred.weighted(np.cos(np.deg2rad(targ.lat))).std().compute().item()
        )
        print(f"Std Devs 4D {pytime.time() - start}")
        print(metrics)

        start = pytime.time()
        r = xskillscore.pearson_r(
            targ.chunk({"lat": -1, "lon": -1, "level": -1}),
            pred.chunk({"lat": -1, "lon": -1, "level": -1}),
            dim=["lat", "lon", "level"],
            weights=weights,
        ).compute()
        metrics[f"PearsonCorrCoef_3D_{conc}"] = r.mean().item()
        print(f"PearsonCorrCoef_3D_ {pytime.time() - start}")

        metrics[f"R2_3D_{conc}"] = (r**2).mean().item()
        print(metrics)

        r2f = freq_mean(r**2, freq="QS")
        rmsef = freq_mean(mse, freq="QS") ** 0.5
        for days in [7, 30, 60, 90]:
            metrics[f"R2_3D_{days}d_{conc}"] = (
                r2f.isel(time=days * 4).mean().compute().item()
            )
            metrics[f"RMSE_3D_{days}d_{conc}"] = rmsef.isel(time=days * 4).mean().item()

        start = pytime.time()
        metrics[f"NSE_3D_{conc}"] = (
            xskillscore.r2(
                targ.chunk({"lat": -1, "lon": -1, "level": -1}),
                pred.chunk({"lat": -1, "lon": -1, "level": -1}),
                dim=["lat", "lon", "level"],
                weights=weights,
            )
            .compute()
            .median()
            .item()
        )  # 1 - mse / (metrics[f"StdDev_Targ_3D_{conc}"]**2 + 1e-12)
        print(f"NSE_3D_ {pytime.time() - start}")

        start = pytime.time()

        targ_mean = targ.weighted(np.cos(np.deg2rad(targ.lat))).mean().compute().item()

        metrics[f"RelRMSE_3D_{conc}"] = (mse.mean().item() ** 0.5) / (targ_mean + 1e-12)

        print(f"RelRMSE_3D_ {pytime.time() - start}")

        start = pytime.time()
        # try:

        metrics[f"Days_R2>0.8_{conc}"] = get_first_idx_below_threshold(
            r**2, threshold=0.8, freq=freq
        )
        metrics[f"Days_R2>0.9_{conc}"] = get_first_idx_below_threshold(
            r**2, threshold=0.9, freq=freq
        )

        r2m = (
            (
                xskillscore.pearson_r(
                    targ.chunk({"lat": -1, "lon": -1}),
                    pred.chunk({"lat": -1, "lon": -1}),
                    dim=["lat", "lon"],
                    weights=weights.isel(level=0),
                )
                ** 2
            )
            .min("level")
            .compute()
        )
        metrics[f"Days_minR2>0.8_{conc}"] = get_first_idx_below_threshold(
            r2m, threshold=0.8, freq=freq
        )  # This Takes first Min(Level), then Freq_mean --> in plot_results is done other way around
        # except:
        #     metrics[f"Days_R2>0.8_{conc}"] = 92
        metrics[f"Days_minR2>0.9_{conc}"] = get_first_idx_below_threshold(
            r2m, threshold=0.9, freq=freq
        )
        print(f"Days_R2 {pytime.time() - start}")
        print(metrics)

        for dim in ["lat", "lon", "level"]:  # , "time"]:
            start = pytime.time()
            mse = (
                ((pred - targ) ** 2)
                .weighted(np.cos(np.deg2rad(targ.lat)))
                .mean(dim)
                .compute()
            )
            pred_mean = pred.weighted(np.cos(np.deg2rad(targ.lat))).mean(dim).compute()
            targ_mean = targ.weighted(np.cos(np.deg2rad(targ.lat))).mean(dim).compute()
            absbias = np.abs(pred_mean - targ_mean).compute()

            metrics[f"RMSE_{dim}_{conc}"] = (mse**0.5).mean().item()
            metrics[f"RelRMSE_{dim}_{conc}"] = (
                ((mse**0.5) / (targ_mean + 1e-12)).mean().item()
            )
            print(f"RMSE RelRMSE {dim} {pytime.time() - start}")

            start = pytime.time()
            r2 = (
                xskillscore.pearson_r(
                    targ.chunk({dim: -1}),
                    pred.chunk({dim: -1}),
                    dim=dim,
                    weights=weights.isel(lon=0, level=0) if dim == "lat" else None,
                ).compute()
                ** 2
            )
            metrics[f"R2_{dim}_{conc}"] = (
                r2.mean().item()
                if dim == "lat"
                else (r2).weighted(np.cos(np.deg2rad(targ.lat))).mean().item()
            )  # (xr.corr(targ, pred, dim = dim)**2).mean().compute().item()
            print(f"R2 {dim} {conc} {pytime.time() - start}")

            start = pytime.time()

            nse = xskillscore.r2(
                targ.chunk({dim: -1}),
                pred.chunk({dim: -1}),
                dim=dim,
                weights=weights.isel(lon=0, level=0) if dim == "lat" else None,
            ).compute()
            metrics[f"NSE_{dim}_{conc}"] = (
                nse.median().item()
            )  # if dim == "lat" else (nse).weighted(np.cos(np.deg2rad(targ.lat))).median().item() # (1 - mse / (targ.var([dim]) + 1e-12)).compute().median().item()
            print(f"NSE {dim} {conc} {pytime.time() - start}")

            start = pytime.time()
            metrics[f"AbsBias_{dim}_{conc}"] = (absbias).mean().item()
            metrics[f"RelAbsBias_{dim}_{conc}"] = (
                (absbias / (targ_mean + 1e-12)).mean().item()
            )
            print(f"AbsBias RelAbsBias {dim} {conc} {pytime.time() - start}")
            print(metrics)

        # start = pytime.time()
        # metrics[f"PearsonCorrCoef_4D_{conc}"] = (xskillscore.pearson_r(targ.compute(), pred.compute(), dim = ["lat", "lon", "level", "time"]).compute()).item()
        # print(f"PearsonCorrCoef_4D_ {pytime.time() - start}")
        # print(metrics)

    df = pd.Series(metrics)

    return df


def compute_local_scores(obs_preds, freq="QS"):
    obs_preds = obs_preds.compute()
    if "co2molemix" not in obs_preds:
        if "co2massmix" not in obs_preds:
            obs_preds["co2molemix"] = massmix_to_molemix(
                density_to_massmix(
                    obs_preds["co2density"], obs_preds["airdensity"], ppm=True
                )
            )
        else:
            obs_preds["co2molemix"] = massmix_to_molemix(obs_preds["co2massmix"])

    rmse = ((obs_preds.obs_co2molemix - obs_preds.co2molemix) ** 2).mean("time") ** 0.5
    r2 = (
        xskillscore.pearson_r(
            obs_preds.obs_co2molemix, obs_preds.co2molemix, dim="time", skipna=True
        )
        ** 2
    )
    nse = xskillscore.r2(
        obs_preds.obs_co2molemix, obs_preds.co2molemix, dim="time", skipna=True
    )
    bias = obs_preds.obs_co2molemix.mean("time") - obs_preds.co2molemix.mean("time")
    relbias = bias / obs_preds.obs_co2molemix.mean("time")

    ds = xr.Dataset(
        {
            "obs_filename": obs_preds.obs_filename.max("time"),
            "obs_height": obs_preds.obs_height.mean("time"),
            "obs_lat": obs_preds.obs_lat.mean("time"),
            "obs_lon": obs_preds.obs_lon.mean("time"),
            "rmse": rmse,
            "r2": r2,
            "nse": nse,
            "bias": bias,
            "relbias": relbias,
        }
    )
    return ds.to_array("vari").transpose("cell", "vari").to_pandas()


def get_tensorboard_df(runpath):
    runpath = Path(runpath)
    eventpaths = list(runpath.glob("**/events.out.tfevents*"))

    if len(eventpaths) > 1:
        print("Found more than one tf event, using the last one")

    event_acc = EventAccumulator(str(eventpaths[-1]))

    def new_proc_img(tag, wall_time, step, image):
        pass

    event_acc._ProcessImage = new_proc_img

    event_acc.Reload()

    df = pd.concat(
        [
            pd.DataFrame(
                [
                    dict(wall_time=e.wall_time, name=name, step=e.step, value=e.value)
                    for e in event_acc.Scalars(name)
                ]
            )
            for name in event_acc.Tags()["scalars"]
        ]
    )

    df2 = df.pivot_table(
        values=(["value"]),
        index=["step"],
        columns="name",
        dropna=False,
    )
    df2.columns = df2.columns.droplevel(0)
    df2.columns.name = None

    return df2
