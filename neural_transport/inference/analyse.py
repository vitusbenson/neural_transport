from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xskillscore
from scipy.stats import linregress
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from neural_transport.tools.conversion import (
    M_C,
    M_CO2,
    density_to_massmix,
    massmix_to_molemix,
)
from neural_transport.tools.metrics import (
    compute_error_maps,
    compute_error_scalars,
    crps,
)

# Ratio of CO2 molecular mass to carbon atomic mass (~3.664)
M_CO2_OVER_M_C = M_CO2 / M_C


def freq_mean(data, freq="QS", average_time=False):
    if "sample" in data.dims:
        data = data.mean("sample")

    if "time" not in data.dims:
        return data

    if freq:
        dates = pd.date_range(data.time[0].values, data.time[-1].values, freq=freq, inclusive="left")
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
        dataf["time"] = dataf.time / (pd.Timedelta(days=1) / data.time.diff("time").values[0]).astype("timedelta64[D]")
    except Exception as e:
        dataf["time"] = dataf.time / 4
        print(f"Could not convert time bins to days properly, defaulting to 4 steps per day: {e}")

    if average_time:
        dataf = dataf.mean("time")

    return dataf


def get_first_idx_below_threshold(data, threshold=0.8, freq="QS"):
    agg_data = freq_mean(data.copy(deep=True), freq=freq).isel(time=slice(1, None))

    idxs_below_threshold = agg_data.compute().where(lambda x: x < threshold, drop=True).time.values

    if len(idxs_below_threshold) == 0:
        return agg_data.time.values[-1]
    else:
        return idxs_below_threshold[0]


def compute_score_df(targs, preds, freq="QS"):
    preds["lat"] = targs["lat"]
    preds["lon"] = targs["lon"]
    preds["level"] = targs["level"]
    molemix_targ = massmix_to_molemix(targs.co2massmix).persist().transpose("time", "level", "lat", "lon")

    molemix_pred = massmix_to_molemix(preds.co2massmix).persist().transpose("time", "level", "lat", "lon")

    # make weights as cosine of the latitude and broadcast
    weights = np.cos(np.deg2rad(targs.lat))
    _, weights = xr.broadcast(targs, weights)

    # Remove the time dimension from weights
    weights = weights.isel(time=0)

    metrics = {}

    ### Metrics

    # Mass RMSE
    # R^2, NSE, RMSE, Rel RMSE, Abs Bias, Rel Abs Bias
    # 3D, per layer, per grid cell, per voxel (in time)
    # num steps before R^2 < 0.8
    # For Taylor Plot: RMSE, sigma_pred, sigma_targ, pearson corr coef
    targ_mass = (targs.co2massmix * targs.airmass) / 1e6
    pred_mass = (preds.co2massmix * targs.airmass) / 1e6

    targ_mass_sum = targ_mass.sum(["lat", "lon", "level"]).compute() / M_CO2_OVER_M_C
    pred_mass_sum = pred_mass.sum(["lat", "lon", "level"]).compute() / M_CO2_OVER_M_C

    metrics["Mass_RMSE"] = ((targ_mass_sum - pred_mass_sum) ** 2).mean().compute().item() ** 0.5

    metrics["RelMass_RMSE"] = (((targ_mass_sum - pred_mass_sum) / targ_mass_sum) ** 2).mean().compute().item() ** 0.5

    rmsef = freq_mean((targ_mass_sum - pred_mass_sum) ** 2, freq="QS") ** 0.5
    relrmsef = freq_mean(((targ_mass_sum - pred_mass_sum) / targ_mass_sum) ** 2, freq="QS") ** 0.5
    for days in [7, 30, 60, 90]:
        metrics[f"Mass_RMSE_{days}d"] = rmsef.isel(time=days * 4).mean().compute().item()
        metrics[f"RelMass_RMSE_{days}d"] = relrmsef.isel(time=days * 4).mean().compute().item()

    for conc, targ, pred in [
        ("co2molemix", molemix_targ, molemix_pred),
    ]:
        mse = xskillscore.mse(
            targ.chunk({"lat": -1, "lon": -1, "level": -1}),
            pred.chunk({"lat": -1, "lon": -1, "level": -1}),
            dim=["lat", "lon", "level"],
            weights=weights,
        ).compute()  # ((pred - targ) ** 2).mean().compute().item()
        metrics[f"RMSE_4D_{conc}"] = mse.mean().item() ** 0.5

        metrics[f"StdDev_Targ_4D_{conc}"] = targ.weighted(np.cos(np.deg2rad(targ.lat))).std().compute().item()
        metrics[f"StdDev_Pred_4D_{conc}"] = pred.weighted(np.cos(np.deg2rad(targ.lat))).std().compute().item()

        r = xskillscore.pearson_r(
            targ.chunk({"lat": -1, "lon": -1, "level": -1}),
            pred.chunk({"lat": -1, "lon": -1, "level": -1}),
            dim=["lat", "lon", "level"],
            weights=weights,
        ).compute()
        metrics[f"PearsonCorrCoef_3D_{conc}"] = r.mean().item()

        metrics[f"R2_3D_{conc}"] = (r**2).mean().item()

        r2f = freq_mean(r**2, freq="QS")
        rmsef = freq_mean(mse, freq="QS") ** 0.5
        for days in [7, 30, 60, 90]:
            metrics[f"R2_3D_{days}d_{conc}"] = r2f.isel(time=days * 4).mean().compute().item()
            metrics[f"RMSE_3D_{days}d_{conc}"] = rmsef.isel(time=days * 4).mean().item()

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

        targ_mean = targ.weighted(np.cos(np.deg2rad(targ.lat))).mean().compute().item()

        metrics[f"RelRMSE_3D_{conc}"] = (mse.mean().item() ** 0.5) / (targ_mean + 1e-12)

        metrics[f"Days_R2>0.8_{conc}"] = get_first_idx_below_threshold(r**2, threshold=0.8, freq=freq)
        metrics[f"Days_R2>0.9_{conc}"] = get_first_idx_below_threshold(r**2, threshold=0.9, freq=freq)

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
        metrics[f"Days_minR2>0.9_{conc}"] = get_first_idx_below_threshold(r2m, threshold=0.9, freq=freq)

        for dim in ["lat", "lon", "level"]:
            mse = ((pred - targ) ** 2).weighted(np.cos(np.deg2rad(targ.lat))).mean(dim).compute()
            pred_mean = pred.weighted(np.cos(np.deg2rad(targ.lat))).mean(dim).compute()
            targ_mean = targ.weighted(np.cos(np.deg2rad(targ.lat))).mean(dim).compute()
            absbias = np.abs(pred_mean - targ_mean).compute()

            metrics[f"RMSE_{dim}_{conc}"] = (mse**0.5).mean().item()
            metrics[f"RelRMSE_{dim}_{conc}"] = ((mse**0.5) / (targ_mean + 1e-12)).mean().item()

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
                r2.mean().item() if dim == "lat" else (r2).weighted(np.cos(np.deg2rad(targ.lat))).mean().item()
            )  # (xr.corr(targ, pred, dim = dim)**2).mean().compute().item()

            nse = xskillscore.r2(
                targ.chunk({dim: -1}),
                pred.chunk({dim: -1}),
                dim=dim,
                weights=weights.isel(lon=0, level=0) if dim == "lat" else None,
            ).compute()
            metrics[f"NSE_{dim}_{conc}"] = (
                nse.median().item()
            )  # if dim == "lat" else (nse).weighted(np.cos(np.deg2rad(targ.lat))).median().item() # (1 - mse / (targ.var([dim]) + 1e-12)).compute().median().item()

            metrics[f"AbsBias_{dim}_{conc}"] = (absbias).mean().item()
            metrics[f"RelAbsBias_{dim}_{conc}"] = (absbias / (targ_mean + 1e-12)).mean().item()
    print("Computed metrics:", metrics)
    df = pd.Series(metrics)

    return df


def compute_score_df_generate(targs, preds, target_var="co2massmix", **generate_kwargs):
    """
    Compute metrics for generated samples where:
      - targs : xr.Dataset, [time, level, lat, lon]
      - preds : xr.Dataset, [sample, level, lat, lon] (or [sample, (flow)time, level, lat, lon] if trajectory)
    """

    preds["lat"] = targs["lat"]
    preds["lon"] = targs["lon"]
    preds["level"] = targs["level"]

    targs = targs.isel(time=-1)
    if "time" in preds.dims:
        preds = preds.isel(time=-1)

    # Convert to mole fraction
    molemix_targ = massmix_to_molemix(targs[target_var]).transpose("level", "lat", "lon")
    molemix_pred = massmix_to_molemix(preds[target_var]).transpose("sample", "level", "lat", "lon")

    # Weights (cos(lat)) – no time dependence, just lat dimension
    weights = np.cos(np.deg2rad(targs.lat))
    _, weights = xr.broadcast(targs[target_var], weights)

    results = []

    for i in range(preds.sizes["sample"]):
        pred_i = molemix_pred.isel(sample=i)

        metrics = {}

        # Mass balance metrics
        if "airmass" in targs and "airmass" in preds:
            targ_mass = (targs[target_var] * targs.airmass) / 1e6
            pred_mass = (preds[target_var].isel(sample=i) * targs.airmass) / 1e6
        else:
            targ_mass = targs[target_var]
            pred_mass = preds[target_var].isel(sample=i)

        targ_mass_sum = targ_mass.sum(["lat", "lon", "level"]).compute() / M_CO2_OVER_M_C
        pred_mass_sum = pred_mass.sum(["lat", "lon", "level"]).compute() / M_CO2_OVER_M_C

        metrics["Mass_RMSE"] = ((targ_mass_sum - pred_mass_sum) ** 2).mean().item() ** 0.5

        metrics["RelMass_RMSE"] = (
            ((targ_mass_sum - pred_mass_sum) / (targ_mass_sum + 1e-12)) ** 2
        ).mean().item() ** 0.5

        ### RMSE / R² across lat, lon, level
        mse = xskillscore.mse(
            molemix_targ.chunk({"lat": -1, "lon": -1, "level": -1}),
            pred_i.chunk({"lat": -1, "lon": -1, "level": -1}),
            dim=["lat", "lon", "level"],
            weights=weights,
        ).compute()

        metrics["RMSE_3D_co2molemix"] = mse.item() ** 0.5

        r = xskillscore.pearson_r(
            molemix_targ.chunk({"lat": -1, "lon": -1, "level": -1}),
            pred_i.chunk({"lat": -1, "lon": -1, "level": -1}),
            dim=["lat", "lon", "level"],
            weights=weights,
        ).compute()

        metrics["PearsonCorrCoef_3D_co2molemix"] = r.item()
        metrics["R2_3D_co2molemix"] = (r**2).item()

        # Relative RMSE
        targ_mean = molemix_targ.weighted(weights).mean().compute().item()
        metrics["RelRMSE_3D_co2molemix"] = (mse.item() ** 0.5) / (targ_mean + 1e-12)

        # Per-dimension metrics (lat, lon, level)
        for dim in ["lat", "lon", "level"]:
            mse_dim = ((pred_i - molemix_targ) ** 2).weighted(weights).mean(dim).compute()
            metrics[f"RMSE_{dim}_co2molemix"] = float(mse_dim.mean() ** 0.5)

        results.append(metrics)

    df = pd.DataFrame(results)
    df.loc["mean"] = df.mean()
    df.loc["std"] = df.std()

    molemix_pred = molemix_pred.transpose("sample", "lat", "lon", "level")
    avg_over_levels = generate_kwargs.get("avg_over_levels", True)

    global_scalars = {}
    maps = {}
    if not avg_over_levels:
        for i, lvl in enumerate(molemix_targ.level.values):
            molemix_pred_lvl = molemix_pred.isel(level=i)
            molemix_targ_lvl = molemix_targ.isel(level=i)

            crps_map_level, crps_mean_level = crps(molemix_pred_lvl, molemix_targ_lvl)
            maps[f"CRPS_map_co2molemix_level{lvl:.0f}"] = (("lat", "lon"), crps_map_level.data)
            global_scalars[f"CRPS_ensemble_mean_level{lvl:.0f}"] = float(crps_mean_level)

            bias_map, rmse_map, mean_map, spread_map = compute_error_maps(molemix_pred_lvl, molemix_targ_lvl)
            maps[f"Bias_map_co2molemix_level{lvl:.0f}"] = (("lat", "lon"), bias_map)
            maps[f"RMSE_map_co2molemix_level{lvl:.0f}"] = (("lat", "lon"), rmse_map)
            maps[f"Mean_map_co2molemix_level{lvl:.0f}"] = (("lat", "lon"), mean_map)
            maps[f"Spread_map_co2molemix_level{lvl:.0f}"] = (("lat", "lon"), spread_map)

            slope, intercept, r_value, p_value, std_err = linregress(
                molemix_targ_lvl.values.flatten(), mean_map.flatten()
            )
            global_scalars[f"LinReg_Slope_level{lvl:.0f}"] = float(slope)
            global_scalars[f"LinReg_Intercept_level{lvl:.0f}"] = float(intercept)
            global_scalars[f"LinReg_R_value_level{lvl:.0f}"] = float(r_value)
            global_scalars[f"LinReg_PValue_level{lvl:.0f}"] = float(p_value)
            global_scalars[f"LinReg_StdErr_level{lvl:.0f}"] = float(std_err)

            bias_scalar, rmse_scalar, mean_scalar, spread_scalar = compute_error_scalars(
                bias_map, rmse_map, mean_map, spread_map, weights=weights.isel(level=i).values
            )
            global_scalars[f"Bias_scalar_level{lvl:.0f}"] = float(bias_scalar)
            global_scalars[f"RMSE_scalar_level{lvl:.0f}"] = float(rmse_scalar)
            global_scalars[f"Mean_scalar_level{lvl:.0f}"] = float(mean_scalar)
            global_scalars[f"Spread_scalar_level{lvl:.0f}"] = float(spread_scalar)

    molemix_pred_mean = molemix_pred.mean(dim="level")  # shape: [sample, lat, lon]
    molemix_targ_mean = molemix_targ.mean(dim="level")  # shape: [lat, lon]

    crps_map, crps_mean = crps(molemix_pred_mean, molemix_targ_mean)
    maps["CRPS_map_co2molemix"] = (("lat", "lon"), crps_map.data)
    global_scalars["CRPS_ensemble_mean"] = float(crps_mean)

    bias_map, rmse_map, mean_map, spread_map = compute_error_maps(molemix_pred_mean, molemix_targ_mean)
    maps["Bias_map_co2molemix"] = (("lat", "lon"), bias_map)
    maps["RMSE_map_co2molemix"] = (("lat", "lon"), rmse_map)
    maps["Mean_map_co2molemix"] = (("lat", "lon"), mean_map)
    maps["Spread_map_co2molemix"] = (("lat", "lon"), spread_map)

    slope, intercept, r_value, p_value, std_err = linregress(molemix_targ_mean.values.flatten(), mean_map.flatten())
    global_scalars["LinReg_Slope"] = float(slope)
    global_scalars["LinReg_Intercept"] = float(intercept)
    global_scalars["LinReg_R_value"] = float(r_value)
    global_scalars["LinReg_PValue"] = float(p_value)
    global_scalars["LinReg_StdErr"] = float(std_err)

    bias_scalar, rmse_scalar, mean_scalar, spread_scalar = compute_error_scalars(
        bias_map, rmse_map, mean_map, spread_map, weights=weights.isel(level=0).values
    )
    global_scalars["Bias_scalar"] = float(bias_scalar)
    global_scalars["RMSE_scalar"] = float(rmse_scalar)
    global_scalars["Mean_scalar"] = float(mean_scalar)
    global_scalars["Spread_scalar"] = float(spread_scalar)

    df_global_scalars = pd.DataFrame({k: [v] for k, v in global_scalars.items()})

    maps = xr.Dataset(
        data_vars={k: (dims, data) for k, (dims, data) in maps.items()},
        coords={
            "lat": molemix_targ.lat,
            "lon": molemix_targ.lon,
            "level": molemix_targ.level,
        },
    )

    print(f"Computed metrics: {results}")
    return df, df_global_scalars, maps


def compute_distributional_score_df(
    gt_anomalies,
    gen_anomalies,
    target_var="co2massmix",
):
    """Compare GT and generated anomaly distributions.

    Args:
        gt_anomalies: xr.Dataset [sample, lat, lon, level] — GT CO2 anomaly patterns
        gen_anomalies: xr.Dataset [sample, lat, lon, level] — generated CO2 anomaly patterns
        target_var: variable name to compare

    Returns:
        pd.DataFrame with all metric values.
    """
    from neural_transport.inference.distributional_metrics import compute_distributional_metrics

    gt_fields = gt_anomalies[target_var].values  # [N, nlat, nlon, nlev]
    gen_fields = gen_anomalies[target_var].values  # [M, nlat, nlon, nlev]

    lat = gt_anomalies.lat.values
    lon = gt_anomalies.lon.values

    metrics = compute_distributional_metrics(gt_fields, gen_fields, lat, lon)

    # Flatten nested dicts and convert to DataFrame
    flat_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat_metrics[f"{k}/{k2}"] = v2
        else:
            flat_metrics[k] = v

    df = pd.DataFrame({k: [v] for k, v in flat_metrics.items()})
    return df


def compute_local_scores(obs_preds, freq="QS"):
    obs_preds = obs_preds.compute()
    if "co2molemix" not in obs_preds:
        if "co2massmix" not in obs_preds:
            obs_preds["co2molemix"] = massmix_to_molemix(
                density_to_massmix(obs_preds["co2density"], obs_preds["airdensity"], ppm=True)
            )
        else:
            obs_preds["co2molemix"] = massmix_to_molemix(obs_preds["co2massmix"])

    rmse = ((obs_preds.obs_co2molemix - obs_preds.co2molemix) ** 2).mean("time") ** 0.5
    r2 = xskillscore.pearson_r(obs_preds.obs_co2molemix, obs_preds.co2molemix, dim="time", skipna=True) ** 2
    nse = xskillscore.r2(obs_preds.obs_co2molemix, obs_preds.co2molemix, dim="time", skipna=True)
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
                [dict(wall_time=e.wall_time, name=name, step=e.step, value=e.value) for e in event_acc.Scalars(name)]
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
