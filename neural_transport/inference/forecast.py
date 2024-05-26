import shutil
import tempfile
import time as pytime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from cdo import Cdo
from tqdm import tqdm


def get_zarrpath_obspath(out_path, rollout, freq, zarr_filename=None):
    if zarr_filename is None:
        if freq:
            zarr_filename = (
                f"co2_pred_rollout_{freq}.zarr"
                if rollout
                else "co2_pred_singlestep.zarr"
            )
        else:
            zarr_filename = (
                "co2_pred_rollout.zarr" if rollout else "co2_pred_singlestep.zarr"
            )

    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True, parents=True)
    zarrpath = out_path / zarr_filename

    obspath = out_path / f"obs_{zarr_filename}"

    return zarrpath, obspath


def iterative_forecast_v2(
    model,
    dataset,
    out_path,
    rollout=False,
    device="cuda",
    verbose=False,
    zarr_filename=None,
    freq=None,
    zero_surfflux=False,
    remap=False,
    target_vars_3d=[],
    target_vars_2d=[],
):
    zarrpath, obspath = get_zarrpath_obspath(out_path, rollout, freq, zarr_filename)

    prototype_zarr = dataset.create_prototype_zarr(
        zarrpath,
        target_vars_3d=target_vars_3d,
        target_vars_2d=target_vars_2d,
        grid="default" if remap else None,
    )

    T = len(dataset)

    model = model.eval().to(device)

    if rollout and freq:
        time = pd.date_range(
            prototype_zarr.time[0].values,
            prototype_zarr.time[-1].values,
            freq=freq,
            inclusive="left",
        )  # freq =W, MS, QS or YS... also possible: QS-MAR
    else:
        time = [prototype_zarr.time[0].values]

    dss = []
    obss = []
    for t in tqdm(range(T), desc="Timestep") if verbose else range(T):
        batch = {k: v.unsqueeze(0).to(device) for k, v in dataset[t].items()}

        if rollout and (prototype_zarr.time[t].values not in time):
            for k in preds:
                batch[k] = preds[k]

        with torch.no_grad():
            preds = model(batch)

        preds["gp"] = batch["gp"]

        ds = xr.Dataset(
            {k: dataset.tensor_to_xarray(pred) for k, pred in preds.items()}
        )

        ds = ds.assign_coords(time=prototype_zarr.isel(time=t + 1).time)

        obs = dataset.readout_stations(ds)
        ds = ds.drop_vars(["gp"])

        dss.append(ds)
        obss.append(obs)

        if (t % 1000 == 999) or (t == T - 1):
            ds = xr.concat(dss, dim="time")

            ds_remap = (
                remap_with_cdo(dataset, prototype_zarr.isel(time=t), ds)
                if remap
                else ds
            )

            timeslice = slice(t - (len(ds_remap.time) - 2), t + 2)

            ds_remap.drop_vars(["height", "lat", "lon"]).to_zarr(
                zarrpath, region=dict(time=timeslice)
            )

            dss = []

    # ds = xr.open_zarr(zarrpath)
    # obs = xr.concat(
    #     [
    #         dataset.readout_stations(ds.isel(time=t), grid="default" if remap else None)
    #         for t in range(len(ds.time))
    #     ],
    #     dim="time",
    # ).fillna({"obs_filename": ""})

    obs = xr.concat(obss, dim="time").fillna({"obs_filename": ""})

    if obspath.exists():
        shutil.rmtree(obspath)
    obs.to_zarr(obspath)


def remap_with_cdo(dataset, prototype_zarr, ds):
    cdo = Cdo()
    prototype_zarr.lat.attrs = {
        "long_name": "latitude",
        "units": "degrees_north",
        "standard_name": "latitude",
    }
    prototype_zarr.lon.attrs = {
        "long_name": "longitude",
        "units": "degrees_east",
        "standard_name": "longitude",
    }

    ds["clon_vertices"] = dataset.grid_ds.clon_vertices
    ds["clat_vertices"] = dataset.grid_ds.clat_vertices
    ds["clon"] = dataset.grid_ds.clon
    ds["clat"] = dataset.grid_ds.clat
    ds["co2massmix"].attrs = {"CDI_grid_type": "unstructured"}
    ds["co2density"].attrs = {"CDI_grid_type": "unstructured"}
    ds["time"].attrs = {"standard_name": "time"}
    ds["height"].attrs = {"standard_name": "air_pressure"}

    grid_temp_file = tempfile.NamedTemporaryFile(
        delete=True, prefix="grid_temp_file_", dir=tempfile.gettempdir()
    )
    prototype_zarr.to_netcdf(grid_temp_file.name)

    ds_temp_file = tempfile.NamedTemporaryFile(
        delete=True, prefix="ds_temp_file_", dir=tempfile.gettempdir()
    )
    ds.transpose("time", "height", "cell", "nv").to_netcdf(ds_temp_file.name)

    ds_remap = cdo.remapcon(
        grid_temp_file.name, input=ds_temp_file.name, returnXDataset=True
    )

    grid_temp_file.close()
    ds_temp_file.close()

    cdo.cleanTempDir()
    return ds_remap


# %%
def iterative_forecast(
    model,
    dataset,
    out_path,
    rollout=False,
    device="cuda",
    verbose=False,
    zarr_filename=None,
    freq=None,
    zero_surfflux=False,
):
    prototype_zarr = (
        dataset.carboscope_ds[["co2density", "clon", "clat"]]
        .reset_encoding()
        .chunk({"time": 10, "cell": -1, "height": -1})
    )
    prototype_zarr["co2massmix"] = prototype_zarr["co2density"]

    if zarr_filename is None:
        if freq:
            zarr_filename = (
                f"co2_pred_rollout_{freq}.zarr"
                if rollout
                else "co2_pred_singlestep.zarr"
            )
        else:
            zarr_filename = (
                "co2_pred_rollout.zarr" if rollout else "co2_pred_singlestep.zarr"
            )

    Path(out_path).mkdir(exist_ok=True, parents=True)
    zarrpath = Path(out_path) / zarr_filename

    if zarrpath.exists():
        shutil.rmtree(zarrpath)

    prototype_zarr.to_zarr(zarrpath, compute=False)

    T = len(dataset)

    model = model.eval().to(device)

    if rollout and freq:
        time = pd.date_range(
            prototype_zarr.time[0].values,
            prototype_zarr.time[-1].values,
            freq=freq,
            inclusive="left",
        )  # freq =W, MS, QS or YS... also possible: QS-MAR
    else:
        time = [prototype_zarr.time[0].values]

    density_preds = []  # np.full((1, 20480, 19), np.NaN)]
    massmix_preds = []
    for t in tqdm(range(T), desc="Timestep") if verbose else range(T):
        # start = pytime.time()
        batch = {k: v.unsqueeze(0).to(device) for k, v in dataset[t].items()}
        # print(f"Data loading {pytime.time() - start}")

        if zero_surfflux:
            batch["co2flux_land"] = 0 * batch["co2flux_land"]
            batch["co2flux_ocean"] = 0 * batch["co2flux_ocean"]
            batch["co2flux_subt"] = 0 * batch["co2flux_subt"]

        if rollout and (prototype_zarr.time[t].values not in time):
            batch["co2density"] = density_pred
            batch["co2massmix"] = massmix_pred

        # start = pytime.time()
        with torch.no_grad():
            density_pred, massmix_pred = model(batch)
        # print(f"Model prediction {pytime.time() - start}")

        density_pred_npy = (
            density_pred[0, ...].cpu().numpy()
        )  # * (dataset.co2mix_max - dataset.co2mix_min) + dataset.co2mix_min
        massmix_pred_npy = massmix_pred[0, ...].cpu().numpy()
        density_preds.append(density_pred_npy)
        massmix_preds.append(massmix_pred_npy)

        if "lat" in prototype_zarr:
            curr_region = prototype_zarr.isel(
                time=slice(t + 1, t + 2)
            )  # .stack({"lat_lons": ("lat", "lon")}).transpose("time", "lat_lons", "height")

            curr_region["co2density"] = (
                ["time", "lat", "lon", "height"],
                density_pred_npy,
            )
            curr_region["co2massmix"] = (
                ["time", "lat", "lon", "height"],
                massmix_pred_npy,
            )

            curr_region.drop_vars(["lat", "lon", "height"]).transpose(
                "time", "height", "lat", "lon"
            ).to_zarr(
                zarrpath, region={"time": slice(t + 1, t + 2)}
            )  # .unstack("lat_lons").drop_vars(['lat', 'lon', 'height'])
        elif (t % 1000 == 999) or (t == T - 1):
            start = pytime.time()

            timeslice = slice(t - (len(density_preds) - 2), t + 2)

            curr_region = prototype_zarr.isel(
                time=timeslice
            )  # .stack({"lat_lons": ("lat", "lon")}).transpose("time", "lat_lons", "height")
            # breakpoint()
            curr_region["co2density"] = (
                ["time", "cell", "height"],
                np.concatenate(density_preds, axis=0),
            )
            curr_region["co2massmix"] = (
                ["time", "cell", "height"],
                np.concatenate(massmix_preds, axis=0),
            )
            # breakpoint()
            curr_region.drop_vars(["clon", "clat", "height"]).transpose(
                "time", "cell", "height"
            ).to_zarr(zarrpath, region={"time": timeslice})
            density_preds = []  # np.full((1, 20480, 19), np.NaN)]
            massmix_preds = []

            print(f"Writing.. {pytime.time() - start}")

    return
