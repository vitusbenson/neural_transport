import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from dask.diagnostics import ProgressBar

from neural_transport.datasets.grids import LATLON_PROTOTYPE_COORDS


def get_area_for_latlongrid(gridname=None, coords=None):

    coords = coords or LATLON_PROTOTYPE_COORDS[gridname]

    latstep = np.abs(np.diff(coords["lat"])).mean()
    lonstep = np.abs(np.diff(coords["lon"])).mean()
    area = xr.DataArray(
        np.stack(
            len(coords["lon"])
            * [
                np.pi
                * 6.3781e6**2
                * np.abs(
                    np.sin(np.radians(coords["lat"] + latstep / 2))
                    - np.sin(np.radians(coords["lat"] - latstep / 2))
                )
                * lonstep
                / 180
            ],
            axis=-1,
        ),
        coords={"lat": coords["lat"], "lon": coords["lon"]},
        dims=("lat", "lon"),
    )

    return area


class Regrid_to_LatLon:
    def __init__(self, ds, gridname, regrid_methods=["bilinear", "conservative"]):
        self.gridname = gridname

        self.area = get_area_for_latlongrid(gridname)
        self.old_area = get_area_for_latlongrid(coords=dict(lat=ds.lat, lon=ds.lon))

        self.regridder = {
            method: xe.Regridder(ds, self.area, method, periodic=True)
            for method in regrid_methods
        }

    def __call__(self, ds, intensive=True, regrid_method="bilinear"):

        regridder = self.regridder[regrid_method]

        if intensive:
            ds_regrid = regridder(ds, keep_attrs=True)
        else:
            ds_regrid = regridder(ds / self.old_area, keep_attrs=True) * self.area
            ds_regrid = (
                ds_regrid / ds_regrid.sum(["lat", "lon"]) * ds.sum(["lat", "lon"])
            )

        ds_regrid["cell_area"] = self.area

        return ds_regrid


def vertical_aggregation(
    ds,
    levels,
):
    vertical_ds = ds[[v for v in ds if "level" in ds[v].dims]]
    all_aggregated_ds = []
    for i, level in enumerate(levels):

        if len(level) > 1:

            pressure_thickness = vertical_ds.p_bottom.isel(
                level=level
            ) - vertical_ds.p_top.isel(level=level)
            pressure_weights = pressure_thickness / pressure_thickness.sum("level")

            ds_aggregated = (vertical_ds.isel(level=level) * pressure_weights).sum(
                "level"
            )
            ds_aggregated["p_bottom"] = vertical_ds.p_bottom.isel(level=level[0])
            ds_aggregated["p_top"] = vertical_ds.p_top.isel(level=level[-1])
            ds_aggregated["gph_bottom"] = vertical_ds.gph_bottom.isel(level=level[0])
            ds_aggregated["gph_top"] = vertical_ds.gph_top.isel(level=level[-1])

            ds_aggregated["airmass"] = vertical_ds.airmass.isel(level=level).sum(
                "level"
            )
            ds_aggregated["co2massmix"] = (
                vertical_ds.co2massmix * vertical_ds.airmass
            ).isel(level=level).sum("level") / ds_aggregated.airmass

            ds_aggregated = ds_aggregated.assign_coords(dict(level=[i]))
        else:
            ds_aggregated = vertical_ds.isel(level=level).assign_coords(dict(level=[i]))

        all_aggregated_ds.append(ds_aggregated)

    ds_aggregated = xr.concat(all_aggregated_ds, "level")

    for v in ds:
        if "level" not in ds[v].dims:
            ds_aggregated[v] = ds[v]

    return ds_aggregated


def temporal_resample(ds, startdate, enddate, freq="6h"):

    timesteps = pd.date_range(startdate, enddate, freq=freq)
    timesteps = timesteps[np.where(timesteps >= ds.time.values[0])[0]]
    timesteps = timesteps[np.where(timesteps <= ds.time.values[-1])[0]]

    timedelta = np.diff(timesteps).mean()

    ds_lin_resample = ds.interp(
        time=timesteps,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )

    staggered_fluxes = (
        ds[["co2flux_land", "co2flux_ocean", "co2flux_anthro"]]
        .interp(
            time=timesteps + timedelta / 2,
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )
        .assign_coords(dict(time=timesteps))
    )

    co2_mass_delta = (
        (ds_lin_resample["co2massmix"] / 1e6 * ds_lin_resample["airmass"])
        .sum(["lat", "lon", "level"])
        .diff("time", label="lower")
    )

    massflux = (
        (
            staggered_fluxes["co2flux_land"]
            + staggered_fluxes["co2flux_ocean"]
            + staggered_fluxes["co2flux_anthro"]
        )
        * ds.cell_area
        * timedelta.astype("timedelta64[s]").astype(int)
        / 1e12
    )

    fluxdiff = (
        (co2_mass_delta - massflux.sum(["lat", "lon"]))
        * 1e12
        / (timedelta.astype("timedelta64[s]").astype(int))
        / ds.cell_area
    )

    fluxcorr_anthro = fluxdiff * (
        staggered_fluxes.co2flux_anthro
        / staggered_fluxes.co2flux_anthro.sum(["lat", "lon"])
    )

    ds_lin_resample["co2flux_land"] = staggered_fluxes["co2flux_land"]
    ds_lin_resample["co2flux_ocean"] = staggered_fluxes["co2flux_ocean"]
    ds_lin_resample["co2flux_anthro"] = (
        staggered_fluxes["co2flux_anthro"] + fluxcorr_anthro
    )

    return ds_lin_resample


def optimize_zarr(ds, chunksize_in_mb=10):

    ds_2d = (
        ds[[v for v in ds.data_vars if "level" not in ds[v].dims]]
        .to_array("vari_2d")
        .transpose("time", "vari_2d", "lat", "lon")
        .astype("float32")
    )

    timesteps_in_chunk = max(
        int(chunksize_in_mb / (np.prod(ds_2d.shape[1:]) * 4 / 1024 / 1024)), 1
    )
    ds_2d = ds_2d.chunk(dict(time=timesteps_in_chunk, vari_2d=-1, lat=-1, lon=-1))

    ds_3d = (
        ds[[v for v in ds.data_vars if "level" in ds[v].dims]]
        .to_array("vari_3d")
        .transpose("time", "vari_3d", "level", "lat", "lon")
        .astype("float32")
    )
    timesteps_in_chunk = max(
        int(chunksize_in_mb / (np.prod(ds_3d.shape[1:]) * 4 / 1024 / 1024)), 1
    )
    ds_3d = ds_3d.chunk(
        dict(time=timesteps_in_chunk, vari_3d=-1, lat=-1, lon=-1, level=-1)
    )

    ds_opt = xr.Dataset({"variables_2d": ds_2d, "variables_3d": ds_3d})

    return ds_opt


def compute_stats(ds):

    print("Computing stats")
    print("Delta Min")
    with ProgressBar():
        ds_min = ds.diff("time").min(["time", "lat", "lon"]).compute()
    print("Delta Mean")
    with ProgressBar():
        ds_mean = (
            ds.diff("time").mean(["time", "lat", "lon"], dtype=np.float64).compute()
        )
    print("Delta Max")
    with ProgressBar():
        ds_max = ds.diff("time").max(["time", "lat", "lon"]).compute()
    print("Delta Std")
    with ProgressBar():
        ds_std = ds.diff("time").std(["time", "lat", "lon"], dtype=np.float64).compute()

    ds_delta_stats = xr.concat([ds_min, ds_mean, ds_max, ds_std], "stats")

    ds_delta_stats = xr.merge(
        [
            ds_delta_stats.variables_3d.to_dataset("vari_3d"),
            ds_delta_stats.variables_2d.to_dataset("vari_2d"),
        ]
    )
    ds_delta_stats = ds_delta_stats.rename(
        {k: f"{k}_delta" for k in ds_delta_stats.data_vars}
    ).assign_coords({"stats": ["min", "mean", "max", "std"]})

    print("Global Min")
    with ProgressBar():
        ds_min = ds.min(["time", "level", "lat", "lon"]).compute()
    print("Global Mean")
    with ProgressBar():
        ds_mean = ds.mean(["time", "level", "lat", "lon"], dtype=np.float64).compute()
    print("Global Max")
    with ProgressBar():
        ds_max = ds.max(["time", "level", "lat", "lon"]).compute()
    print("Global Std")
    with ProgressBar():
        ds_std = ds.std(["time", "level", "lat", "lon"], dtype=np.float64).compute()

    ds_stats = xr.concat([ds_min, ds_mean, ds_max, ds_std], "stats")

    ds_stats = xr.merge(
        [
            ds_stats.variables_3d.to_dataset("vari_3d"),
            ds_stats.variables_2d.to_dataset("vari_2d"),
        ]
    ).assign_coords({"stats": ["min", "mean", "max", "std"]})

    for v in ds_stats.data_vars:
        ds_stats[f"{v}_next"] = ds_stats[v]

    ds_stats = ds_delta_stats.merge(ds_stats, compat="override")

    return ds_stats
