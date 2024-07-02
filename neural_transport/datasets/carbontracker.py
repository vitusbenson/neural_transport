import urllib.request
from pathlib import Path

import dask
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from neural_transport.datasets.common import (
    Regrid_to_LatLon,
    compute_stats,
    optimize_zarr,
    temporal_resample,
    vertical_aggregation,
)
from neural_transport.datasets.grids import VERTICAL_LAYERS_PROTOTYPE_COORDS
from neural_transport.datasets.solar_radiation import (
    get_toa_incident_solar_radiation_for_xarray,
)
from neural_transport.tools.conversion import *
from neural_transport.tools.obspack_helper import extract_obspack_locs_from_xarray

dask.config.set(scheduler="threads")


def download_data(save_dir):
    save_dir = Path(save_dir)

    BASEPATH = "https://gml.noaa.gov/aftp/products/carbontracker/co2/"
    MOLEFRACTION_PATH = f"{BASEPATH}molefractions/co2_total/"
    FLUXES_PATH = f"{BASEPATH}fluxes/three-hourly/"

    for date in tqdm(np.arange("2000-01-01", "2021-03-01", dtype="datetime64")):

        molefraction_filename = (
            f"CT2022.molefrac_glb3x2_{date}.nc"  # f"CT2022.molefrac_nam1x1_{date}.nc"
        )

        url = f"{MOLEFRACTION_PATH}{molefraction_filename}"

        outpath = save_dir / "Carbontracker" / "CT2022_molefrac" / molefraction_filename

        outpath.parent.mkdir(parents=True, exist_ok=True)

        if not outpath.is_file():
            try:
                urllib.request.urlretrieve(url, str(outpath))
            except:
                print(f"Error downloading {url}")

        fluxes_filename = f"CT2022.flux1x1.{date.item().strftime('%Y%m%d')}.nc"

        url = f"{FLUXES_PATH}{fluxes_filename}"

        outpath = save_dir / "Carbontracker" / "CT2022_flux" / fluxes_filename

        outpath.parent.mkdir(parents=True, exist_ok=True)

        if not outpath.is_file():
            try:
                urllib.request.urlretrieve(url, str(outpath))
            except:
                print(f"Error downloading {url}")

    print("Done!")


def load_and_regrid(
    molefraction_path, fluxes_path, regridder_molefraction, regridder_fluxes
):

    molefraction = (
        xr.open_dataset(molefraction_path)
        .rename(dict(latitude="lat", longitude="lon"))
        .astype("float64")
    )
    fluxes = (
        xr.open_dataset(fluxes_path)
        .rename(dict(latitude="lat", longitude="lon"))
        .astype("float64")
    )

    intensive_3d_latlon = regridder_molefraction(
        molefraction[
            [
                "pressure",
                "gph",
                "temperature",
                "specific_humidity",
                "u",
                "v",
                "blh",
                "orography",
            ]
        ],
        intensive=True,
        regrid_method="bilinear",
    )

    molefraction["airmass"] = molefraction.air_mass * 1e-12  # Pg Air

    molefraction["co2mass"] = (
        molemix_to_massmix(molefraction.co2) * 1e-6
    ) * molefraction.airmass

    extensive_3d_latlon = regridder_molefraction(
        molefraction[["co2mass", "airmass"]],
        intensive=False,
        regrid_method="conservative",
    )

    fluxes_latlon = regridder_fluxes(
        fluxes, intensive=True, regrid_method="conservative"
    )

    ds = xr.Dataset(
        dict(
            co2massmix=(
                (extensive_3d_latlon.co2mass / extensive_3d_latlon.airmass) * 1e6
            ).assign_attrs(units="1e-6kg / kg"),
            airmass=extensive_3d_latlon.airmass.assign_attrs(units="Pg"),  # Pg Air
            p_bottom=intensive_3d_latlon.pressure.isel(boundary=slice(None, -1))
            .rename({"boundary": "level"})
            .assign_coords({"level": intensive_3d_latlon.level})
            .assign_attrs(units="hPa")
            / 100,
            p_top=intensive_3d_latlon.pressure.isel(boundary=slice(1, None))
            .rename({"boundary": "level"})
            .assign_coords({"level": intensive_3d_latlon.level})
            .assign_attrs(units="hPa")
            / 100,
            gph_bottom=intensive_3d_latlon.gph.isel(boundary=slice(None, -1))
            .rename({"boundary": "level"})
            .assign_coords({"level": intensive_3d_latlon.level})
            .assign_attrs(units="km")
            / 1000,
            gph_top=intensive_3d_latlon.gph.isel(boundary=slice(1, None))
            .rename({"boundary": "level"})
            .assign_coords({"level": intensive_3d_latlon.level})
            .assign_attrs(units="km")
            / 1000,
            t=intensive_3d_latlon.temperature,
            q=intensive_3d_latlon.specific_humidity,
            u=intensive_3d_latlon.u,
            v=intensive_3d_latlon.v,
            cell_area=intensive_3d_latlon.cell_area.assign_attrs(units="m"),
            blh=intensive_3d_latlon.blh,
            orography=intensive_3d_latlon.orography,
            co2flux_land=(fluxes_latlon.bio_flux_opt * M_CO2).assign_attrs(
                units="kg/m^2/s"
            ),
            co2flux_ocean=(fluxes_latlon.ocn_flux_opt * M_CO2).assign_attrs(
                units="kg/m^2/s"
            ),
            co2flux_anthro=(
                (fluxes_latlon.fossil_flux_imp + fluxes_latlon.fire_flux_imp) * M_CO2
            ).assign_attrs(units="kg/m^2/s"),
            tisr=get_toa_incident_solar_radiation_for_xarray(extensive_3d_latlon),
        )
    ).astype("float32")

    return ds


CARBONTRACKER_LEVEL_AGG = dict(
    l34=[[i] for i in range(34)],
    l10=[
        [0],
        [1],
        [2],
        [3],
        [4, 5],
        [6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18],
        [19, 20, 21, 22, 23, 24, 25, 26],
        [27, 28, 29, 30, 31, 32, 33],
    ],
    l20=[[i] for i in range(6)] + [[i, i + 1] for i in range(6, 34, 2)],
)


def regrid_carbontracker(save_dir, gridname="latlon2x3", vertical_levels="l34"):
    save_dir = Path(save_dir)
    for date in tqdm(np.arange("2000-01-01", "2021-03-01", dtype="datetime64")):

        molefraction_path = (
            save_dir
            / "Carbontracker"
            / "CT2022_molefrac"
            / f"CT2022.molefrac_glb3x2_{date}.nc"
        )

        fluxes_path = (
            save_dir
            / "Carbontracker"
            / "CT2022_flux"
            / f"CT2022.flux1x1.{date.item().strftime('%Y%m%d')}.nc"
        )

        out_path = (
            save_dir
            / "Carbontracker"
            / "CT2022_regrid"
            / f"CT2022_regrid_{gridname}_{vertical_levels}.zarr"
        )

        if not out_path.is_dir():

            regridder_molefraction = Regrid_to_LatLon(
                xr.open_dataset(molefraction_path).rename(
                    dict(latitude="lat", longitude="lon")
                ),
                gridname,
            )
            regridder_fluxes = Regrid_to_LatLon(
                xr.open_dataset(fluxes_path).rename(
                    dict(latitude="lat", longitude="lon")
                ),
                gridname,
            )

        ds = load_and_regrid(
            molefraction_path, fluxes_path, regridder_molefraction, regridder_fluxes
        )

        ds = vertical_aggregation(ds, levels=CARBONTRACKER_LEVEL_AGG[vertical_levels])
        ds["level"] = VERTICAL_LAYERS_PROTOTYPE_COORDS[vertical_levels]["level"]

        ds = ds.chunk(dict(time=-1, lat=-1, lon=-1, level=-1))

        if not out_path.is_dir():
            ds.to_zarr(out_path)
        else:
            ds.to_zarr(out_path, mode="a", append_dim="time")


def resample_carbontracker(
    save_dir, gridname="latlon2x3", vertical_levels="l34", freq="6h"
):
    save_dir = Path(save_dir)

    ds = xr.open_zarr(
        save_dir
        / "Carbontracker"
        / "CT2022_regrid"
        / f"CT2022_regrid_{gridname}_{vertical_levels}.zarr"
    )

    for date in tqdm(np.arange("2000-01-01", "2021-03-01", dtype="datetime64")):

        ds_resample = temporal_resample(
            ds.sel(
                time=slice(date - np.timedelta64(1, "D"), date + np.timedelta64(2, "D"))
            ),
            str(date),
            str(date + np.timedelta64(1, "D")),
            freq=freq,
        ).sel(time=str(date))

        out_path = (
            save_dir
            / "Carbontracker"
            / "CT2022_regrid"
            / f"CT2022_regrid_{gridname}_{vertical_levels}_{freq}.zarr"
        )

        if not out_path.is_dir():
            ds_resample.to_zarr(out_path)
        else:
            ds_resample.to_zarr(out_path, mode="a", append_dim="time")


def write_carbontracker(
    save_dir, gridname="latlon2x3", vertical_levels="l34", freq="6h"
):
    save_dir = Path(save_dir)

    ds = xr.open_zarr(
        save_dir
        / "Carbontracker"
        / "CT2022_regrid"
        / f"CT2022_regrid_{gridname}_{vertical_levels}_{freq}.zarr"
    )

    for split, timeslice in zip(
        ["val", "test", "train"],
        [
            slice("2017-01-01", "2017-12-31"),
            slice("2018-01-01", "2020-12-31"),
            slice(None, "2016-12-31"),
        ],
    ):
        print(f"Writing {split} to zarr")
        out_dir = save_dir / "Carbontracker" / split
        out_dir.mkdir(parents=True, exist_ok=True)

        ds_opt = optimize_zarr(ds.sel(time=timeslice))

        with ProgressBar():
            ds_opt.to_zarr(
                out_dir / f"carbontracker_{gridname}_{vertical_levels}_{freq}.zarr",
                mode="w",
            )


def stats_carbontracker(
    save_dir, gridname="latlon2x3", vertical_levels="l34", freq="6h"
):
    save_dir = Path(save_dir)
    train_dir = save_dir / "Carbontracker" / "train"
    val_dir = save_dir / "Carbontracker" / "val"
    test_dir = save_dir / "Carbontracker" / "test"

    ds = xr.open_zarr(
        train_dir / f"carbontracker_{gridname}_{vertical_levels}_{freq}.zarr"
    )

    ds_stats = compute_stats(ds)

    for out_dir in [train_dir, val_dir, test_dir]:
        ds_stats.to_zarr(
            out_dir / f"carbontracker_{gridname}_{vertical_levels}_{freq}_stats.zarr",
            mode="w",
        )


def obspack_carbontracker(
    save_dir, gridname="latlon2x3", vertical_levels="l34", freq="6h"
):
    save_dir = Path(save_dir)
    test_dir = save_dir / "Carbontracker" / "test"

    ds = xr.open_zarr(
        test_dir / f"carbontracker_{gridname}_{vertical_levels}_{freq}.zarr"
    )
    ds = xr.merge(
        [ds.variables_2d.to_dataset("vari_2d"), ds.variables_3d.to_dataset("vari_3d")]
    )[["co2massmix", "gph_bottom", "gph_top"]].compute()

    obs = (
        xr.open_zarr(save_dir / "Obspack" / "obspack.zarr")
        .sel(time=ds.time, method="nearest")
        .compute()
    )

    obs["lat"] = obs.lat.interpolate_na(
        dim="time", method="nearest", fill_value="extrapolate"
    )
    obs["lon"] = obs.lon.interpolate_na(
        dim="time", method="nearest", fill_value="extrapolate"
    )
    obs["height"] = obs.height.interpolate_na(
        dim="time", method="nearest", fill_value="extrapolate"
    )
    with ProgressBar():
        obs_carbontracker = (
            xr.concat(
                [
                    extract_obspack_locs_from_xarray(ds.isel(time=i), obs)
                    for i in range(len(ds.time))
                ],
                dim="time",
            )
            .compute()
            .reset_encoding()
            .chunk({"time": -1, "cell": 100})
        )

    with ProgressBar():
        obs_carbontracker.to_zarr(
            test_dir / f"obs_carbontracker_{gridname}_{vertical_levels}_{freq}.zarr",
            mode="w",
        )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--gridname", type=str, default="latlon2x3")
    parser.add_argument("--vertical_levels", type=str, default="l34")
    parser.add_argument("--freq", type=str, default="3h")
    args = parser.parse_args()

    download_data(args.save_dir)

    regrid_carbontracker(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels,
    )

    resample_carbontracker(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels,
        freq=args.freq,
    )

    write_carbontracker(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels,
        freq=args.freq,
    )

    stats_carbontracker(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels,
        freq=args.freq,
    )

    obspack_carbontracker(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels,
        freq=args.freq,
    )
