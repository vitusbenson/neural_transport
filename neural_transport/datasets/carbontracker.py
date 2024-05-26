import urllib.request
from pathlib import Path

import dask
import numpy as np
import xarray as xr
import xesmf as xe
from dask.diagnostics import ProgressBar
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from neural_transport.tools.conversion import *

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


def latlon_to_zarr(save_dir):
    save_dir = Path(save_dir)

    for year in range(2000, 2017):
        print(f"Processing year {year}")
        for month in [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ]:
            print(f"Processing month {month}")
            with ProgressBar():
                ds_pl = (
                    xr.open_mfdataset(
                        (save_dir / "Carbontracker" / "CT2022_molefrac").glob(
                            f"CT2022.molefrac_glb3x2_{year}-{month}*.nc"
                        )
                    )
                    # .compute()
                    .chunk(
                        dict(time=1, latitude=-1, longitude=-1, level=-1, boundary=-1)
                    )
                )
            with ProgressBar():
                ds_sfc = (
                    xr.open_mfdataset(
                        (save_dir / "Carbontracker" / "CT2022_flux").glob(
                            f"CT2022.flux1x1.{year}{month}*.nc"
                        )
                    )
                    # .compute()
                    .chunk(dict(time=1, latitude=-1, longitude=-1))
                )
            print("Loaded data")
            lat = np.linspace(-89, 89, 90)
            lon = np.linspace(-178.5, 178.5, 144)
            dxyp = xr.DataArray(
                np.stack(
                    len(lon)
                    * [
                        np.pi
                        * 6.375e6**2
                        * (np.sin(np.radians(lat + 1)) - np.sin(np.radians(lat - 1)))
                        * 2
                        / 180
                    ],
                    axis=-1,
                ),
                coords={"latitude": lat, "longitude": lon},
                dims=("latitude", "longitude"),
            )

            regridder = xe.Regridder(ds_pl, dxyp, "conservative_normed", periodic=True)

            ds_pl = regridder(ds_pl, keep_attrs=True)

            regridder = xe.Regridder(ds_sfc, dxyp, "conservative_normed", periodic=True)

            ds_sfc = regridder(ds_sfc, keep_attrs=True)

            V = dxyp * ds_pl.gph.diff("boundary").rename(
                {"boundary": "level"}
            ).assign_coords({"level": ds_pl.level.values})

            ds_pl["airdensity"] = ds_pl.air_mass / V
            ds_pl["volume"] = V

            ds_pl["gp"] = (
                ds_pl.gph.isel(boundary=slice(None, -1))
                .rename({"boundary": "level"})
                .assign_coords({"level": ds_pl.level.values})
                + ds_pl.gph.isel(boundary=slice(1, None))
                .rename({"boundary": "level"})
                .assign_coords({"level": ds_pl.level.values})
            ) / 2
            ds_pl["p"] = (
                ds_pl.pressure.isel(boundary=slice(None, -1))
                .rename({"boundary": "level"})
                .assign_coords({"level": ds_pl.level.values})
                + ds_pl.pressure.isel(boundary=slice(1, None))
                .rename({"boundary": "level"})
                .assign_coords({"level": ds_pl.level.values})
            ) / 2

            ds_pl["co2massmix"] = molemix_to_massmix(ds_pl.co2)
            ds_pl["co2density"] = massmix_to_density(
                ds_pl["co2massmix"], ds_pl.airdensity, ppm=True
            )
            ds_sfc["blh"] = ds_pl.blh
            ds_sfc["orography"] = ds_pl.orography

            ds_pl = ds_pl.rename(
                {
                    "level": "height",
                    "latitude": "lat",
                    "longitude": "lon",
                    "co2": "co2molemix",
                    "temperature": "t",
                    "specific_humidity": "q",
                }
            ).drop_vars(
                [
                    "blh",
                    "gph",
                    "pressure",
                    "orography",
                    "calendar_components",
                    "boundary",
                ]
            )

            ds_pl["height"] = [
                9.6205969e02,
                9.5344867e02,
                9.3724054e02,
                9.1289655e02,
                8.7725018e02,
                8.3036420e02,
                7.7615771e02,
                7.1108643e02,
                6.4732275e02,
                5.7938220e02,
                5.2086908e02,
                4.8320065e02,
                4.4741156e02,
                4.1372739e02,
                3.8207999e02,
                3.4553558e02,
                3.1128784e02,
                2.8607285e02,
                2.5710156e02,
                2.2520422e02,
                2.0088553e02,
                1.7908185e02,
                1.5524625e02,
                1.3402344e02,
                1.1521425e02,
                9.6265671e01,
                7.9630470e01,
                6.7339714e01,
                5.0337818e01,
                2.8204695e01,
                1.2295965e01,
                5.2308245e00,
                2.1191823e00,
                5.3707880e-01,
            ]

            ds_sfc["cell_area"] = dxyp
            ds_sfc["co2flux_land"] = ds_sfc.bio_flux_opt
            ds_sfc["co2flux_ocean"] = ds_sfc.ocn_flux_opt
            ds_sfc["co2flux_subt"] = ds_sfc.fossil_flux_imp + ds_sfc.fire_flux_imp

            ds_sfc = ds_sfc.rename(
                {
                    "latitude": "lat",
                    "longitude": "lon",
                }
            )

            for split, timeslice in zip(
                ["val", "test", "train"],
                [
                    slice("2017-01-01", "2017-12-31"),
                    slice("2018-01-01", "2020-12-31"),
                    slice(None, "2016-12-31"),
                ],
            ):
                if split != "train":
                    continue
                ds_2d = (
                    ds_sfc.sel(time=timeslice)
                    .isel(time=slice(None, None, 2))
                    .to_array("vari_2d")
                    .chunk(dict(vari_2d=-1, time=10, lat=-1, lon=-1))
                    .transpose("time", "vari_2d", "lat", "lon")
                    .astype("float32")
                )

                ds_3d = (
                    ds_pl.sel(time=timeslice)
                    .isel(time=slice(None, None, 2))
                    .to_array("vari_3d")
                    .chunk(dict(vari_3d=-1, time=1, lat=-1, lon=-1, height=-1))
                    .transpose("time", "vari_3d", "height", "lat", "lon")
                    .astype("float32")
                )
                ds = xr.Dataset({"variables_2d": ds_2d, "variables_3d": ds_3d})
                print(f"Writing {split} to zarr")
                out_dir = save_dir / "Carbontracker" / split
                out_dir.mkdir(parents=True, exist_ok=True)
                if year == 2000 and month == "01":
                    with ProgressBar():
                        ds.to_zarr(
                            out_dir / f"carbontracker_latlon2.zarr",
                        )
                else:
                    with ProgressBar():
                        ds.to_zarr(
                            out_dir / f"carbontracker_latlon2.zarr", append_dim="time"
                        )


def fix_zarr(old_dir, save_dir):
    old_dir = Path(old_dir)
    save_dir = Path(save_dir)
    for split in ["val", "test", "train"]:
        ds = xr.open_zarr(
            old_dir / "Carbontracker" / split / "carbontracker_latlon2.zarr"
        )
        ds_sfc = ds.variables_2d.to_dataset("vari_2d")
        ds_pl = ds.variables_3d.to_dataset("vari_3d")

        ds_sfc["co2flux_land"] = ds_sfc.bio_flux_opt
        ds_sfc["co2flux_ocean"] = ds_sfc.ocn_flux_opt
        ds_sfc["co2flux_subt"] = ds_sfc.fossil_flux_imp + ds_sfc.fire_flux_imp

        ds_pl["height"] = [
            9.6205969e02,
            9.5344867e02,
            9.3724054e02,
            9.1289655e02,
            8.7725018e02,
            8.3036420e02,
            7.7615771e02,
            7.1108643e02,
            6.4732275e02,
            5.7938220e02,
            5.2086908e02,
            4.8320065e02,
            4.4741156e02,
            4.1372739e02,
            3.8207999e02,
            3.4553558e02,
            3.1128784e02,
            2.8607285e02,
            2.5710156e02,
            2.2520422e02,
            2.0088553e02,
            1.7908185e02,
            1.5524625e02,
            1.3402344e02,
            1.1521425e02,
            9.6265671e01,
            7.9630470e01,
            6.7339714e01,
            5.0337818e01,
            2.8204695e01,
            1.2295965e01,
            5.2308245e00,
            2.1191823e00,
            5.3707880e-01,
        ]

        ds_2d = (
            ds_sfc.isel(time=slice(None, None, 2))
            .to_array("vari_2d")
            .chunk(dict(vari_2d=-1, time=10, lat=-1, lon=-1))
            .transpose("time", "vari_2d", "lat", "lon")
            .astype("float32")
        )

        ds_3d = (
            ds_pl.isel(time=slice(None, None, 2))
            .to_array("vari_3d")
            .chunk(dict(vari_3d=-1, time=1, lat=-1, lon=-1, height=-1))
            .transpose("time", "vari_3d", "height", "lat", "lon")
            .astype("float32")
        )
        ds = xr.Dataset({"variables_2d": ds_2d, "variables_3d": ds_3d})
        print(split)
        with ProgressBar():
            ds.to_zarr(
                save_dir / "Carbontracker" / split / "carbontracker_latlon2.zarr",
                mode="w",
            )


def stats_dataset(save_dir):
    save_dir = Path(save_dir)
    train_dir = save_dir / "Carbontracker" / "train"
    val_dir = save_dir / "Carbontracker" / "val"
    test_dir = save_dir / "Carbontracker" / "test"

    ds = xr.open_zarr(train_dir / "carbontracker_latlon2.zarr")

    with ProgressBar():
        ds_min = ds.diff("time").min(["time", "lat", "lon"]).compute()
    with ProgressBar():
        ds_mean = (
            ds.diff("time").mean(["time", "lat", "lon"], dtype=np.float64).compute()
        )
    with ProgressBar():
        ds_max = ds.diff("time").max(["time", "lat", "lon"]).compute()
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

    with ProgressBar():
        ds_min = ds.min(["time", "height", "lat", "lon"]).compute()
    with ProgressBar():
        ds_mean = ds.mean(["time", "height", "lat", "lon"], dtype=np.float64).compute()
    with ProgressBar():
        ds_max = ds.max(["time", "height", "lat", "lon"]).compute()
    with ProgressBar():
        ds_std = ds.std(["time", "height", "lat", "lon"], dtype=np.float64).compute()

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

    for out_dir in [train_dir, val_dir, test_dir]:
        ds_stats.to_netcdf(out_dir / "carbontracker_stats.nc")


if __name__ == "__main__":

    # download_data("data")
    fix_zarr("data", "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/")
    stats_dataset("/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/")  # "data")
