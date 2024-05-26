import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from ecmwfapi import ECMWFService
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from neural_transport.tools.conversion import *

CAMS_MODEL_RUNS = [
    {
        "expver": "gf39",
        "date": "2015-01-01/2016-03-07",
        "date_sel": "2015-01-01/2016-03-07",
        "resolution": "tl1279 (16km)",
        "class": "rd",
        "stream": "lwda",
        "IFS cycle": "CY41R2",
        "Notes": "Cray hpc,free running (cyclic FC)",
    },
    {
        "expver": "gnoo",
        "date": "2016-03-08/2017-01-31",
        "date_sel": "2016-03-08/2016-12-31",
        "resolution": "tco1279 (9km)",
        "class": "rd",
        "stream": "lwda",
        "IFS cycle": "CY43R1",
        "Notes": "Cray hpc,nudged to CAMS CO2 analysis",
    },
    {
        "expver": "gqpe",
        "date": "2017-01-01/2018-12-31",
        "date_sel": "2017-01-01/2018-05-31",
        "resolution": "tco1279 (9km)",
        "class": "rd",
        "stream": "lwda",
        "IFS cycle": "CY43R1",
        "Notes": "Cray hpc,nudged to CAMS CO2 analysis",
    },
    {
        "expver": "gznv",
        "date": "2018-06-01/2019-12-31",
        "date_sel": "2018-06-01/2019-08-31",
        "resolution": "tco1279 (9km)",
        "class": "rd",
        "stream": "lwda",
        "IFS cycle": "CY45R1",
        "Notes": "Cray hpc,nudged to CAMS CO2 analysis",
    },
    {
        "expver": "h9sp",
        "date": "2019-09-01/2021-02-12",
        "date_sel": "2019-09-01/2019-12-31",
        "resolution": "tco1279 (9km)",
        "class": "rd",
        "stream": "lwda",
        "IFS cycle": "CY46R1",
        "Notes": "Cray hpc,nudged to CAMS CO2 analysis",
    },
    {
        "expver": "he9h",
        "date": "2020-01-01/2021-12-01",
        "date_sel": "2020-01-01/2021-03-31",
        "resolution": "tco1279 (9km)",
        "class": "rd",
        "stream": "lwda",
        "IFS cycle": "CY47R1",
        "Notes": "Cray hpc, nudged to CAMS CO2 analysis",
    },
    {
        "expver": "hlld",
        "date": "2021-04-01/2022-10-30",
        "date_sel": "2021-04-01/2022-09-30",
        "resolution": "tco1279 (9km)",
        "class": "rd",
        "stream": "lwda",
        "IFS cycle": "CY7R3",
        "Notes": "Atos hpc, nudged to CAMS CO2 analysis",
    },
    {
        "expver": "hueu",
        "date": "2022-09-19/2024-02-29",
        "date_sel": "2022-10-01/2024-02-29",
        "resolution": "tco1279 (9km)",
        "class": "rd",
        "stream": "oper",
        "IFS cycle": "CY47R3",
        "Notes": "Atos hpc, nudged to CAMS CO2 analysis",
    },
    # {
    #     "expver": "0001",
    #     "date": "2024-02-26/present",
    #     "date_sel": "2024-03-01/2024-03-20",
    #     "resolution": "tco1279 (9km)",
    #     "class": "gg",
    #     "stream": "oper",
    #     "IFS cycle": "CY48R1",
    #     "Notes": "osuite, Atos hpc, nudged to CAMS CO2 analysis",
    # },
]

VARS_3D = ["co2", "r", "t", "u", "v", "w", "z"]
VARS_2D = ["co2apf", "co2fire", "co2of", "fco2nee", "msl", "t2m", "tp", "u10", "v10"]

months = [
    ["01", "01-01", "01-31"],
    ["02", "02-01", "02-28"],
    ["03", "03-01", "03-31"],
    ["04", "04-01", "04-30"],
    ["05", "05-01", "05-31"],
    ["06", "06-01", "06-30"],
    ["07", "07-01", "07-31"],
    ["08", "08-01", "08-31"],
    ["09", "09-01", "09-30"],
    ["10", "10-01", "10-31"],
    ["11", "11-01", "11-30"],
    ["12", "12-01", "12-31"],
]


def download_data(save_dir):

    save_dir = Path(save_dir)

    server = ECMWFService("mars")

    for model_run in CAMS_MODEL_RUNS[::-1]:
        start_date, end_date = model_run["date_sel"].split("/")
        expver = model_run["expver"]
        resolution = model_run["resolution"]
        stream = model_run["stream"]
        cycle = model_run["IFS cycle"]

        print(f"Downloading {expver} {start_date} to {end_date}")

        pl_dir = save_dir / "pl" / expver
        sfc_dir = save_dir / "sfc" / expver
        pl_dir.mkdir(parents=True, exist_ok=True)
        sfc_dir.mkdir(parents=True, exist_ok=True)

        for year in (
            pbar1 := tqdm(
                range(int(start_date[:4]), int(end_date[:4])),
                position=1,
                desc="Year",
                leave=False,
            )
        ):
            pbar1.set_postfix({"year": year})

            for month, month_start, month_end in (
                pbar2 := tqdm(months, position=2, desc="Month", leave=False)
            ):
                if month == "02" and year in [2004, 2008, 2012, 2016, 2020, 2024]:
                    month_end = "02-29"

                pbar2.set_postfix({"month": month})

                if year == int(start_date[:4]):
                    if month < start_date[5:7]:
                        continue
                    elif month == start_date[5:7]:
                        month_start = start_date[5:10]
                if year == int(end_date[:4]):
                    if month > end_date[5:7]:
                        continue
                    elif month == end_date[5:7]:
                        month_end = end_date[5:10]

                start_day = int(month_start[3:5])
                end_day = int(month_end[3:5])

                for day in range(start_day, end_day + 1):

                    curr_date = f"{year}-{month}-{day:02}"

                    pl_file_path = pl_dir / f"{curr_date}.nc"
                    sfc_file_path = sfc_dir / f"{curr_date}.nc"

                    if not pl_file_path.is_file():
                        try:
                            server.execute(
                                {
                                    "class": "rd",
                                    "date": curr_date,  # f"{year}-{month_start}/to/{year}-{month_end}",
                                    "expver": expver,
                                    "levtype": "pl",
                                    "levelist": "all",
                                    "param": "co2/ch4/co/z/t/q/w/r/u/v",  # "61.210/62.210/123.210/129.128/130.128/131/132/133.128/135.128/157.128",
                                    "step": "all",  # "0/3/6/9/12/15/18/21",
                                    "grid": "1.0/1.0",
                                    "stream": stream,
                                    "time": "00:00:00",
                                    "type": "fc",
                                    "format": "netcdf",
                                },
                                str(pl_file_path),
                            )
                        except KeyboardInterrupt:
                            return
                        except:
                            print(f"{expver} {year} {month} pl not working")

                    if not sfc_file_path.is_file():
                        try:
                            server.execute(
                                {
                                    "class": "rd",
                                    "date": curr_date,  # f"{year}-{month_start}/to/{year}-{month_end}",
                                    "expver": expver,
                                    "levtype": "sfc",
                                    "param": "sst/tcco2/tcch4/tcco/co2of/co2nbf/co2apf/ch4f/fco2nee/z/sshf/slhf/msl/blh/10u/10v/2t/lsm/ssr/str/e/tp/skt/pev/co2fire/ch4fire",  # "34.128/64.210/65.210/67.210/68.210/69.210/70.210/83.228/127.210/129.128/146.128/147.128/151.128/159.128/165.128/166.128/167.128/172.128/176.128/177.128/182.128/228.128/235.128/251.228/210061/210062/210080/210082/210123/219005",  # "64.210/65.210/67.210/69.210/70.210/80.210/82.210/83.228/129.128/146.128/147.128/151.128/165.128/166.128/167.128/228.128",  # "34.128/64.210/65.210/67.210/68.210/69.210/70.210/80.210/82.210/83.228/127.210/129.128/146.128/147.128/151.128/159.128/165.128/166.128/167.128/172.128/176.128/182.128/235.128/251.228",
                                    "step": "all",  # "0/3/6/9/12/15/18/21",
                                    "grid": "1.0/1.0",
                                    "stream": stream,
                                    "time": "00:00:00",
                                    "type": "fc",
                                    "format": "netcdf",
                                },
                                str(sfc_file_path),
                            )
                        except KeyboardInterrupt:
                            return
                        except:
                            print(f"{expver} {year} {month} sfc not working")

    print("Done!")


def latlon_to_zarr(save_dir, expver="hueu"):
    save_dir = Path(save_dir)

    ds_pl = xr.open_mfdataset(
        (save_dir / "CAMS_CO2_forecast" / "pl" / expver).glob("*.nc"),
        preprocess=lambda dstemp: dstemp.rename({"time": "step"})
        .assign_coords({"time": dstemp.time[:1]})
        .assign_coords({"step": (dstemp.time - dstemp.time[0]).values})
        .isel(
            step=[
                0,
                2,
                4,
                6,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
            ]
        ),
    )

    ds_sfc = xr.open_mfdataset(
        (save_dir / "CAMS_CO2_forecast" / "sfc" / expver).glob("*.nc"),
        preprocess=lambda dstemp: dstemp.rename({"time": "step"})
        .assign_coords({"time": dstemp.time[:1]})
        .assign_coords({"step": (dstemp.time - dstemp.time[0]).values})
        .isel(
            step=[
                0,
                2,
                4,
                6,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
            ]
        ),
    ).ffill("step")

    gph = ds_pl.z
    T = ds_pl.t
    RH = ds_pl.r
    dxyp = xr.DataArray(
        np.stack(
            len(ds_pl.longitude)
            * [
                np.pi
                * 6.375e6**2
                * (
                    np.sin(
                        np.radians(
                            np.concatenate(
                                [[-90], np.linspace(-89.5, 89.5, 180), [90]]
                            )[1:]
                        )
                    )
                    - np.sin(
                        np.radians(
                            np.concatenate(
                                [[-90], np.linspace(-89.5, 89.5, 180), [90]]
                            )[:-1]
                        )
                    )
                )
                * 1
                / 180
            ],
            axis=-1,
        ),
        coords={"latitude": ds_pl.latitude, "longitude": ds_pl.longitude},
        dims=("latitude", "longitude"),
    )
    T_celsius = T - 273.15
    Psat = 0.61121 * np.exp(
        (18.678 - T_celsius / 234.5) * (T_celsius / (257.14 + T_celsius))
    )

    Pv = RH / 100 * Psat
    Pd = gph.level - 10 * Pv
    rho = 100 * Pd / (287.050676 * T)
    midpoints = (
        gph.isel(level=slice(-1)).assign_coords(
            {"level": gph.level.isel(level=slice(1, None))}
        )
        + gph.isel(level=slice(1, None))
    ) / 2
    V = dxyp * xr.concat(
        [
            120000
            - midpoints.isel(level=0).assign_coords({"level": gph.level.isel(level=0)}),
            -midpoints.diff("level", label="lower"),
            -midpoints.isel(level=-1) + ds_sfc.z / 9.80665,
        ],
        dim="level",
    )

    ds_pl["airdensity"] = rho
    ds_pl["volume"] = V

    ds_pl["co2density"] = ds_pl.co2 * ds_pl.airdensity
    ds_pl["ch4density"] = ds_pl.ch4 * ds_pl.airdensity
    ds_pl["codensity"] = ds_pl.co * ds_pl.airdensity

    ds_pl = ds_pl.rename(
        {
            "level": "height",
            "latitude": "lat",
            "longitude": "lon",
            "co2": "co2massmix",
            "ch4": "ch4massmix",
            "co": "comassmix",
        }
    )

    ds_pl["co2massmix"] = 1e6 * ds_pl["co2massmix"]
    ds_pl["ch4massmix"] = 1e6 * ds_pl["ch4massmix"]
    ds_pl["comassmix"] = 1e6 * ds_pl["comassmix"]

    ds_sfc["cell_area"] = dxyp
    ds_sfc = ds_sfc.rename({"latitude": "lat", "longitude": "lon", "z": "z_surf"})

    for split, timeslice in zip(
        ["val", "test", "train"],
        [slice(-100, -50), slice(-50, None), slice(None, -100)],
    ):
        ds_2d = (
            ds_sfc.isel(time=timeslice)
            .to_array("vari_2d")
            .chunk(dict(vari_2d=-1, time=1, step=1, lat=-1, lon=-1))
            .transpose("time", "step", "vari_2d", "lat", "lon")
            .astype("float32")
        )

        ds_3d = (
            ds_pl.isel(time=timeslice)
            .to_array("vari_3d")
            .chunk(dict(vari_3d=-1, time=1, step=1, lat=-1, lon=-1, height=-1))
            .transpose("time", "step", "vari_3d", "height", "lat", "lon")
            .astype("float32")
        )
        ds = xr.Dataset({"variables_2d": ds_2d, "variables_3d": ds_3d})
        print(f"Writing {split} to zarr")
        out_dir = save_dir / "CAMS_CO2_forecast" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        with ProgressBar():
            ds.to_zarr(
                out_dir / f"camsfc_latlon1.zarr",
                mode="w",
            )


def compute_delta_stats(ds, k):
    ds_delta = ds[k].compute().diff("time")
    ds_min = ds_delta.min(["time", "step", "lat", "lon"])  # .to_array("var")
    ds_mean = ds_delta.mean(["time", "step", "lat", "lon"])  # .to_array("var")
    ds_max = ds_delta.max(["time", "step", "lat", "lon"])  # .to_array("var")
    ds_std = ds_delta.std(["time", "step", "lat", "lon"])  # .to_array("var")
    return [f"{k}_delta", xr.concat([ds_min, ds_mean, ds_max, ds_std], "stats")]


def stats_dataset(save_dir):
    save_dir = Path(save_dir)
    train_dir = save_dir / "CAMS_CO2_forecast" / "train"
    val_dir = save_dir / "CAMS_CO2_forecast" / "val"
    test_dir = save_dir / "CAMS_CO2_forecast" / "test"

    ds = xr.open_zarr(train_dir / "camsfc_latlon1.zarr")

    ds = xr.merge(
        [
            ds.variables_3d.to_dataset("vari_3d"),
            ds.variables_2d.to_dataset("vari_2d"),
        ]
    )

    ds["co2density"] = massmix_to_density(
        ds.co2massmix, ds.airdensity, ppm=False, eps=1e-12
    )
    ds["co2molemix"] = massmix_to_molemix(ds.co2massmix)
    ds["ch4density"] = massmix_to_density(
        ds.ch4massmix, ds.airdensity, ppm=False, eps=1e-12
    )
    ds["ch4molemix"] = massmix_to_molemix(ds.ch4massmix, M=M_CH4)
    ds["codensity"] = massmix_to_density(
        ds.comassmix, ds.airdensity, ppm=False, eps=1e-12
    )
    ds["comolemix"] = massmix_to_molemix(ds.comassmix, M=M_CO)

    all_stats = process_map(
        compute_delta_stats,
        [ds] * (len(ds.data_vars) - 1),
        [k for k in ds.data_vars if k != "cell_area"],
        max_workers=8,
    )

    ds_delta_stats = xr.Dataset({k: v for k, v in all_stats})

    ds_min = ds.min().to_array("var")
    ds_mean = ds.mean().to_array("var")
    ds_max = ds.max().to_array("var")
    ds_std = ds.std().to_array("var")
    with ProgressBar():
        ds_stats = (
            xr.concat([ds_min, ds_mean, ds_max, ds_std], "stats")
            .assign_coords({"stats": ["min", "mean", "max", "std"]})
            .to_dataset("var")
            .compute()
        )

    for v in ds_stats.data_vars:
        ds_stats[f"{v}_next"] = ds_stats[v]

    ds_stats = ds_delta_stats.merge(ds_stats, compat="override")

    for out_dir in [train_dir, val_dir, test_dir]:
        ds_stats.to_netcdf(out_dir / "camsfc_stats.nc")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CAMS Forecast.")
    parser.add_argument("save_dir", type=str)

    args = parser.parse_args()

    #download_data(args.save_dir)
    latlon_to_zarr(args.save_dir)
    stats_dataset(args.save_dir)
