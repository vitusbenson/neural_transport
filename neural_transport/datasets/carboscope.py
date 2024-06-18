import argparse
import gzip
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from cdo import Cdo
from dask.diagnostics import ProgressBar
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from neural_transport.datasets.grids import (
    CARBOSCOPE_79_LOCATIONS,
    CARBOSCOPE_79_LONLAT,
    OBSPACK_287_LONLAT,
)
from neural_transport.neural_transport.datasets.solar_radiation import (
    get_toa_incident_solar_radiation_for_xarray,
)
from neural_transport.models.gnn.mesh import ICONGrid
from neural_transport.tools.conversion import *


def download_data(save_dir):
    save_dir = Path(save_dir)

    BASEPATH = (
        "https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis/pressure/"
    )

    print("Downloading NCEP Meteo")
    for var in tqdm(
        ["air", "omega", "rhum", "shum", "uwnd", "vwnd"],
        position=0,
        leave=False,
        desc="Variable",
    ):
        for year in tqdm(range(1957, 1976), position=0, leave=False, desc="Year"):
            fp = f"{BASEPATH}{var}.{year}.nc"

            outpath = save_dir / "NCEP_NCAR_Reanalysis" / "latlon" / (fp.split("/")[-1])

            urllib.request.urlretrieve(fp, str(outpath))

    urllib.request.urlretrieve(
        "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface/land.nc",
        str(save_dir / "NCEP_NCAR_Reanalysis" / "land.nc"),
    )

    print("Done!")
    print("Downloading Carboscope Train set")
    BASEPATH = "http://www.bgc-jena.mpg.de/CarboScope/s/INVERSION/OUTPUT/"
    for year in tqdm(range(1957, 2017), position=0, leave=False, desc="Year"):
        fp = f"{BASEPATH}sEXTocNEET_v4.3_mix_{year}.nc.gz"

        outpath = save_dir / "Carboscope" / "mixratio" / (fp.split("/")[-1])

        outpath.parent.mkdir(parents=True, exist_ok=True)

        urllib.request.urlretrieve(fp, str(outpath))

        with gzip.open(str(outpath), "rb") as f_in:
            with open(str(outpath)[:-3], "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(str(outpath))

    fp = f"{BASEPATH}sEXTocNEET_v4.3_daily.nc.gz"

    outpath = save_dir / "Carboscope" / "surfflux" / (fp.split("/")[-1])

    outpath.parent.mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(fp, str(outpath))

    with gzip.open(str(outpath), "rb") as f_in:
        with open(str(outpath)[:-3], "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(str(outpath))
    print("Done!")

    print("Downloading Carboscope Test set")
    for year in tqdm(range(2018, 2022), position=0, leave=False, desc="Year"):
        fp = f"{BASEPATH}s93oc_v2022_mix_{year}.nc.gz"

        outpath = save_dir / "Carboscope" / "mixratio" / (fp.split("/")[-1])

        outpath.parent.mkdir(parents=True, exist_ok=True)

        urllib.request.urlretrieve(fp, str(outpath))

        with gzip.open(str(outpath), "rb") as f_in:
            with open(str(outpath)[:-3], "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(str(outpath))

    fp = f"{BASEPATH}s93oc_v2022_daily.nc.gz"

    outpath = save_dir / "Carboscope" / "surfflux" / (fp.split("/")[-1])

    outpath.parent.mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(fp, str(outpath))

    with gzip.open(str(outpath), "rb") as f_in:
        with open(str(outpath)[:-3], "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(str(outpath))

    print("Done!")


def download_icon_grid(save_dir):
    save_dir = Path(save_dir)

    BASEPATH = "http://icon-downloads.mpimet.mpg.de/grids/public/mpim/"

    all_num_res = [
        ("0030", "R02B03"),
        ("0013", "R02B04"),
        ("0019", "R02B05"),
        ("0021", "R02B06"),
        ("0023", "R02B07"),
    ]

    # "0005/icon_grid_0005_R02B04_G.nc"

    for grid_num, grid_res in tqdm(all_num_res, position=0, leave=False, desc="Grid"):
        URL = BASEPATH + f"{grid_num}/icon_grid_{grid_num}_{grid_res}_G.nc"

        outpath = save_dir / "icon_grid" / f"{grid_res}_G.nc"

        outpath.parent.mkdir(parents=True, exist_ok=True)

        urllib.request.urlretrieve(URL, str(outpath))

    print("Done!")


# cdo remapcon,data/icon_grid/R02B04_G.nc data/Carboscope/surfflux/s93oc_co2flux.nc  data/Carboscope/icon_surfflux/s93oc_co2flux.nc


def aggregate_fluxes(save_dir):
    save_dir = Path(save_dir)

    ds = xr.open_dataset(save_dir / "Carboscope/surfflux/s93oc_v2022_daily.nc")
    fluxes = xr.Dataset(
        {
            v: (
                ds[v] / ds.dxyp / (365 * 24 * 60 * 60) * 1e12 * 44.009e-3 / 12.011e-3
            ).assign_attrs(
                {
                    "long_name": {
                        "co2flux_land": "Flux of Carbon Dioxide Net Ecosystem Exchange",
                        "co2flux_ocean": "Ocean flux of Carbon Dioxide",
                        "co2flux_subt": "Anthropogenic emissions of Carbon Dioxide",
                    }[v],
                    "units": "kg m**-2 s**-1",
                }
            )
            for v in ["co2flux_land", "co2flux_ocean", "co2flux_subt"]
        }
    )

    fluxes.to_netcdf(save_dir / "Carboscope/surfflux/s93oc_co2flux.nc")

    ds = xr.open_dataset(save_dir / "Carboscope/surfflux/sEXTocNEET_v4.3_daily.nc")

    fluxes = xr.Dataset(
        {
            v: (
                ds[v] / ds.dxyp / (365 * 24 * 60 * 60) * 1e12 * 44.009e-3 / 12.011e-3
            ).assign_attrs(
                {
                    "long_name": {
                        "co2flux_land": "Flux of Carbon Dioxide Net Ecosystem Exchange",
                        "co2flux_ocean": "Ocean flux of Carbon Dioxide",
                        "co2flux_subt": "Anthropogenic emissions of Carbon Dioxide",
                    }[v],
                    "units": "kg m**-2 s**-1",
                }
            )
            for v in ["co2flux_land", "co2flux_ocean", "co2flux_subt"]
        }
    )

    for v in ["co2flux_land", "co2flux_ocean", "co2flux_subt"]:
        fluxarr = fluxes[v].values
        fluxarr[:, 0, :] = fluxarr[:, 0, :1]
        fluxarr[:, -1, :] = fluxarr[:, -1, :1]

    fluxes.to_netcdf(save_dir / "Carboscope/surfflux/sEXTocNEET_co2flux.nc")

    print("Done!")


def remap_one(gridpath, inpath, outpath):
    cdo = Cdo()
    cdo.remapcon(str(gridpath), input=str(inpath), output=str(outpath))


def remap_to_icosa(
    save_dir, level="L5", min_level=None, resolved_locations=None, gridname=None
):
    save_dir = Path(save_dir)

    cdo = Cdo()

    if min_level is not None:
        L5Grid = ICONGrid.create(
            min_level, int(level[-1]), resolved_locations=resolved_locations
        )
        if gridname is None:
            gridname = f"icosa{level}-L{min_level}a{len(resolved_locations)}"
    else:
        L5Grid = ICONGrid.create(0, int(level[-1]), resolved_locations=None)
        gridname = f"icosa{level}"

    gridpath = save_dir / "icosa_grid" / f"{gridname}.nc"
    gridpath.parent.mkdir(parents=True, exist_ok=True)

    L5Grid.to_netcdf(gridpath)

    inpaths = []
    outpaths = []
    gridpaths = []
    print("Remapping Carboscope Train set to ICON")
    for year in tqdm(range(1957, 2017), position=0, leave=True, desc="Year"):
        inpath = save_dir / "Carboscope" / "mixratio" / f"sEXTocNEET_v4.3_mix_{year}.nc"
        outpath = (
            save_dir
            / "Carboscope"
            / f"{gridname}_mixratio"
            / f"sEXTocNEET_v4.3_mix_{year}.nc"
        )
        outpath.parent.mkdir(parents=True, exist_ok=True)
        gridpaths.append(gridpath)
        inpaths.append(inpath)
        outpaths.append(outpath)
        # cdo.remapcon(str(gridpath), input=str(inpath), output=str(outpath))
    # process_map(remap_one, gridpaths, inpaths, outpaths, max_workers=64, desc="Year")

    inpath = save_dir / "Carboscope" / "surfflux" / f"sEXTocNEET_co2flux.nc"
    outpath = (
        save_dir / "Carboscope" / f"{gridname}_surfflux" / f"sEXTocNEET_co2flux.nc"
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)

    gridpaths.append(gridpath)
    inpaths.append(inpath)
    outpaths.append(outpath)
    # cdo.remapcon(str(gridpath), input=str(inpath), output=str(outpath))

    print("Done!")

    print("Remapping Carboscope Test set to ICON")
    for year in tqdm(range(2018, 2022), position=0, leave=True, desc="Year"):
        inpath = save_dir / "Carboscope" / "mixratio" / f"s93oc_v2022_mix_{year}.nc"
        outpath = (
            save_dir
            / "Carboscope"
            / f"{gridname}_mixratio"
            / f"s93oc_v2022_mix_{year}.nc"
        )
        outpath.parent.mkdir(parents=True, exist_ok=True)

        gridpaths.append(gridpath)
        inpaths.append(inpath)
        outpaths.append(outpath)
        # cdo.remapcon(str(gridpath), input=str(inpath), output=str(outpath))
    # process_map(remap_one, gridpaths, inpaths, outpaths, max_workers=64, desc="Year")

    inpath = save_dir / "Carboscope" / "surfflux" / f"s93oc_co2flux.nc"
    outpath = save_dir / "Carboscope" / f"{gridname}_surfflux" / f"s93oc_co2flux.nc"
    outpath.parent.mkdir(parents=True, exist_ok=True)

    gridpaths.append(gridpath)
    inpaths.append(inpath)
    outpaths.append(outpath)
    # cdo.remapcon(str(gridpath), input=str(inpath), output=str(outpath))

    print("Done!")

    print("Remapping NCEP Meteo to ICON")
    for var in tqdm(
        ["air", "omega", "rhum", "shum", "uwnd", "vwnd"],
        position=0,
        leave=True,
        desc="Variable",
    ):
        for year in tqdm(range(1957, 2022), position=0, leave=False, desc="Year"):
            inpath = save_dir / "NCEP_NCAR_Reanalysis" / "latlon" / f"{var}.{year}.nc"
            outpath = (
                save_dir / "NCEP_NCAR_Reanalysis" / f"{gridname}" / f"{var}.{year}.nc"
            )
            outpath.parent.mkdir(parents=True, exist_ok=True)

            # cdo.remapcon(str(gridpath), input=str(inpath), output=str(outpath))
            gridpaths.append(gridpath)
            inpaths.append(inpath)
            outpaths.append(outpath)
        # cdo.remapcon(str(gridpath), input=str(inpath), output=str(outpath))
    process_map(
        remap_one, gridpaths, inpaths, outpaths, max_workers=64, desc="All_Jobs"
    )

    print("Done!")


def remap_to_icon(save_dir):
    save_dir = Path(save_dir)

    cdo = Cdo()

    gridpath = save_dir / "icon_grid" / "R02B04_G.nc"

    # print("Remapping Carboscope Train set to ICON")
    # for year in tqdm(range(1957, 2017), position=0, leave=True, desc="Year"):
    #     inpath = save_dir/"Carboscope"/"mixratio"/f"sEXTocNEET_v4.3_mix_{year}.nc"
    #     outpath = save_dir/"Carboscope"/"icon_mixratio"/f"sEXTocNEET_v4.3_mix_{year}.nc"

    #     cdo.remapcon(str(gridpath),input=str(inpath),output=str(outpath))

    inpath = save_dir / "Carboscope" / "surfflux" / f"sEXTocNEET_co2flux.nc"
    outpath = save_dir / "Carboscope" / "icon_surfflux" / f"sEXTocNEET_co2flux.nc"

    cdo.remapcon(str(gridpath), input=str(inpath), output=str(outpath))

    print("Done!")

    # print("Remapping Carboscope Test set to ICON")
    # for year in tqdm(range(2018, 2022), position=0, leave=True, desc="Year"):
    #     inpath = save_dir/"Carboscope"/"mixratio"/f"s93oc_v2022_mix_{year}.nc"
    #     outpath = save_dir/"Carboscope"/"icon_mixratio"/f"s93oc_v2022_mix_{year}.nc"

    #     cdo.remapcon(str(gridpath),input=str(inpath),output=str(outpath))

    inpath = save_dir / "Carboscope" / "surfflux" / f"s93oc_co2flux.nc"
    outpath = save_dir / "Carboscope" / "icon_surfflux" / f"s93oc_co2flux.nc"

    cdo.remapcon(str(gridpath), input=str(inpath), output=str(outpath))

    print("Done!")

    # print("Remapping NCEP Meteo to ICON")
    # for var in tqdm(["air", "omega", "rhum", "shum", "uwnd", "vwnd"], position=0, leave=True, desc="Variable"):
    #     for year in tqdm(range(1957, 2022), position=0, leave=False, desc="Year"):

    #         inpath = save_dir/"NCEP_NCAR_Reanalysis"/"latlon"/f"{var}.{year}.nc"
    #         outpath = save_dir/"NCEP_NCAR_Reanalysis"/"icon"/f"{var}.{year}.nc"

    #         cdo.remapcon(str(gridpath),input=str(inpath),output=str(outpath))

    # print("Done!")


def icon_to_zarr(save_dir, gridtype="icon"):
    save_dir = Path(save_dir)
    train_dir = save_dir / "Carboscope" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    test_dir = save_dir / "Carboscope" / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    if gridtype == "icon":
        grid = xr.open_dataset(save_dir / f"icon_grid" / "R02B04_G.nc")
    else:
        grid = xr.open_dataset(save_dir / f"icosa_grid" / f"{gridtype}.nc")

    mixing_ratios_train = xr.open_mfdataset(
        (save_dir / "Carboscope" / f"{gridtype}_mixratio").glob("sEXTocNEET_*.nc")
    )
    mixing_ratios_test = xr.open_mfdataset(
        [
            save_dir
            / "Carboscope"
            / f"{gridtype}_mixratio"
            / f"s93oc_v2022_mix_{year}.nc"
            for year in range(2018, 2022)
        ]
    )
    fluxes_train = xr.open_dataset(
        save_dir / "Carboscope" / f"{gridtype}_surfflux" / "sEXTocNEET_co2flux.nc"
    ).sel(mtime=slice("1957-01-01", "2017-12-31"))
    fluxes_test = xr.open_dataset(
        save_dir / "Carboscope" / f"{gridtype}_surfflux" / "s93oc_co2flux.nc"
    ).sel(mtime=slice("2018-01-01", "2022-01-01"))

    all_vars_train = {}
    all_vars_test = {}
    for v in ["co2flux_land", "co2flux_ocean", "co2flux_subt"]:
        all_vars_train[v] = (
            fluxes_train[v]
            .rename({"mtime": "time"})
            .interp(
                time=mixing_ratios_train.time.values,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
        )
        all_vars_test[v] = (
            fluxes_test[v]
            .rename({"mtime": "time"})
            .interp(
                time=mixing_ratios_test.time.values,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
        )

    co2_train = mixing_ratios_train.co2mix + 312.788
    co2_test = mixing_ratios_test.co2mix + 310.052

    all_vars_train["gp"] = mixing_ratios_train.gph.assign_attrs(
        {"long_name": "Geopotential Height", "units": "m"}
    )
    all_vars_test["gp"] = mixing_ratios_test.gph.assign_attrs(
        {"long_name": "Geopotential Height", "units": "m"}
    )

    for v in ["air", "omega", "rhum", "shum", "uwnd", "vwnd"]:
        varname = {
            "air": "t",
            "omega": "omeg",
            "rhum": "r",
            "shum": "q",
            "uwnd": "u",
            "vwnd": "v",
        }[v]
        attrs = {
            "t": {"long_name": "Air temperature", "units": "K"},
            "omeg": {"long_name": "Omega", "units": "Pa s^-1"},
            "r": {"long_name": "Relative humidity", "units": "%"},
            "q": {"long_name": "Specific humidity", "units": "kg/kg"},
            "u": {"long_name": "U wind", "units": "m/s"},
            "v": {"long_name": "V wind", "units": "m/s"},
        }[varname]
        ncep = xr.open_mfdataset(
            (save_dir / "NCEP_NCAR_Reanalysis" / f"{gridtype}").glob(f"{v}*.nc")
        )[v]
        new_levels = ncep.level.values
        new_levels[np.argmax(new_levels)] = 1013
        ncep["level"] = new_levels
        ncep = ncep.rename({"level": "height"})

        all_vars_train[varname] = (
            ncep.chunk({"time": 10, "height": -1, "cell": -1})
            .interp(
                height=co2_train.height.values,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            .sel(time=co2_train.time.values, method="nearest")
            .assign_attrs(attrs)
        )
        all_vars_test[varname] = (
            ncep.chunk({"time": 10, "height": -1, "cell": -1})
            .interp(
                height=co2_test.height.values,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            .sel(time=co2_test.time.values, method="nearest")
            .assign_attrs(attrs)
        )

    for all_vars, co2 in zip([all_vars_train, all_vars_test], [co2_train, co2_test]):
        gph = all_vars["gp"]
        T = all_vars["t"]
        RH = all_vars["r"]

        dxyp = grid.cell_area * 6.375e6**2

        T_celsius = T - 273.15

        Psat = 0.61121 * np.exp(
            (18.678 - T_celsius / 234.5) * (T_celsius / (257.14 + T_celsius))
        )

        Pv = RH / 100 * Psat
        Pd = gph.height - 10 * Pv

        rho = 100 * Pd / (287.050676 * T)

        midpoints = (
            gph.isel(height=slice(-1)).assign_coords(
                {"height": gph.height.isel(height=slice(1, None))}
            )
            + gph.isel(height=slice(1, None))
        ) / 2
        V = dxyp * xr.concat(
            [
                midpoints.isel(height=0).assign_coords(
                    {"height": gph.height.isel(height=0)}
                )
                - gph.isel(height=0),
                midpoints.diff("height", label="lower"),
                -midpoints.isel(height=-1) + gph.isel(height=-1),
            ],
            dim="height",
        )

        all_vars["airdensity"] = rho.assign_attrs(
            {
                "CDI_grid_type": "unstructured",
                "long_name": "Dry Air Density",
                "standard_name": "dry_air_density",
                "units": "kg m^-3",
            }
        )
        all_vars["volume"] = V.assign_attrs(
            {
                "CDI_grid_type": "unstructured",
                "long_name": "Grid Cell Volume",
                "standard_name": "grid_cell_volume",
                "units": "m^3",
            }
        )
        all_vars["cell_area"] = dxyp

        co2density = (co2 * 44.009e-3 / 28.9652e-3 * 1e-6) * rho
        all_vars["co2density"] = co2density.assign_attrs(
            {
                "CDI_grid_type": "unstructured",
                "long_name": "Carbon dioxide density",
                "standard_name": "co2_density",
                "units": "kg m^-3",
            }
        )

    try:
        ds = xr.Dataset(all_vars_train).chunk({"time": 10, "height": -1, "cell": -1})
        ds_test = xr.Dataset(all_vars_test).chunk(
            {"time": 10, "height": -1, "cell": -1}
        )
    except:
        breakpoint()

    print("To Zarr")

    ds_train = ds.sel(time=slice(None, "2014-12-31"))
    ds_val = ds.sel(time=slice("2015-01-01", "2016-12-31"))

    for split, ds in zip(["val", "test", "train"], [ds_val, ds_test, ds_train]):
        ds_2d = (
            ds[[v for v in ds.data_vars if "height" not in ds[v].dims]]
            .to_array("vari_2d")
            .chunk({"time": 50, "vari_2d": -1, "cell": -1})
            .transpose("time", "vari_2d", "cell")
            .astype("float32")
        )

        ds_3d = (
            ds[[v for v in ds.data_vars if "height" in ds[v].dims]]
            .to_array("vari_3d")
            .chunk({"time": 1, "vari_3d": -1, "height": -1, "cell": -1})
            .transpose("time", "vari_3d", "height", "cell")
            .astype("float32")
        )
        ds2 = xr.Dataset({"variables_2d": ds_2d, "variables_3d": ds_3d})
        print(f"Writing {split} to zarr")
        out_dir = save_dir / "Carboscope" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        with ProgressBar():
            ds2.to_zarr(
                out_dir / f"carboscope_{gridtype}.zarr",
                mode="w",
            )

    # with ProgressBar():
    #     ds_test.to_zarr(test_dir / f"carboscope_{gridtype}.zarr")

    # with ProgressBar():
    #     ds_train.to_zarr(train_dir / f"carboscope_{gridtype}.zarr")


def latlon_to_zarr(save_dir, higher_res=False):
    save_dir = Path(save_dir)
    train_dir = save_dir / "Carboscope" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    test_dir = save_dir / "Carboscope" / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    if higher_res:
        grid = xr.open_dataset(
            save_dir / "Carboscope" / "surfflux" / "s93oc_v2022_daily.nc"
        )[["dxyp"]]

        ds_out = xr.Dataset(
            {
                "lat": (["lat"], grid.lat.values, {"units": "degrees_north"}),
                "lon": (["lon"], grid.lon.values, {"units": "degrees_east"}),
            }
        )

    else:
        grid = xr.open_dataset(
            save_dir / "Carboscope" / "surfflux" / "sEXTocNEET_v4.3_daily.nc"
        )[["dxyp"]]

        lat = np.linspace(-88, 88, 45)
        ds_out = xr.Dataset(
            {
                "lat": (["lat"], lat, {"units": "degrees_north"}),
                "lon": (["lon"], grid.lon.values, {"units": "degrees_east"}),
                "dxyp": (
                    ["lat", "lon"],
                    np.stack(
                        len(grid.lon.values)
                        * [
                            np.pi
                            * 6.375e6**2
                            * (
                                np.sin(np.radians(lat + 2))
                                - np.sin(np.radians(lat - 2))
                            )
                            * 5
                            / 180
                        ],
                        axis=-1,
                    ),
                ),
            }
        )
        grid = ds_out

    mixing_ratios_train = xr.open_mfdataset(
        (save_dir / "Carboscope" / "mixratio").glob("sEXTocNEET_*.nc")
    )
    mixing_ratios_test = xr.open_mfdataset(
        [
            save_dir / "Carboscope" / "mixratio" / f"s93oc_v2022_mix_{year}.nc"
            for year in range(2018, 2022)
        ]
    )
    fluxes_train = xr.open_dataset(
        save_dir / "Carboscope" / "surfflux" / "sEXTocNEET_co2flux.nc"
    ).sel(mtime=slice("1957-01-01", "2017-12-31"))
    fluxes_test = xr.open_dataset(
        save_dir / "Carboscope" / "surfflux" / "s93oc_co2flux.nc"
    ).sel(mtime=slice("2018-01-01", "2022-01-01"))

    regridder = xe.Regridder(
        mixing_ratios_train, ds_out, "conservative_normed", periodic=True
    )
    mixing_ratios_train = regridder(
        mixing_ratios_train[["co2mix", "gph"]], keep_attrs=True
    )

    regridder = xe.Regridder(
        mixing_ratios_test, ds_out, "conservative_normed", periodic=True
    )
    mixing_ratios_test = regridder(
        mixing_ratios_test[["co2mix", "gph"]], keep_attrs=True
    )

    regridder = xe.Regridder(fluxes_train, ds_out, "conservative_normed", periodic=True)
    fluxes_train = regridder(
        fluxes_train[["co2flux_land", "co2flux_ocean", "co2flux_subt"]], keep_attrs=True
    )

    regridder = xe.Regridder(fluxes_test, ds_out, "conservative_normed", periodic=True)
    fluxes_test = regridder(
        fluxes_test[["co2flux_land", "co2flux_ocean", "co2flux_subt"]], keep_attrs=True
    )

    all_vars_train = {}
    all_vars_test = {}
    for v in ["co2flux_land", "co2flux_ocean", "co2flux_subt"]:
        all_vars_train[v] = (
            fluxes_train[v]
            .rename({"mtime": "time"})
            .interp(
                time=mixing_ratios_train.time.values,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
        )
        all_vars_test[v] = (
            fluxes_test[v]
            .rename({"mtime": "time"})
            .interp(
                time=mixing_ratios_test.time.values,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
        )

    co2_train = mixing_ratios_train.co2mix + 312.788
    co2_test = mixing_ratios_test.co2mix + 310.052

    all_vars_train["gp"] = mixing_ratios_train.gph.assign_attrs(
        {"long_name": "Geopotential Height", "units": "m"}
    )
    all_vars_test["gp"] = mixing_ratios_test.gph.assign_attrs(
        {"long_name": "Geopotential Height", "units": "m"}
    )

    for v in ["air", "omega", "rhum", "shum", "uwnd", "vwnd"]:
        varname = {
            "air": "t",
            "omega": "omeg",
            "rhum": "r",
            "shum": "q",
            "uwnd": "u",
            "vwnd": "v",
        }[v]
        attrs = {
            "t": {"long_name": "Air temperature", "units": "K"},
            "omeg": {"long_name": "Omega", "units": "Pa s^-1"},
            "r": {"long_name": "Relative humidity", "units": "%"},
            "q": {"long_name": "Specific humidity", "units": "kg/kg"},
            "u": {"long_name": "U wind", "units": "m/s"},
            "v": {"long_name": "V wind", "units": "m/s"},
        }[varname]
        ncep = xr.open_mfdataset(
            (save_dir / "NCEP_NCAR_Reanalysis" / "latlon").glob(f"{v}*.nc")
        )

        regridder = xe.Regridder(ncep, ds_out, "conservative_normed", periodic=True)
        ncep = regridder(ncep[[v]], keep_attrs=True)[v]

        new_levels = ncep.level.values
        new_levels[np.argmax(new_levels)] = 1013
        ncep["level"] = new_levels
        ncep = ncep.rename({"level": "height"})

        all_vars_train[varname] = (
            ncep.chunk({"time": 10, "height": -1, "lat": -1, "lon": -1})
            .interp(
                height=co2_train.height.values,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            .sel(time=co2_train.time.values, method="nearest")
            .assign_attrs(attrs)
        )
        all_vars_test[varname] = (
            ncep.chunk({"time": 10, "height": -1, "lat": -1, "lon": -1})
            .interp(
                height=co2_test.height.values,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            .sel(time=co2_test.time.values, method="nearest")
            .assign_attrs(attrs)
        )

    for all_vars, co2 in zip([all_vars_train, all_vars_test], [co2_train, co2_test]):
        gph = all_vars["gp"]
        T = all_vars["t"]
        RH = all_vars["r"]

        dxyp = grid.dxyp

        T_celsius = T - 273.15

        Psat = 0.61121 * np.exp(
            (18.678 - T_celsius / 234.5) * (T_celsius / (257.14 + T_celsius))
        )

        Pv = RH / 100 * Psat
        Pd = gph.height - 10 * Pv

        rho = 100 * Pd / (287.050676 * T) # THIS USES THE WRONG PRESSURE FIELD !

        midpoints = (
            gph.isel(height=slice(-1)).assign_coords(
                {"height": gph.height.isel(height=slice(1, None))}
            )
            + gph.isel(height=slice(1, None))
        ) / 2
        V = dxyp * xr.concat(
            [
                midpoints.isel(height=0).assign_coords(
                    {"height": gph.height.isel(height=0)}
                )
                - gph.isel(height=0),
                midpoints.diff("height", label="lower"),
                -midpoints.isel(height=-1) + gph.isel(height=-1),
            ],
            dim="height",
        )

        all_vars["airdensity"] = rho.assign_attrs(
            {
                "long_name": "Dry Air Density",
                "standard_name": "dry_air_density",
                "units": "kg m^-3",
            }
        )
        all_vars["volume"] = V.assign_attrs(
            {
                "long_name": "Grid Cell Volume",
                "standard_name": "grid_cell_volume",
                "units": "m^3",
            }
        )
        all_vars["cell_area"] = dxyp

        co2density = (co2 * 44.009e-3 / 28.9652e-3 * 1e-6) * rho
        all_vars["co2density"] = co2density.assign_attrs(
            {
                "long_name": "Carbon dioxide density",
                "standard_name": "co2_density",
                "units": "kg m^-3",
            }
        )

    ds = xr.Dataset(all_vars_train).chunk(
        {"time": 1, "height": -1, "lat": -1, "lon": -1}
    )
    ds["tisr"] = get_toa_incident_solar_radiation_for_xarray(ds)
    ds_test = xr.Dataset(all_vars_test).chunk(
        {"time": 1, "height": -1, "lat": -1, "lon": -1}
    )
    ds_test["tisr"] = get_toa_incident_solar_radiation_for_xarray(ds_test)

    ds_train = ds.sel(time=slice(None, "2014-12-31"))
    ds_val = ds.sel(time=slice("2015-01-01", "2016-12-31"))

    for split, ds in zip(["val", "test", "train"], [ds_val, ds_test, ds_train]):
        ds_2d = (
            ds[[v for v in ds.data_vars if "height" not in ds[v].dims]]
            .to_array("vari_2d")
            .chunk({"time": 50, "vari_2d": -1, "lat": -1, "lon": -1})
            .transpose("time", "vari_2d", "lat", "lon")
            .astype("float32")
        )

        ds_3d = (
            ds[[v for v in ds.data_vars if "height" in ds[v].dims]]
            .to_array("vari_3d")
            .chunk({"time": 1, "vari_3e": -1, "height": -1, "lat": -1, "lon": -1})
            .transpose("time", "vari_3e", "height", "lat", "lon")
            .astype("float32")
        )
        ds2 = xr.Dataset({"variables_2d": ds_2d, "variables_3d": ds_3d})
        print(f"Writing {split} to zarr")
        out_dir = save_dir / "Carboscope" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        with ProgressBar():
            ds2.to_zarr(
                out_dir / f"carboscope_{'latlon2' if higher_res else 'latlon4'}.zarr",
                mode="w",
            )

    # print("To Zarr")

    # with ProgressBar():

    #     ds_test.to_zarr(
    #         test_dir / "carboscope_latlon.zarr"
    #         if higher_res
    #         else test_dir / "carboscope_latlon_lowres.zarr"
    #     )

    # with ProgressBar():
    #     ds_train.to_zarr(
    #         train_dir / "carboscope_latlon.zarr"
    #         if higher_res
    #         else train_dir / "carboscope_latlon_lowres.zarr"
    #     )

    # co2_train = xr.open_zarr(train_dir/"co2.zarr")
    # co2_test = xr.open_zarr(test_dir/"co2.zarr")

    # for var in tqdm(["air", "omega", "rhum", "shum", "uwnd", "vwnd"]):
    #     ncep = xr.open_mfdataset((save_dir/"NCEP_NCAR_Reanalysis"/"icon").glob(f"{var}*.nc"))#.compute()
    #     print(f"Regridding NCEP {var}")
    #     new_levels = ncep.level.values
    #     new_levels[np.argmax(new_levels)] = 1013
    #     ncep["level"] = new_levels
    #     ncep = ncep.rename({"level": "height"}).chunk({"time":10, "height": -1, "cell": -1})[[var]]

    #     print("To Zarr")
    #     ncep_train = ncep.chunk({"time":10, "height": -1, "cell": -1}).interp(height = co2_train.height.values, method = "linear", kwargs={"fill_value": "extrapolate"}).sel(time = co2_train.time.values, method = "nearest").chunk({"time":10, "height": -1, "cell": -1})
    #     with ProgressBar():
    #         ncep_train.to_zarr(train_dir/f"ncep_{var}.zarr")

    #     ncep_test = ncep.chunk({"time":10, "height": -1, "cell": -1}).interp(height = co2_test.height.values, method = "linear", kwargs={"fill_value": "extrapolate"}).sel(time = co2_test.time.values, method = "nearest").chunk({"time":10, "height": -1, "cell": -1})
    #     with ProgressBar():
    #         ncep_test.to_zarr(test_dir/f"ncep_{var}.zarr")

    # for curr_dir in [train_dir, test_dir]:

    #     gph = xr.open_zarr(curr_dir/"co2.zarr").gph
    #     T = xr.open_zarr(curr_dir/"ncep_air.zarr").air
    #     RH = xr.open_zarr(curr_dir/"ncep_rhum.zarr").rhum
    #     dxyp = xr.open_dataset(save_dir/"icon_grid"/"R02B04_G.nc").cell_area

    #     T_celsius = T - 273.15

    #     Psat = 0.61121 * np.exp((18.678 - T_celsius/234.5)*(T_celsius/(257.14 + T_celsius)))

    #     Pv = RH/100 * Psat
    #     Pd = gph.height - 10*Pv

    #     rho = 100*Pd / (287.050676*T)

    #     V = dxyp * xr.concat([gph.diff("height", label = "lower"), 50000-gph.isel(height = -1)], dim = "height")

    #     m_air = rho * V * 1e-6

    #     dryairmass = m_air.to_dataset(name = "dryairmass")
    #     dryairmass.dryairmass.attrs = {"var_desc" : "Dry Air Mass", "units": "t"}

    #     with ProgressBar():
    #         dryairmass.chunk({"time":10, "height": -1, "cell": -1}).to_zarr(curr_dir/f"dryairmass.zarr")


def stats_dataset(save_dir):
    save_dir = Path(save_dir)
    train_dir = save_dir / "Carboscope" / "train"
    test_dir = save_dir / "Carboscope" / "test"

    ds = xr.open_zarr(train_dir / "carboscope_latlon4.zarr").sel(
        time=slice(None, "2015-12-31")
    )

    # ds_min = ds.min().to_array("var")
    # ds_mean = ds.mean().to_array("var")
    # ds_max = ds.max().to_array("var")
    # ds_std = ds.std().to_array("var")
    # with ProgressBar():
    #     ds_stats = xr.concat([ds_min, ds_mean, ds_max, ds_std], "stats").assign_coords({"stats": ["min", "mean", "max", "std"]}).to_dataset("var").compute()

    # ds_stats["co2density_next"] = ds_stats.co2density

    # co2density_delta = ds.co2density.diff("time")
    # with ProgressBar():
    #     delta_stats = xr.concat([co2density_delta.min(["time", "cell"]),co2density_delta.mean(["time", "cell"]),co2density_delta.max(["time", "cell"]),co2density_delta.std(["time", "cell"])], "stats").assign_coords({"stats": ["min", "mean", "max", "std"]}).compute()
    # ds_stats["co2density_delta"] = delta_stats

    ds_stats = xr.open_dataset(train_dir / "carboscope_stats.nc")

    # ds_stats["airdensity_next"] = ds_stats.airdensity

    # co2massmix = density_to_massmix(ds.co2density, ds.airdensity, ppm=True, eps=1e-12)
    # co2massmix_delta = co2massmix.diff("time")

    # with ProgressBar():
    #     massmix_stats = (
    #         xr.concat(
    #             [
    #                 co2massmix.min(),
    #                 co2massmix.mean(),
    #                 co2massmix.max(),
    #                 co2massmix.std(),
    #             ],
    #             "stats",
    #         )
    #         .assign_coords({"stats": ["min", "mean", "max", "std"]})
    #         .compute()
    #     )
    # ds_stats["co2massmix"] = massmix_stats
    # ds_stats["co2massmix_next"] = massmix_stats

    # ds_stats["co2molemix"] = massmix_to_molemix(massmix_stats)
    # ds_stats["co2molemix_next"] = massmix_to_molemix(massmix_stats)

    # with ProgressBar():
    #     massmixdelta_stats = (
    #         xr.concat(
    #             [
    #                 co2massmix_delta.min(["time", "cell"]),
    #                 co2massmix_delta.mean(["time", "cell"]),
    #                 co2massmix_delta.max(["time", "cell"]),
    #                 co2massmix_delta.std(["time", "cell"]),
    #             ],
    #             "stats",
    #         )
    #         .assign_coords({"stats": ["min", "mean", "max", "std"]})
    #         .compute()
    #     )
    # ds_stats["co2massmix_delta"] = massmixdelta_stats
    # ds_stats["co2molemix_delta"] = massmix_to_molemix(massmixdelta_stats)

    all_stats = {}
    for k in (pbar := tqdm(ds.data_vars)):
        pbar.set_postfix_str(k)
        if k == "cell_area":
            continue
        ds_delta = ds[k].compute().diff("time")
        ds_min = ds_delta.min(["time", "lat", "lon"])  # .to_array("var")
        ds_mean = ds_delta.mean(["time", "lat", "lon"])  # .to_array("var")
        ds_max = ds_delta.max(["time", "lat", "lon"])  # .to_array("var")
        ds_std = ds_delta.std(["time", "lat", "lon"])  # .to_array("var")
        all_stats[f"{k}_delta"] = xr.concat([ds_min, ds_mean, ds_max, ds_std], "stats")

    ds_delta_stats = xr.Dataset(all_stats)

    # ds_delta = ds.diff("time")
    # ds_min = ds_delta.min(["time", "lat", "lon"])  # .to_array("var")
    # ds_mean = ds_delta.mean(["time", "lat", "lon"])  # .to_array("var")
    # ds_max = ds_delta.max(["time", "lat", "lon"])  # .to_array("var")
    # ds_std = ds_delta.std(["time", "lat", "lon"])  # .to_array("var")

    # with ProgressBar():
    #     ds_delta_stats = xr.Dataset(
    #         {
    #             f"{k}_delta": xr.concat(
    #                 [ds_min[k], ds_mean[k], ds_max[k], ds_std[k]], "stats"
    #             )
    #             for k in ds_min.data_vars
    #         }
    #     ).compute()
    #     # ds_delta_stats = (
    #     #     xr.concat([ds_min, ds_mean, ds_max, ds_std], "stats")
    #     #     .assign_coords({"stats": ["min", "mean", "max", "std"]})
    #     #     .to_dataset("var")
    #     #     .compute()
    #     # ).rename({v: f"{v}_delta" for v in ds.data_vars})
    ds_delta_stats.to_netcdf(train_dir / f"carboscope_delta_stats.nc")

    ds_stats = ds_delta_stats.merge(ds_stats, compat="override")

    # for v in ds.data_vars:
    #     if "height" not in ds[v].dims:
    #         ds_stats[f"{v}_delta"] = ds_stats[f"{v}_delta"].isel(height=0)
    # ds_stats = ds_stats.drop_vars(
    #     [
    #         k
    #         for k in ds_stats.data_vars
    #         if (k.endswith("_delta_next") or k.endswith("_next_next"))
    #     ]
    # )

    ds_stats.to_netcdf(train_dir / "carboscope_stats2.nc")
    ds_stats.to_netcdf(test_dir / "carboscope_stats2.nc")


def stddev_weights(save_dir):
    save_dir = Path(save_dir)
    train_dir = save_dir / "Carboscope" / "train"
    test_dir = save_dir / "Carboscope" / "test"

    ds = xr.open_zarr(train_dir / "carboscope_icon.zarr").sel(
        time=slice(None, "2015-12-31")
    )

    co2density_per_cell = ds.co2density.std("time")
    co2massmix_per_cell = density_to_massmix(
        ds.co2density, ds.airdensity, ppm=True, eps=1e-12
    ).std("time")

    co2density_delta_per_cell = ds.co2density.diff("time").std("time")
    co2massmix_delta_per_cell = (
        density_to_massmix(ds.co2density, ds.airdensity, ppm=True, eps=1e-12)
        .diff("time")
        .std("time")
    )

    co2density_per_level = ds.co2density.std(["time", "cell"])
    co2massmix_per_level = density_to_massmix(
        ds.co2density, ds.airdensity, ppm=True, eps=1e-12
    ).std(["time", "cell"])

    co2density_delta_per_level = ds.co2density.diff("time").std(["time", "cell"])
    co2massmix_delta_per_level = (
        density_to_massmix(ds.co2density, ds.airdensity, ppm=True, eps=1e-12)
        .diff("time")
        .std(["time", "cell"])
    )

    with ProgressBar():
        stddev_weights = xr.Dataset(
            {
                "co2density_per_cell": co2density_per_cell,
                "co2massmix_per_cell": co2massmix_per_cell,
                "co2density_delta_per_cell": co2density_delta_per_cell,
                "co2massmix_delta_per_cell": co2massmix_delta_per_cell,
                "co2density_per_level": co2density_per_level,
                "co2massmix_per_level": co2massmix_per_level,
                "co2density_delta_per_level": co2density_delta_per_level,
                "co2massmix_delta_per_level": co2massmix_delta_per_level,
            }
        ).compute()

    stddev_weights.to_netcdf(train_dir / "carboscope_stddev.nc")
    stddev_weights.to_netcdf(test_dir / "carboscope_stddev.nc")


def y_rotation(vector, theta):
    """Rotates 3-D vector around y-axis"""
    theta = np.radians(theta)
    R = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    return np.dot(R, vector)


def z_rotation(vector, theta):
    """Rotates 3-D vector around z-axis"""
    theta = np.radians(theta)
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return np.dot(R, vector)


def latlon_to_xyz(lat, lon):
    x = np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    y = np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    z = np.sin(np.radians(lat))
    return x, y, z


def pos_in_local_grid(lat_receiver, lon_receiver, lat_sender, lon_sender):
    x, y, z = y_rotation(
        z_rotation(latlon_to_xyz(lat_sender, lon_sender), -lon_receiver), lat_receiver
    )
    return x, y, z


def multimesh_from_icon(save_dir):
    save_dir = Path(save_dir)

    grid = xr.open_dataset(save_dir / "icon_grid" / "R02B04_G.nc")

    print("Collecting Multimesh data")

    mm_x = xr.Dataset(
        {
            "area": (grid.cell_area - grid.cell_area.mean()) / grid.cell_area.std(),
            "cos_lat": np.cos(grid.lat_cell_centre),
            "sin_lon": np.sin(grid.lon_cell_centre),
            "cos_lon": np.cos(grid.lon_cell_centre),
        }
    )

    mm_edge_index = xr.Dataset(
        {
            "sender_idx": xr.concat(
                [
                    grid.adjacent_cell_of_edge.isel(nc=0) - 1,
                    grid.adjacent_cell_of_edge.isel(nc=1) - 1,
                ],
                dim="edge",
            ),
            "receiver_idx": xr.concat(
                [
                    grid.adjacent_cell_of_edge.isel(nc=1) - 1,
                    grid.adjacent_cell_of_edge.isel(nc=0) - 1,
                ],
                dim="edge",
            ),
        }
    )  # .to_dataframe()

    lat_receiver = np.degrees(
        grid.lat_cell_centre.isel(cell=mm_edge_index.receiver_idx.values)
    )
    lon_receiver = np.degrees(
        grid.lon_cell_centre.isel(cell=mm_edge_index.receiver_idx.values)
    )
    lat_sender = np.degrees(
        grid.lat_cell_centre.isel(cell=mm_edge_index.sender_idx.values)
    )
    lon_sender = np.degrees(
        grid.lon_cell_centre.isel(cell=mm_edge_index.sender_idx.values)
    )

    rel_pos = np.stack(
        [
            np.array(
                pos_in_local_grid(
                    lat_receiver[i], lon_receiver[i], lat_sender[i], lon_sender[i]
                )
            )
            for i in range(len(lat_receiver))
        ]
    )

    mm_edge_attr = xr.Dataset(
        {
            "dual_edge_length": xr.concat(
                [grid.dual_edge_length, grid.dual_edge_length], dim="edge"
            ),
            "edge_length": xr.concat([grid.edge_length, grid.edge_length], dim="edge"),
        }
    )
    mm_edge_attr["x_rel"] = (("edge",), rel_pos[:, 0])
    mm_edge_attr["y_rel"] = (("edge",), rel_pos[:, 1])
    mm_edge_attr["z_rel"] = (("edge",), rel_pos[:, 2])

    mm_x.to_netcdf(save_dir / "icon_multimesh" / "mm_x.nc")
    mm_edge_index.to_netcdf(save_dir / "icon_multimesh" / "mm_edge_index.nc")
    mm_edge_attr.to_netcdf(save_dir / "icon_multimesh" / "mm_edge_attr.nc")

    print("Done!")


def regrid_data(save_dir):
    save_dir = Path(save_dir)

    if False:
        mixing_ratios_train = xr.open_mfdataset(
            (save_dir / "carboscope_mixingratios").glob("s76_*.nc")
        )
        mixing_ratios_test = xr.open_mfdataset(
            [
                save_dir / "carboscope_mixingratios" / f"s93oc_v2022_mix_{year}.nc"
                for year in range(2018, 2022)
            ]
        )
        fluxes_train = xr.open_dataset(save_dir / "carboscope_flux/s76_v4.1_daily.nc")
        fluxes_test = xr.open_dataset(
            save_dir / "carboscope_flux/s93oc_v2022_daily.nc"
        ).sel(mtime=slice("2018-01-01", "2022-01-01"))

        ds_out = xr.Dataset(
            {
                "lat": (["lat"], fluxes_test.lat.values, {"units": "degrees_north"}),
                "lon": (["lon"], fluxes_test.lon.values, {"units": "degrees_east"}),
            }
        )

        print("Regridding Carboscope")
        regridder = xe.Regridder(
            mixing_ratios_train, ds_out, "conservative_normed", periodic=True
        )

        mixing_ratios_train_regrid = regridder(
            mixing_ratios_train[["co2mix", "gph"]], keep_attrs=True
        )

        regridder = xe.Regridder(
            mixing_ratios_test, ds_out, "conservative_normed", periodic=True
        )

        mixing_ratios_test_regrid = regridder(
            mixing_ratios_test[["co2mix", "gph"]], keep_attrs=True
        )

        regridder = xe.Regridder(
            fluxes_train, ds_out, "conservative_normed", periodic=True
        )

        fluxes_train_regrid = regridder(fluxes_train, keep_attrs=True)

        fluxes_train_regrid["co2flux"] = (
            fluxes_train_regrid.co2flux_land
            + fluxes_train_regrid.co2flux_excl
            + fluxes_train_regrid.co2flux_ocean
            + fluxes_train_regrid.co2flux_subt
        )

        fluxes_test["co2flux"] = (
            fluxes_test.co2flux_land
            + fluxes_test.co2flux_excl
            + fluxes_test.co2flux_ocean
            + fluxes_test.co2flux_subt
        )

        # Interpolate Flux in Time to 6-hourly
        co2flux_train = fluxes_train_regrid.co2flux.rename({"mtime": "time"}).interp(
            time=mixing_ratios_train_regrid.time.values,
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )

        co2flux_test = fluxes_test.co2flux.rename({"mtime": "time"}).interp(
            time=mixing_ratios_test_regrid.time.values,
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )

        co2mix_train = mixing_ratios_train_regrid.co2mix + 310.052
        co2mix_test = mixing_ratios_test_regrid.co2mix + 310.052

        co2_train = xr.Dataset(
            {
                "co2flux": co2flux_train,
                "co2mix": co2mix_train,
                "gph": mixing_ratios_train_regrid.gph,
            }
        ).chunk({"time": 10, "height": -1, "lat": -1, "lon": -1})
        co2_test = xr.Dataset(
            {
                "co2flux": co2flux_test,
                "co2mix": co2mix_test,
                "gph": mixing_ratios_test_regrid.gph,
            }
        ).chunk({"time": 10, "height": -1, "lat": -1, "lon": -1})

        train_dir = save_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)

        test_dir = save_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)

        print("To Zarr")
        with ProgressBar():
            co2_train.to_zarr(train_dir / "co2.zarr")
        with ProgressBar():
            co2_test.to_zarr(test_dir / "co2.zarr")

    train_dir = save_dir / "train"
    test_dir = save_dir / "test"

    if False:
        co2_train = xr.open_zarr(train_dir / "co2.zarr")
        co2_test = xr.open_zarr(test_dir / "co2.zarr")

        for var in tqdm(["air", "omega", "rhum", "shum", "uwnd", "vwnd"]):
            ncep = xr.open_mfdataset(
                (save_dir / "ncep_meteo").glob(f"{var}*.nc")
            )  # .compute()
            print(f"Regridding NCEP {var}")
            new_levels = ncep.level.values
            new_levels[np.argmax(new_levels)] = 1013
            ncep["level"] = new_levels
            ncep = ncep.rename({"level": "height"}).chunk(
                {"time": 10, "height": -1, "lat": -1, "lon": -1}
            )

            regridder = xe.Regridder(ncep, ds_out, "bilinear", periodic=True)
            ncep_regrid = regridder(ncep, keep_attrs=True)

            print("To Zarr")
            ncep_train = (
                ncep_regrid.chunk({"time": 10, "height": -1, "lat": -1, "lon": -1})
                .interp(
                    height=co2_train.height.values,
                    method="linear",
                    kwargs={"fill_value": "extrapolate"},
                )
                .sel(time=co2_train.time.values, method="nearest")
                .chunk({"time": 10, "height": -1, "lat": -1, "lon": -1})
            )
            with ProgressBar():
                ncep_train.to_zarr(train_dir / f"ncep_{var}.zarr")

            ncep_test = (
                ncep_regrid.chunk({"time": 10, "height": -1, "lat": -1, "lon": -1})
                .interp(
                    height=co2_test.height.values,
                    method="linear",
                    kwargs={"fill_value": "extrapolate"},
                )
                .sel(time=co2_test.time.values, method="nearest")
                .chunk({"time": 10, "height": -1, "lat": -1, "lon": -1})
            )
            with ProgressBar():
                ncep_test.to_zarr(test_dir / f"ncep_{var}.zarr")

    for curr_dir in [train_dir, test_dir]:
        gph = xr.open_zarr(curr_dir / "co2.zarr").gph
        T = xr.open_zarr(curr_dir / "ncep_air.zarr").air
        RH = xr.open_zarr(curr_dir / "ncep_rhum.zarr").rhum
        dxyp = xr.open_dataset(save_dir / "carboscope_flux/s93oc_v2022_daily.nc").dxyp

        T_celsius = T - 273.15

        Psat = 0.61121 * np.exp(
            (18.678 - T_celsius / 234.5) * (T_celsius / (257.14 + T_celsius))
        )

        Pv = RH / 100 * Psat
        Pd = gph.height - 10 * Pv

        rho = 100 * Pd / (287.050676 * T)

        V = dxyp * xr.concat(
            [gph.diff("height", label="lower"), 50000 - gph.isel(height=-1)],
            dim="height",
        )

        m_air = rho * V * 1e-6

        dryairmass = m_air.to_dataset(name="dryairmass")
        dryairmass.dryairmass.attrs = {"var_desc": "Dry Air Mass", "units": "t"}

        with ProgressBar():
            dryairmass.chunk({"time": 10, "height": -1, "lat": -1, "lon": -1}).to_zarr(
                curr_dir / f"dryairmass.zarr"
            )


# def compute_weights(save_dir):
#     save_dir = Path(save_dir)
#     train_dir = save_dir/"train"
#     co2 = xr.open_zarr(train_dir/"co2.zarr").stack({"lat_lons": ("lat", "lon")}).transpose("time", "lat_lons", "height")

#     co2mix_min = 283.64737
#     co2mix_max = 490.48425
#     co2flux_min = -0.17495158
#     co2flux_max = 0.29721269

#     co2mix = (co2.co2mix - co2mix_min) / (co2mix_max - co2mix_min)

#     print("Saving Height Weights")
#     height_weights = co2.height/co2.height.sum()

#     weight_dir = save_dir/"weights"
#     weight_dir.mkdir(parents = True, exist_ok= True)

#     curr_path = weight_dir/"height_weights.nc"
#     if curr_path.is_file():
#         curr_path.unlink()

#     height_weights.to_netcdf(curr_path)

#     print("Loading CO2 Diff")
#     #with ProgressBar():
#     co2diff = co2mix.diff("time")#.compute().chunk({"height": -1})

#     print("Saving CO2 Diff Mean")
#     co2diff_mean = co2diff.mean(["time", "lat_lons"]).compute()

#     curr_path = weight_dir/"co2diff_mean.nc"
#     if curr_path.is_file():
#         curr_path.unlink()

#     co2diff_mean.to_netcdf(curr_path)

#     print("Saving CO2 Diff Std")
#     co2diff_std = co2diff.std(["time", "lat_lons"]).compute()

#     curr_path = weight_dir/"co2diff_std.nc"
#     if curr_path.is_file():
#         curr_path.unlink()

#     co2diff_std.to_netcdf(curr_path)

#     print("Saving CO2 Diff Min")
#     co2diff_min = co2diff.min(["time", "lat_lons"]).compute()

#     curr_path = weight_dir/"co2diff_min.nc"
#     if curr_path.is_file():
#         curr_path.unlink()

#     co2diff_min.to_netcdf(curr_path)

#     print("Saving CO2 Diff Max")
#     co2diff_max = co2diff.max(["time", "lat_lons"]).compute()

#     curr_path = weight_dir/"co2diff_max.nc"
#     if curr_path.is_file():
#         curr_path.unlink()

#     co2diff_max.to_netcdf(curr_path)

#     # print("Saving CO2 Diff q05")
#     # with ProgressBar():
#     #     co2diff_q05 = co2diff.quantile(q = 0.05, dim = "lat_lons", skipna=False).mean("time").compute()

#     # curr_path = weight_dir/"co2diff_q05.nc"
#     # if curr_path.is_file():
#     #     curr_path.unlink()

#     # co2diff_q05.to_netcdf(curr_path)

#     # print("Saving CO2 Diff q95")
#     # with ProgressBar():
#     #     co2diff_q95 = co2diff.quantile(q = 0.95, dim = "lat_lons", skipna=False).mean("time").compute()

#     # curr_path = weight_dir/"co2diff_q95.nc"
#     # if curr_path.is_file():
#     #     curr_path.unlink()

#     # co2diff_q95.to_netcdf(curr_path)

#     return height_weights, co2diff_mean, co2diff_std, co2diff_min, co2diff_max#, co2diff_q05, co2diff_q95


def download_obspack(data_dir):
    """
    Download Data from https://gml.noaa.gov/ccgg/obspack/data.php
    """

    data_dir = Path(data_dir)

    fp = "http://gml.noaa.gov/ccgg/obspack/tmp/obspack_nbVFeb/obspack_co2_1_GLOBALVIEWplus_v9.1_2023-12-08.nc.tar.gz"

    outpath = data_dir / "Obspack" / (fp.split("/")[-1])

    outpath.parent.mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(fp, str(outpath))

    shutil.unpack_archive(outpath, data_dir / "Obspack")


def open_one_obspack(obspack_path):
    try:
        obspack_obs = xr.open_dataset(obspack_path)
        obspack_obs = (
            xr.Dataset(
                {
                    "lon": ("time", obspack_obs.longitude.values),
                    "lat": ("time", obspack_obs.latitude.values),
                    "co2molemix": ("time", obspack_obs.value.values * 1e6),
                    "height": ("time", obspack_obs.altitude.values),
                },
                coords={"time": ("time", obspack_obs.time.values)},
            )
            .drop_duplicates("time")
            .sortby("time")
        )
        obspack_obs = obspack_obs.resample(
            time="6h", origin="start_day", offset="3h", closed="left", label="left"
        ).mean()
        offset = pd.tseries.frequencies.to_offset("6h") / 2
        obspack_obs["time"] = obspack_obs.get_index("time") + offset

        return {obspack_path.stem: obspack_obs.to_array("vari")}
    except:
        print(f"Error with {obspack_path}")
        return None


def prepare_obspack_for_carboscope(data_dir):
    data_dir = Path(data_dir)

    obspack_dir = data_dir / "Obspack"

    obspack_paths = sorted(
        list((data_dir / "Obspack").glob("obspack_co2_*/data/nc/*.nc"))
    )

    all_obs = process_map(open_one_obspack, obspack_paths, max_workers=32, chunksize=1)
    all_obs = sorted(
        [a for a in all_obs if a is not None], key=lambda x: list(x.keys())[0]
    )

    obs = xr.merge(all_obs, join="outer")
    obs = obs.to_array("cell").to_dataset("vari")
    obs["filename"] = ("cell", obs.cell.values)
    obs["cell"] = ("cell", np.arange(len(obs.cell)))

    obs = obs.chunk({"time": 2000, "cell": -1})

    with ProgressBar():
        obs.to_zarr(obspack_dir / "obspack.zarr")

    attrs = []
    for obspack_path in obspack_paths:
        id = obspack_path.stem
        molecule, station, type, level, quality = id.split("_")
        attrs.append(
            dict(
                id=id,
                molecule=molecule,
                station=station,
                type=type,
                level=level,
                quality=quality,
            )
            | xr.open_dataset(obspack_path).attrs
        )

    df = pd.DataFrame.from_records(attrs)
    df.to_csv(obspack_dir / "obspack_metadata.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("save_dir", type=str)

    args = parser.parse_args()

    icon_to_zarr(args.save_dir, gridtype="icosaL3")

    # icon_to_zarr(args.save_dir, gridtype="icosaL4-L1-a287")
    # download_data(args.save_dir)
    # download_icon_grid(args.save_dir)
    # aggregate_fluxes(args.save_dir)
    # remap_to_icosa(
    #     args.save_dir,
    #     level="L4",
    # )
    # remap_to_icon(args.save_dir)
    icon_to_zarr(args.save_dir, gridtype="icosaL4")
    # latlon_to_zarr(args.save_dir, higher_res=False)
    # stats_dataset(args.save_dir)
    # stddev_weights(args.save_dir)
    # multimesh_from_icon(args.save_dir)
    # regrid_data(args.save_dir)
    # compute_weights(args.save_dir)
    # prepare_obspack_for_carboscope(args.save_dir)
    # remap_to_icosa(
    #     args.save_dir,
    #     level="L3",
    # min_level=1,
    # resolved_locations=OBSPACK_287_LONLAT,
    # gridname="icosaL4-L1-a287",
    # )
    # remap_to_icon(args.save_dir)
