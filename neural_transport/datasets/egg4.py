from pathlib import Path

import cdsapi
import numpy as np
import requests
import xarray as xr
import xesmf as xe
import yaml
import zarr
from cdo import Cdo
from dask.diagnostics import ProgressBar
from numcodecs import Blosc
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map

from neural_transport.tools.conversion import *

requests.packages.urllib3.disable_warnings(
    requests.packages.urllib3.exceptions.InsecureRequestWarning
)


SINGLE_LEVEL_VARS_IMPORTANT = [
    "flux_of_carbon_dioxide_net_ecosystem_exchange",
    "anthropogenic_emissions_of_carbon_dioxide",
    "ocean_flux_of_carbon_dioxide",
    "wildfire_flux_of_carbon_dioxide",
    "wildfire_flux_of_methane",
    "methane_loss_rate_due_to_radical_hydroxyl_oh",
    "methane_surface_fluxes",
    "land_sea_mask",
]

SINGLE_LEVEL_VARS_EXTRA = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "accumulated_carbon_dioxide_ecosystem_respiration",
    "accumulated_carbon_dioxide_gross_primary_production",
    "accumulated_carbon_dioxide_net_ecosystem_exchange",
    "boundary_layer_height",
    "ch4_column_mean_molar_fraction",
    "co2_column_mean_molar_fraction",
    "convective_available_potential_energy",
    "convective_inhibition",
    "convective_precipitation",
    "downward_uv_radiation_at_the_surface",
    "evaporation",
    "flux_of_carbon_dioxide_ecosystem_respiration",
    "flux_of_carbon_dioxide_gross_primary_production",
    "forecast_albedo",
    "gpp_coefficient_from_biogenic_flux_adjustment_system",
    "high_cloud_cover",
    "large_scale_precipitation",
    "low_cloud_cover",
    "mean_sea_level_pressure",
    "medium_cloud_cover",
    "photosynthetically_active_radiation_at_the_surface",
    "potential_evaporation",
    "precipitation_type",
    "rec_coefficient_from_biogenic_flux_adjustment_system",
    "sea_ice_cover",
    "sea_surface_temperature",
    "skin_reservoir_content",
    "skin_temperature",
    "snow_albedo",
    "snow_depth",
    "sunshine_duration",
    "surface_geopotential",
    "surface_latent_heat_flux",
    "surface_net_solar_radiation",
    "surface_net_solar_radiation_clear_sky",
    "surface_net_thermal_radiation",
    "surface_net_thermal_radiation_clear_sky",
    "surface_sensible_heat_flux",
    "surface_solar_radiation_downward_clear_sky",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downward_clear_sky",
    "surface_thermal_radiation_downwards",
    "toa_incident_solar_radiation",
    "top_net_solar_radiation",
    "top_net_solar_radiation_clear_sky",
    "top_net_thermal_radiation",
    "top_net_thermal_radiation_clear_sky",
    "total_cloud_cover",
    "total_column_cloud_ice_water",
    "total_column_cloud_liquid_water",
    "total_column_water",
    "total_column_water_vapour",
    "total_precipitation",
    "visibility",
]

SINGLE_LEVEL_VARS = SINGLE_LEVEL_VARS_IMPORTANT + SINGLE_LEVEL_VARS_EXTRA

MULTI_LEVEL_VARS_IMPORTANT = [
    "carbon_dioxide",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "methane",
]

MULTI_LEVEL_VARS_EXTRA = [
    "fraction_of_cloud_cover",
    "geopotential",
    "logarithm_of_surface_pressure",
    "potential_vorticity",
    "relative_humidity",
    "specific_cloud_ice_water_content",
    "specific_cloud_liquid_water_content",
    "specific_humidity",
    "specific_rain_water_content",
    "specific_snow_water_content",
]

MULTI_LEVEL_VARS = MULTI_LEVEL_VARS_IMPORTANT + MULTI_LEVEL_VARS_EXTRA

PRESSURE_LEVELS = [
    "1",
    "2",
    "3",
    "5",
    "7",
    "10",
    "20",
    "30",
    "50",
    "70",
    "100",
    "150",
    "200",
    "250",
    "300",
    "400",
    "500",
    "600",
    "700",
    "800",
    "850",
    "900",
    "925",
    "950",
    "1000",
]

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
    """
    First: Obtain Credentials and save to ~/.adsapirc following the steps at https://ads.atmosphere.copernicus.eu/api-how-to
    Second: Accept License at https://ads.atmosphere.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products
    """

    save_dir = Path(save_dir)

    with open(Path.home() / ".adsapirc", "r") as f:
        credentials = yaml.safe_load(f)

    c = cdsapi.Client(url=credentials["url"], key=credentials["key"])

    print("Loading Multi Levels")
    multilevel_dir = save_dir / "multi_level"

    for year in (
        pbar1 := tqdm(range(2003, 2021), position=1, desc="Year", leave=False)
    ):
        pbar1.set_postfix({"year": year})

        year_dir = multilevel_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        for month, month_start, month_end in (
            pbar2 := tqdm(months, position=2, desc="Month", leave=False)
        ):
            if month == "02" and year in [2004, 2008, 2012, 2016, 2020, 2024]:
                month_end = "02-29"

            pbar2.set_postfix({"month": month})
            file_path = year_dir / f"{month}.nc"

            if not file_path.is_file():
                try:
                    c.retrieve(
                        "cams-global-ghg-reanalysis-egg4",
                        {
                            "format": "netcdf",
                            "variable": MULTI_LEVEL_VARS,
                            "pressure_level": PRESSURE_LEVELS,
                            "date": f"{year}-{month_start}/{year}-{month_end}",
                            "step": ["0", "3", "6", "9", "12", "15", "18", "21"],
                        },
                        str(file_path),
                    )

                except KeyboardInterrupt:
                    return
                except:
                    print(f"{year} {month} not working")

    singlelevel_dir = save_dir / "single_level"

    print("Loading Single Levels")

    for year in (
        pbar1 := tqdm(range(2003, 2021), position=1, desc="Year", leave=False)
    ):
        pbar1.set_postfix({"year": year})

        year_dir = singlelevel_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        for month, month_start, month_end in (
            pbar2 := tqdm(months, position=2, desc="Month", leave=False)
        ):
            pbar2.set_postfix({"month": month})
            file_path = year_dir / f"{month}.nc"

            if month == "02" and year in [2004, 2008, 2012, 2016, 2020, 2024]:
                month_end = "02-29"

            if not file_path.is_file():
                try:
                    c.retrieve(
                        "cams-global-ghg-reanalysis-egg4",
                        {
                            "format": "netcdf",
                            "variable": SINGLE_LEVEL_VARS,
                            "date": f"{year}-{month_start}/{year}-{month_end}",
                            "step": ["0", "3", "6", "9", "12", "15", "18", "21"],
                        },
                        str(file_path),
                    )

                except KeyboardInterrupt:
                    return
                except:
                    print(f"{year} {month} not working")

    print("Done!")


def remap_one_file(gridpath, inpath, outpath):
    cdo = Cdo()
    cdo.remapcon(str(gridpath), input=str(inpath), output=str(outpath))


def remap_to_icon(save_dir):
    save_dir = Path(save_dir)

    gridpath = save_dir / "icon_grid" / "R02B04_G.nc"

    print("Remapping CAMS EGG4 atmosphere to ICON")

    inpaths = []
    outpaths = []
    for year in tqdm(range(2003, 2021), position=0, leave=True, desc="Year"):
        for month in tqdm(range(1, 13), position=1, leave=False, desc="Month"):
            inpath = (
                save_dir
                / "CAMS_EGG4"
                / "multi_level"
                / str(year)
                / f"{str(month).zfill(2)}.nc"
            )
            outpath = (
                save_dir
                / "CAMS_EGG4"
                / "icon_multi_level"
                / str(year)
                / f"{str(month).zfill(2)}.nc"
            )

            outpath.parent.mkdir(exist_ok=True, parents=True)

            inpaths.append(inpath)
            outpaths.append(outpath)

    gridpaths = len(inpaths) * [gridpath]

    process_map(remap_one_file, gridpaths, inpaths, outpaths, max_workers=8)

    print("Done!")

    print("Remapping CAMS EGG4 atmosphere to ICON")

    inpaths = []
    outpaths = []
    for year in tqdm(range(2003, 2021), position=0, leave=True, desc="Year"):
        for month in tqdm(range(1, 13), position=1, leave=False, desc="Month"):
            inpath = (
                save_dir
                / "CAMS_EGG4"
                / "single_level"
                / str(year)
                / f"{str(month).zfill(2)}.nc"
            )
            outpath = (
                save_dir
                / "CAMS_EGG4"
                / "icon_single_level"
                / str(year)
                / f"{str(month).zfill(2)}.nc"
            )

            outpath.parent.mkdir(exist_ok=True, parents=True)

            inpaths.append(inpath)
            outpaths.append(outpath)

    gridpaths = len(inpaths) * [gridpath]

    process_map(remap_one_file, gridpaths, inpaths, outpaths, max_workers=8)

    print("Done!")


def latlon_to_zarr(data_dir):
    data_dir = Path(data_dir)
    print("Opening EGG4a")
    egg4a = xr.open_mfdataset(
        (data_dir / "CAMS_EGG4" / "multi_level").glob("*/*.nc")
    ).reset_encoding()
    print("Opening EGG4s")
    egg4s = xr.open_mfdataset(
        (data_dir / "CAMS_EGG4" / "single_level").glob("*/*.nc")
    ).reset_encoding()
    print("Merging EGG4")
    egg4 = xr.merge([egg4a, egg4s.rename({"z": "z_surf"})])

    print("Regridding EGG4")
    lat = np.linspace(-90, 90, 181)  # GraphCast 1° ERA5 compliant
    lon = np.linspace(0, 359, 360)
    lat_b = np.concatenate([[-90], np.linspace(-89.5, 89.5, 180), [90]])
    lon_b = np.concatenate([[359.5], np.linspace(0.5, 359.5, 360)])
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], lat, {"units": "degrees_north"}),
            "lon": (["lon"], lon, {"units": "degrees_east"}),
            "lat_b": (["lat_b"], lat_b, {"units": "degrees_north"}),
            "lon_b": (["lon_b"], lon_b, {"units": "degrees_east"}),
            "dxyp": (
                ["lat", "lon"],
                np.stack(
                    len(lon)
                    * [
                        np.pi
                        * 6.375e6**2
                        * (
                            np.sin(np.radians(lat_b[1:]))
                            - np.sin(np.radians(lat_b[:-1]))
                        )
                        * 1
                        / 180
                    ],
                    axis=-1,
                ),
            ),
        }
    )

    regridder = xe.Regridder(egg4, ds_out, "conservative_normed", periodic=True)
    egg4 = regridder(egg4, keep_attrs=True)

    print("Computing extra vars")
    gph = egg4.z / 9.80665
    T = egg4.t
    RH = egg4.r
    dxyp = ds_out.dxyp

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
            -midpoints.isel(level=-1) + egg4.z_surf / 9.80665,
        ],
        dim="level",
    )

    egg4["airdensity"] = rho
    egg4.airdensity.attrs = {
        "CDI_grid_type": "unstructured",
        "long_name": "Dry Air Density",
        "standard_name": "dry_air_density",
        "units": "kg m^-3",
    }
    egg4["volume"] = V
    egg4.volume.attrs = {
        "CDI_grid_type": "unstructured",
        "long_name": "Grid Cell Volume",
        "standard_name": "grid_cell_volume",
        "units": "m^3",
    }
    egg4["cell_area"] = dxyp

    co2density = egg4.co2 * egg4.airdensity
    egg4["co2density"] = co2density
    egg4.co2density.attrs = {
        "CDI_grid_type": "unstructured",
        "long_name": "Carbon dioxide density",
        "standard_name": "co2_density",
        "units": "kg m^-3",
    }

    ch4density = egg4.ch4 * egg4.airdensity
    egg4["ch4density"] = ch4density
    egg4.ch4density.attrs = {
        "CDI_grid_type": "unstructured",
        "long_name": "Methane density",
        "standard_name": "ch4_density",
        "units": "kg m^-3",
    }

    train_dir = data_dir / "CAMS_EGG4" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    val_dir = data_dir / "CAMS_EGG4" / "val"
    val_dir.mkdir(parents=True, exist_ok=True)

    test_dir = data_dir / "CAMS_EGG4" / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    compressor = Blosc(cname="lz4", clevel=5)
    enc = {x: {"compressor": compressor} for x in egg4.data_vars}
    print("To Zarr")
    with ProgressBar():
        egg4.sel(time=slice("2018-01-01", "2018-12-31")).chunk(
            {"time": 10, "level": -1, "lat": -1, "lon": -1}
        ).to_zarr(val_dir / "egg4_latlon1.zarr", encoding=enc)

    with ProgressBar():
        egg4.sel(time=slice("2003-01-01", "2017-12-31")).chunk(
            {"time": 10, "level": -1, "lat": -1, "lon": -1}
        ).to_zarr(train_dir / "egg4_latlon1.zarr", encoding=enc)

    with ProgressBar():
        egg4.sel(time=slice("2019-01-01", "2020-12-31")).chunk(
            {"time": 10, "level": -1, "lat": -1, "lon": -1}
        ).to_zarr(test_dir / "egg4_latlon1.zarr", encoding=enc)


def data_to_zarr(data_dir):
    print("Atmosphere data to Zarr")
    data_dir = Path(data_dir)
    egg4a = xr.open_mfdataset(
        (data_dir / "CAMS_EGG4" / "icon_multi_level").glob("*/*.nc")
    ).reset_encoding()
    egg4s = xr.open_mfdataset(
        (data_dir / "CAMS_EGG4" / "icon_single_level").glob("*/*.nc")
    ).reset_encoding()

    egg4 = xr.merge([egg4a, egg4s.rename({"z": "z_surf"})])
    # egg4.coords['longitude'] = (egg4.coords['longitude'] + 180) % 360 - 180
    # egg4 = egg4.rename({"longitude": "lon", "latitude": "lat"})
    # egg4 = egg4.chunk({"time": 1, "lat": -1, "lon": -1, "level": -1})

    gridpath = data_dir / "icon_grid" / "R02B04_G.nc"
    grid = xr.open_dataset(gridpath)

    gph = ds.z / 9.80665
    T = ds.t
    RH = ds.r
    dxyp = grid.cell_area

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
            -midpoints.isel(level=-1) + ds.z_surf / 9.80665,
        ],
        dim="level",
    )

    ds["airdensity"] = rho
    ds.airdensity.attrs = {
        "CDI_grid_type": "unstructured",
        "long_name": "Dry Air Density",
        "standard_name": "dry_air_density",
        "units": "kg m^-3",
    }
    ds["volume"] = V
    ds.volume.attrs = {
        "CDI_grid_type": "unstructured",
        "long_name": "Grid Cell Volume",
        "standard_name": "grid_cell_volume",
        "units": "m^3",
    }
    ds["cell_area"] = dxyp

    co2density = ds.co2 * ds.airdensity
    ds["co2density"] = co2density
    ds.co2density.attrs = {
        "CDI_grid_type": "unstructured",
        "long_name": "Carbon dioxide density",
        "standard_name": "co2_density",
        "units": "kg m^-3",
    }

    ch4density = ds.ch4 * ds.airdensity
    ds["ch4density"] = ch4density
    ds.ch4density.attrs = {
        "CDI_grid_type": "unstructured",
        "long_name": "Methane density",
        "standard_name": "ch4_density",
        "units": "kg m^-3",
    }

    train_dir = data_dir / "CAMS_EGG4" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    val_dir = data_dir / "CAMS_EGG4" / "val"
    val_dir.mkdir(parents=True, exist_ok=True)

    test_dir = data_dir / "CAMS_EGG4" / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    compressor = Blosc(cname="lz4", clevel=5)
    enc = {x: {"compressor": compressor} for x in egg4.data_vars}

    with ProgressBar():
        egg4.sel(time=slice("2018-01-01", "2018-12-31")).chunk(
            {"time": 10, "level": -1, "cell": -1}
        ).to_zarr(val_dir / "egg4_icon.zarr", encoding=enc)

    with ProgressBar():
        egg4.sel(time=slice("2003-01-01", "2017-12-31")).chunk(
            {"time": 10, "level": -1, "cell": -1}
        ).to_zarr(train_dir / "egg4_icon.zarr", encoding=enc)

    with ProgressBar():
        egg4.sel(time=slice("2019-01-01", "2020-12-31")).chunk(
            {"time": 10, "level": -1, "cell": -1}
        ).to_zarr(test_dir / "egg4_icon.zarr", encoding=enc)

    # print("Surface data to Zarr")
    # egg4s = xr.open_mfdataset((data_dir/"single_level").glob("*/*.nc"))
    # egg4s.coords['longitude'] = (egg4s.coords['longitude'] + 180) % 360 - 180
    # egg4s = egg4s.rename({"longitude": "lon", "latitude": "lat"})
    # egg4s = egg4s.chunk({"time": 1, "lat": -1, "lon": -1})

    # with ProgressBar():
    #     egg4s.sel(time=slice("2003-01-01","2017-12-31")).to_zarr(train_dir/"surface.zarr")

    # with ProgressBar():
    #     egg4s.sel(time=slice("2018-01-01","2018-12-31")).to_zarr(val_dir/"surface.zarr")

    # with ProgressBar():
    #     egg4s.sel(time=slice("2019-01-01","2020-12-31")).to_zarr(test_dir/"surface.zarr")


def fix_one_chunk(zarrgroup, egg4, var):
    for i in range(0, len(egg4.time), 100):
        print(var, i)
        arr = egg4[var].isel(time=slice(i, i + 100)).values
        if (~np.isfinite(arr)).sum() > 0:
            print(var, i, "not finite")
        zarrgroup[var][i : i + 100, ...] = arr


def fix_data_to_zarr(data_dir):
    data_dir = Path(data_dir)
    egg4a = xr.open_mfdataset(
        (data_dir / "CAMS_EGG4" / "icon_multi_level").glob("*/*.nc")
    )
    # egg4s = xr.open_mfdataset((data_dir/"CAMS_EGG4"/"icon_single_level").glob("*/*.nc"))

    # egg4 = xr.merge([egg4a, egg4s.rename({"z":"z_surf"})])

    train_dir = data_dir / "CAMS_EGG4" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    val_dir = data_dir / "CAMS_EGG4" / "val"
    val_dir.mkdir(parents=True, exist_ok=True)

    test_dir = data_dir / "CAMS_EGG4" / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    data_vars = ["z"]  # list(egg4.data_vars)
    print("Fix Train")
    zarrpath = train_dir / "egg4_icon.zarr"
    zarrgroup = zarr.open_group(str(zarrpath))
    egg4train = egg4a.sel(time=slice("2003-01-01", "2017-12-31"))

    thread_map(
        fix_one_chunk,
        len(data_vars) * [zarrgroup],
        len(data_vars) * [egg4train],
        data_vars,
        max_workers=128,
    )

    # for var in tqdm(list(egg4.data_vars), position=0, leave=True, desc="Var"):

    #     for i in tqdm(range(0, len(egg4.time.sel(time=slice("2003-01-01","2017-12-31"))), 100), position=1, leave=False, desc="Time"):
    #         zarrgroup[var][i:i+100,...] = egg4[var].sel(time=slice("2003-01-01","2017-12-31")).isel(time = slice(i,i+100)).values

    print("Fix Val")
    zarrpath = val_dir / "egg4_icon.zarr"
    zarrgroup = zarr.open_group(str(zarrpath))
    egg4val = egg4a.sel(time=slice("2018-01-01", "2018-12-31"))

    thread_map(
        fix_one_chunk,
        len(data_vars) * [zarrgroup],
        len(data_vars) * [egg4val],
        data_vars,
        max_workers=128,
    )

    # for var in tqdm(list(egg4.data_vars), position=0, leave=True, desc="Var"):

    #     for i in tqdm(range(0, len(egg4.time.sel(time=slice("2018-01-01","2018-12-31"))), 100), position=1, leave=False, desc="Time"):
    #         zarrgroup[var][i:i+100,...] = egg4[var].sel(time=slice("2018-01-01","2018-12-31")).isel(time = slice(i,i+100)).values

    print("Fix Test")
    zarrpath = test_dir / "egg4_icon.zarr"
    zarrgroup = zarr.open_group(str(zarrpath))
    egg4test = egg4a.sel(time=slice("2019-01-01", "2020-12-31"))

    thread_map(
        fix_one_chunk,
        len(data_vars) * [zarrgroup],
        len(data_vars) * [egg4test],
        data_vars,
        max_workers=128,
    )

    # for var in tqdm(list(egg4.data_vars), position=0, leave=True, desc="Var"):

    #     for i in tqdm(range(0, len(egg4.time.sel(time=slice("2019-01-01","2020-12-31"))), 100), position=1, leave=False, desc="Time"):
    #         zarrgroup[var][i:i+100,...] = egg4[var].sel(time=slice("2019-01-01","2020-12-31")).isel(time = slice(i,i+100)).values


def compute_weights(save_dir):
    weight_path = Path(save_dir) / "CAMS_EGG4" / "weights"
    weight_path.mkdir(exist_ok=True, parents=True)

    ds = xr.open_zarr(Path(save_dir) / "CAMS_EGG4" / "train" / "egg4_icon.zarr")
    co2diff_mean = ds.co2.diff("time").mean(["cell", "time"])
    with ProgressBar():
        co2diff_mean.to_dataset(name="co2mix").to_netcdf(
            weight_path / "co2diff_mean.nc"
        )
    co2diff_std = ds.co2.diff("time").std(["cell", "time"])
    with ProgressBar():
        co2diff_std.to_dataset(name="co2mix").to_netcdf(weight_path / "co2diff_std.nc")
    co2diff_min = ds.co2.diff("time").min(["cell", "time"])
    with ProgressBar():
        co2diff_min.to_dataset(name="co2mix").to_netcdf(weight_path / "co2diff_min.nc")
    co2diff_max = ds.co2.diff("time").max(["cell", "time"])
    with ProgressBar():
        co2diff_max.to_dataset(name="co2mix").to_netcdf(weight_path / "co2diff_max.nc")

    xr.DataArray(data=np.ones(25) / 25, coords={"level": ds.level.values}).to_netcdf(
        weight_path / "equal_height_weights.nc"
    )


def stats_dataset(save_dir):
    save_dir = Path(save_dir)
    train_dir = save_dir / "CAMS_EGG4" / "train"
    val_dir = save_dir / "CAMS_EGG4" / "val"
    test_dir = save_dir / "CAMS_EGG4" / "test"

    ds = xr.open_zarr(train_dir / "egg4_latlon1.zarr")

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

    for molecule in ["ch4", "co2"]:
        density_var = f"{molecule}density"
        density_delta = ds[density_var].diff("time")
        with ProgressBar():
            delta_stats = (
                xr.concat(
                    [
                        density_delta.min(["time", "lat", "lon"]),
                        density_delta.mean(["time", "lat", "lon"]),
                        density_delta.max(["time", "lat", "lon"]),
                        density_delta.std(["time", "lat", "lon"]),
                    ],
                    "stats",
                )
                .assign_coords({"stats": ["min", "mean", "max", "std"]})
                .compute()
            )
        ds_stats[f"{molecule}density_delta"] = delta_stats

        massmix = density_to_massmix(
            ds[density_var], ds.airdensity, ppm=True, eps=1e-12
        )
        massmix_delta = massmix.diff("time")

        with ProgressBar():
            massmix_stats = (
                xr.concat(
                    [
                        massmix.min(),
                        massmix.mean(),
                        massmix.max(),
                        massmix.std(),
                    ],
                    "stats",
                )
                .assign_coords({"stats": ["min", "mean", "max", "std"]})
                .compute()
            )
        ds_stats[f"{molecule}massmix"] = massmix_stats
        ds_stats[f"{molecule}massmix_next"] = massmix_stats

        ds_stats[f"{molecule}molemix"] = massmix_to_molemix(
            massmix_stats, M=44.009e-3 if molecule == "co2" else 16.043e-3
        )
        ds_stats[f"{molecule}molemix_next"] = massmix_to_molemix(
            massmix_stats, M=44.009e-3 if molecule == "co2" else 16.043e-3
        )

        with ProgressBar():
            massmixdelta_stats = (
                xr.concat(
                    [
                        massmix_delta.min(["time", "lat", "lon"]),
                        massmix_delta.mean(["time", "lat", "lon"]),
                        massmix_delta.max(["time", "lat", "lon"]),
                        massmix_delta.std(["time", "lat", "lon"]),
                    ],
                    "stats",
                )
                .assign_coords({"stats": ["min", "mean", "max", "std"]})
                .compute()
            )
        ds_stats[f"{molecule}massmix_delta"] = massmixdelta_stats
        ds_stats[f"{molecule}molemix_delta"] = massmix_to_molemix(
            massmixdelta_stats, M=44.009e-3 if molecule == "co2" else 16.043e-3
        )

    for var in ds_stats.data_vars:
        ds_stats[f"{var}_next"] = ds_stats[var]

    # ds_stats = xr.open_dataset(train_dir / "egg4_stats_tmp.nc")

    ds_delta = ds.diff("time")
    ds_min = ds_delta.min(["time", "lat", "lon"]).to_array("var")
    ds_mean = ds_delta.mean(["time", "lat", "lon"]).to_array("var")
    ds_max = ds_delta.max(["time", "lat", "lon"]).to_array("var")
    ds_std = ds_delta.std(["time", "lat", "lon"]).to_array("var")
    with ProgressBar():
        ds_delta_stats = (
            xr.concat([ds_min, ds_mean, ds_max, ds_std], "stats")
            .assign_coords({"stats": ["min", "mean", "max", "std"]})
            .to_dataset("var")
            .compute()
        ).rename({v: f"{v}_delta" for v in ds.data_vars})
    ds_delta_stats.to_netcdf(train_dir / f"egg4_delta_stats.nc")
    ds_delta_stats.to_netcdf(val_dir / f"egg4_delta_stats.nc")
    ds_delta_stats.to_netcdf(test_dir / f"egg4_delta_stats.nc")

    ds_stats = ds_delta_stats.merge(ds_stats, compat="override")

    for v in ds.data_vars:
        if "level" not in ds[v].dims:
            ds_stats[f"{v}_delta"] = ds_stats[f"{v}_delta"].isel(level=0)
    ds_stats = ds_stats.drop_vars(
        [
            k
            for k in ds_stats.data_vars
            if (k.endswith("_delta_next") or k.endswith("_next_next"))
        ]
    )

    ds_stats.to_netcdf(train_dir / f"egg4_stats.nc")
    ds_stats.to_netcdf(val_dir / f"egg4_stats.nc")
    ds_stats.to_netcdf(test_dir / f"egg4_stats.nc")


if __name__ == "__main__":
    # download_data("/Net/Groups/BGI/work_2/CAMS_EGG4")

    # remap_to_icon("/Net/Groups/BGI/people/vbenson/graph_tm/data")
    # latlon_to_zarr("/Net/Groups/BGI/people/vbenson/graph_tm/data")
    # data_to_zarr("/Net/Groups/BGI/people/vbenson/graph_tm/data")
    # fix_data_to_zarr("/Net/Groups/BGI/people/vbenson/graph_tm/data")
    # compute_weights("/Net/Groups/BGI/people/vbenson/graph_tm/data")
    stats_dataset("/Net/Groups/BGI/people/vbenson/graph_tm/data")
