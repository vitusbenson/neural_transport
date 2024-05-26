EGG4_METEO3D_VARS = ["airdensity", "pv", "q", "r", "t", "u", "v", "w", "z", "volume"]
EGG4_METEO2D_VARS = [
    "asn",
    "blh",
    "cape",
    # "cin", # faulty !
    "cp",
    "d2m",
    "e",
    "fal",
    "lsp",
    "msl",
    "p3020",
    "pev",
    "sd",
    "siconc",
    "skt",
    "slhf",
    "src",
    "sshf",
    "ssr",
    "ssrc",
    "ssrd",
    "ssrdc",
    "sst",
    "str",
    "strc",
    "strd",
    "strdc",
    "sund",
    "t2m",
    "tcc",
    "tciw",
    "tclw",
    "tcw",
    "tcwv",
    "tisr",
    "tp",
    "tsr",
    "tsrc",
    "ttr",
    "ttrc",
    "u10",
    "uvb",
    "v10",
    # "z_surf", # not changing
]
EGG4_CARBON3D_VARS = ["ch4density", "co2density"]
EGG4_CARBON2D_VARS = ["ch4f", "ch4fire", "co2apf", "co2fire", "co2of", "fco2nee"]

CARBOSCOPE_CARBON3D_VARS = ["co2density"]
CARBOSCOPE_METEO3D_VARS = [
    "airdensity",
    "omeg",
    "q",
    "r",
    "t",
    "u",
    "v",
    "gp",
    "volume",
]
CARBOSCOPE_CARBON2D_VARS = ["co2flux_land", "co2flux_ocean", "co2flux_subt"]
CARBOSCOPE_METEO2D_VARS = []

CARBONTRACKER_CARBON3D_VARS = ["co2density"]
CARBONTRACKER_METEO3D_VARS = [
    "air_mass",
    "q",
    "t",
    "u",
    "v",
    "airdensity",
    "volume",
    "gp",
    "p",
]
CARBONTRACKER_CARBON2D_VARS = [
    "co2flux_land",
    "co2flux_ocean",
    "co2flux_subt",
    # "bio_flux_opt",
    # "ocn_flux_opt",
    # "fossil_flux_imp",
    # "fire_flux_imp",
]
CARBONTRACKER_METEO2D_VARS = ["blh", "orography"]
