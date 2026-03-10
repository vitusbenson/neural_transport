from pathlib import Path

from neural_transport.plots.ipcc_colormaps.colormaps_ipcc import load_ipcc_cmaps

# ---- Colormap configuration ----
cmap_selected = [
    "berlin",
    "berlin_r",
    "managua",
    "BrBG",
    "Greys",
    "binary",
    "bone_r",
    "copper_r",
    "pink",
    "pink_r",
    "cividis_r",
    "cividis",
    "managua_r",
    "crest",
    "mako_r",
    "coolwarm",
]

default_cmap = ["bone_r"]

# IPCC colormap folder
ipcc_folder = Path(__file__).parents[1] / "ipcc_colormaps"

# Load IPCC continuous and discrete colormaps
cmap_ipcc_cont = load_ipcc_cmaps(
    [
        ipcc_folder / "chem_div.txt",
        ipcc_folder / "chem_seq.txt",
    ]
)

cmap_ipcc_disc = load_ipcc_cmaps(
    [
        ipcc_folder / "chem_div_disc.txt",
        ipcc_folder / "chem_seq_disc.txt",
    ],
    discrete=True,
)

cmap_ipcc_list = cmap_ipcc_cont + cmap_ipcc_disc


def get_cmap_list(use_ipcc: bool = False, use_ipcc_one: bool = False, use_selected: bool = False, n_samples: int = 1):
    """Return a list of colormaps according to the selected options."""
    if use_ipcc:
        return cmap_ipcc_list
    if use_ipcc_one:
        return [cmap_ipcc_list[0]] * n_samples
    if use_selected:
        return cmap_selected
    return default_cmap
