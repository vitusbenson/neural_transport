"""Load IPCC colormaps (continuous and discrete) as ListedColormap objects."""

from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap


def load_ipcc_cmaps(file_paths: list[str], discrete: bool = False) -> list[ListedColormap]:
    """
    Load IPCC colormaps (continuous or discrete) and return a list of ListedColormap objects.

    Parameters
    ----------
    file_paths : list of str
        Paths to IPCC colormap txt files.
    discrete : bool
        If True, treat the files as discrete colormaps with headers.

    Returns
    -------
    list of ListedColormap
    """
    cmaps = []

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")

        if not discrete:
            # Continuous colormap: just RGB rows
            rgb_data = np.loadtxt(path, dtype=int) / 255.0
            cmaps.append(ListedColormap(rgb_data, name=path.stem))
        else:
            # Discrete colormap: headers + RGB rows
            with Path.open(path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]

            current_name = None
            current_rgb = []
            for line in lines:
                # Header line starts with 'chem_' or any non-numeric line
                if not line[0].isdigit():
                    # save previous cmap
                    if current_name and current_rgb:
                        rgb_array = np.array(current_rgb, dtype=int) / 255.0
                        cmaps.append(ListedColormap(rgb_array, name=current_name))
                    current_name = line
                    current_rgb = []
                else:
                    rgb_vals = [int(x) for x in line.split()]
                    current_rgb.append(rgb_vals)
            # last one
            if current_name and current_rgb:
                rgb_array = np.array(current_rgb, dtype=int) / 255.0
                cmaps.append(ListedColormap(rgb_array, name=current_name))

    return cmaps
