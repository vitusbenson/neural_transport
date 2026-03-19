"""GridInfo and InferenceDataLoader for inference/generation data access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from neural_transport.configs import DataConfig
from neural_transport.datasets.grids import LATLON_PROTOTYPE_COORDS, VERTICAL_LAYERS_PROTOTYPE_COORDS


@dataclass
class GridInfo:
    """Grid geometry and coordinate metadata."""

    nlat: int
    nlon: int
    nlev: int
    lat: np.ndarray
    lon: np.ndarray
    levels: np.ndarray

    @classmethod
    def from_config(cls, config: DataConfig) -> GridInfo:
        """Build GridInfo from a DataConfig.

        Raises ValueError for non-latlon grids.
        """
        grid = config.grid
        vlev = config.vertical_levels

        if grid not in LATLON_PROTOTYPE_COORDS:
            raise ValueError(
                f"GridInfo.from_config only supports latlon grids, got non-latlon grid {grid!r}. "
                f"Available: {sorted(LATLON_PROTOTYPE_COORDS)}"
            )

        coords = LATLON_PROTOTYPE_COORDS[grid]
        lat = coords["lat"]
        lon = coords["lon"]

        level_coords = VERTICAL_LAYERS_PROTOTYPE_COORDS[vlev]
        levels = level_coords["level"]

        return cls(
            nlat=len(lat),
            nlon=len(lon),
            nlev=len(levels),
            lat=lat,
            lon=lon,
            levels=levels,
        )

    @property
    def cos_lat_weights_2d(self) -> np.ndarray:
        """Cosine-latitude weights with shape ``(nlat, 1)``, normalized to mean 1."""
        w = np.cos(np.radians(self.lat))[:, None]
        return w / w.mean()

    @property
    def cos_lat_weights_flat(self) -> np.ndarray:
        """Cosine-latitude weights with shape ``(nlat*nlon, 1)``, normalized to mean 1.

        Matches the carbonbench pattern:
        ``np.cos(np.radians(lat))[:, None, None].repeat(nlon, axis=1).reshape(-1, 1)``
        then divide by mean.
        """
        w = np.cos(np.radians(self.lat))[:, None, None].repeat(self.nlon, axis=1).reshape(-1, 1)
        return w / w.mean()

    @property
    def cos_lat_weights(self) -> np.ndarray:
        """Alias for :attr:`cos_lat_weights_flat`."""
        return self.cos_lat_weights_flat


class InferenceDataLoader:
    """Lightweight data loader for inference/generation, decoupled from Lightning.

    Parameters
    ----------
    data_config : DataConfig
        Describes *what* data (dataset name, grid, levels, freq, variables).
    data_path : str | Path
        Where the data lives on disk. Separate from DataConfig because it's
        deployment-dependent (varies per split / machine).
    load_obspack : bool
        Whether to load obspack data when constructing the CarbonDataset.
    """

    def __init__(self, data_config: DataConfig, data_path: str | Path, *, load_obspack: bool = False):
        self._config = data_config
        self._data_path = str(data_path)
        self._load_obspack = load_obspack
        self._grid_info = GridInfo.from_config(data_config)
        self._dataset = None  # lazy

    @property
    def grid_info(self) -> GridInfo:
        return self._grid_info

    def load_dataset(self):
        """Load (or reload) the underlying CarbonDataset. Returns it."""
        from neural_transport.datamodule import CarbonDataset

        self._dataset = CarbonDataset(
            data_path=self._data_path,
            dataset=self._config.dataset,
            grid=self._config.grid,
            vertical_levels=self._config.vertical_levels,
            freq=self._config.freq,
            n_timesteps=1,
            target_vars=self._config.target_vars,
            forcing_vars=self._config.forcing_vars,
            load_obspack=self._load_obspack,
            new_zarr=True,
        )
        return self._dataset

    @property
    def dataset(self):
        """Lazily loaded CarbonDataset."""
        if self._dataset is None:
            self.load_dataset()
        return self._dataset

    def get_batch(self, idx: int, device: str = "cuda") -> dict[str, torch.Tensor]:
        """Return a single sample as a batch (dim 0 added), moved to *device*."""
        sample = self.dataset[idx]
        return {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}

    def get_gt_field(self, idx: int) -> np.ndarray:
        """Return first target variable as numpy array with shape ``[cell, level]``."""
        sample = self.dataset[idx]
        target_var = self._config.target_vars[0]
        field = sample[target_var]
        if isinstance(field, torch.Tensor):
            field = field.numpy()
        # Handle optional time dim: [T, N, C] → [N, C]
        if field.ndim == 3:
            field = field[0]
        return field

    def get_normalization_stats(self) -> dict:
        """Extract ``_offset`` / ``_scale`` keys from a sample."""
        sample = self.dataset[0]
        return {k: v for k, v in sample.items() if k.endswith("_offset") or k.endswith("_scale")}

    def __len__(self) -> int:
        return len(self.dataset)
