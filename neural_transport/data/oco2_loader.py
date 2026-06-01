"""OCO-2 observation loading: ObservationBatch and OCO2DataLoader.

Phase 15: Separates OCO-2 observation handling from generation logic.
Time alignment, window aggregation, mask creation, and AK cleanup
belong in the data layer, not in GenerationPipeline._run_timeseries().
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from neural_transport.configs import DataConfig
from neural_transport.data.inference_loader import InferenceDataLoader


@dataclass
class ObservationBatch:
    """Structured container for OCO-2 observation data.

    Replaces the ad-hoc dict manipulation in ``_run_timeseries()``.
    All tensors have shape ``[B, T, N, C]`` (or ``C_levels`` for AK/profile fields).
    """

    obs_values: Tensor  # [B, T, N, C] XCO2 observations (NaN outside mask)
    obs_mask: Tensor  # [B, T, N, C] bool mask
    averaging_kernel: Tensor | None  # [B, T, N, C_levels] cleaned AK
    xco2_prior: Tensor | None  # [B, T, N, 1]
    co2_profile_prior: Tensor | None  # [B, T, N, C_levels]
    pressure_weights: Tensor | None  # [B, T, N, C_levels]
    obs_mean: Tensor  # normalization offset
    obs_std: Tensor  # normalization scale
    raw_batch: dict[str, Tensor]  # full aggregated OCO-2 batch dict
    pressure_levels: Tensor | None = None  # [B, T, N, C_levels] retrieval native pressures [hPa] (P1)

    def inject_into_batch(
        self,
        batch: dict[str, Tensor],
        *,
        target_var: str,
        forcing_vars: Sequence[str],
    ) -> None:
        """Copy observation fields and forcing variables into a model batch dict.

        Modifies *batch* in-place, setting:
        - ``obs_mask``, ``obs_mask_original``  (from ``self.obs_mask``)
        - ``obs_values``  (from ``self.obs_values``)
        - target_var and each forcing_var  (from ``self.raw_batch``)
        """
        batch["obs_mask"] = self.obs_mask.clone()
        batch["obs_mask_original"] = self.obs_mask.clone()
        batch["obs_values"] = self.obs_values.clone()

        # Copy the target variable and forcing variables from raw OCO-2 batch
        for k in [target_var, *forcing_vars]:
            if k in self.raw_batch:
                batch[k] = self.raw_batch[k]


class OCO2DataLoader(InferenceDataLoader):
    """Extends :class:`InferenceDataLoader` with OCO-2 observation handling.

    Encapsulates:
    - Time alignment between GT and OCO-2 datasets
    - Window aggregation with ``nanmean`` for sparse observations
    - Mask creation (real sparsity or test patterns)
    - Averaging-kernel cleanup
    """

    def __init__(
        self,
        data_config: DataConfig,
        data_path: str | Path,
        *,
        target_var: str = "xco2_2019_scale",
        forcing_vars: tuple[str, ...] = (
            "xco2_averaging_kernel",
            "xco2_apriori",
            "co2_profile_apriori",
        ),
        load_obspack: bool = False,
    ):
        super().__init__(data_config, data_path, load_obspack=load_obspack)
        self.target_var = target_var
        self.forcing_vars = forcing_vars

    # ── Time alignment ────────────────────────────────────────────────

    @staticmethod
    def align_time(time: np.ndarray, time_gen: np.ndarray) -> int:
        """Find index offset to align two time axes.

        Returns
        -------
        offset : int
            Such that ``time_gen[i + offset] ≈ time[i]``.

        Raises
        ------
        ValueError
            If the two time axes have no overlap.
        """
        if time[-1] < time_gen[0]:
            raise ValueError("No overlap: time from trained dataset is before masking dataset.")
        elif time[0] > time_gen[-1]:
            raise ValueError("No overlap: time from trained dataset is after masking dataset.")
        elif time[0] == time_gen[0]:
            return 0
        elif time[0] > time_gen[0]:
            offset = (time_gen == time[0]).argmax().item()
            return offset
        else:  # time[0] < time_gen[0]
            offset = -(time == time_gen[0]).argmax().item()
            return offset

    def compute_offset(self, gt_dataset) -> int:
        """Compute time offset between this loader and a GT dataset.

        Parameters
        ----------
        gt_dataset
            Any object with ``.ds.time.values`` (e.g. a ``CarbonDataset``
            or another ``InferenceDataLoader``).
        """
        if hasattr(gt_dataset, "dataset"):
            # InferenceDataLoader — unwrap
            gt_ds = gt_dataset.dataset
        else:
            gt_ds = gt_dataset
        return self.align_time(gt_ds.ds.time.values, self.dataset.ds.time.values)

    # ── Window batch loading ──────────────────────────────────────────

    def get_window_batch(
        self,
        start_idx: int,
        window_steps: int = 1,
        device: str = "cuda",
        agg: str = "mean",
    ) -> dict[str, Tensor]:
        """Load and aggregate ``window_steps`` consecutive samples.

        Parameters
        ----------
        start_idx : int
            First index into the dataset.
        window_steps : int
            Number of consecutive timesteps to aggregate.
        device : str
            Target device for tensors.
        agg : str
            ``'mean'`` for GT data, ``'nanmean'`` for sparse OCO-2 data.

        Returns
        -------
        dict[str, Tensor]
            Aggregated batch with batch dim 0 added.
        """
        end_idx = min(start_idx + window_steps, len(self))
        batch_list = []
        for idx in range(start_idx, end_idx):
            sample = self.dataset[idx]
            batch_list.append({k: v.unsqueeze(0).to(device) if isinstance(v, Tensor) else v for k, v in sample.items()})

        batch: dict[str, Tensor] = {}
        for k in batch_list[0]:
            vals = [b[k] for b in batch_list if isinstance(b[k], Tensor)]
            if not vals:
                batch[k] = batch_list[0][k]
                continue
            stacked = torch.stack(vals, dim=0)
            if agg == "nanmean":
                batch[k] = torch.nanmean(stacked, dim=0)
            else:
                batch[k] = stacked.mean(dim=0)
        return batch

    # ── Full observation pipeline ─────────────────────────────────────

    def get_observations(
        self,
        time_idx: int,
        *,
        offset: int = 0,
        window_steps: int = 1,
        device: str = "cuda",
        mask_pattern: str | None = None,
        nlat: int | None = None,
        nlon: int | None = None,
    ) -> ObservationBatch:
        """Full observation pipeline: aggregate window, create mask, clean AK.

        Parameters
        ----------
        time_idx : int
            Logical timestep index (in the GT timeline).
        offset : int
            Offset from :meth:`compute_offset` — added to ``time_idx``
            to index into this loader's dataset.
        window_steps : int
            Number of timesteps to aggregate.
        device : str
            Target device.
        mask_pattern : str or None
            ``None`` → real sparsity mask via :func:`create_oco2_mask`.
            Otherwise a test pattern name via :func:`create_oco2_mask_test`.
        nlat, nlon : int or None
            Grid dimensions for test masks. Defaults to ``grid_info``.

        Returns
        -------
        ObservationBatch
        """
        from neural_transport.inference.masking import create_oco2_mask, create_oco2_mask_test

        if nlat is None:
            nlat = self.grid_info.nlat
        if nlon is None:
            nlon = self.grid_info.nlon

        # Aggregate the OCO-2 window with nanmean (sparse data)
        start = time_idx + offset if offset > 0 else time_idx
        batch_gen = self.get_window_batch(start, window_steps=window_steps, device=device, agg="nanmean")

        # Create mask
        if mask_pattern is None:
            obs_mask, obs_values = create_oco2_mask(batch_gen, target_var=self.target_var)
        else:
            obs_mask, obs_values = create_oco2_mask_test(
                batch_gen,
                target_var=self.target_var,
                mask_pattern=mask_pattern,
                nlat=nlat,
                nlon=nlon,
            )

        # Extract normalization stats
        offset_key = f"{self.target_var}_offset"
        scale_key = f"{self.target_var}_scale"
        obs_mean = batch_gen.get(offset_key, torch.tensor(0.0, device=device))
        obs_std = batch_gen.get(scale_key, torch.tensor(1.0, device=device))

        # Extract optional AK / prior fields
        ak = batch_gen.get("xco2_averaging_kernel")
        xco2_prior = batch_gen.get("xco2_apriori")
        co2_profile_prior = batch_gen.get("co2_profile_apriori")
        pressure_weights = batch_gen.get("pressure_weight")
        # P1: native retrieval pressure levels [hPa], present only in the
        # native-level (l20) obs product; enables the interpolate-then-apply op.
        pressure_levels = batch_gen.get("pressure_levels")

        return ObservationBatch(
            obs_values=obs_values,
            obs_mask=obs_mask,
            averaging_kernel=ak,
            xco2_prior=xco2_prior,
            co2_profile_prior=co2_profile_prior,
            pressure_weights=pressure_weights,
            obs_mean=obs_mean,
            obs_std=obs_std,
            raw_batch=batch_gen,
            pressure_levels=pressure_levels,
        )
