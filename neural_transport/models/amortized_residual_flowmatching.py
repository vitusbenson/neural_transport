"""Amortized conditional residual flow matching (P3.8).

Extends :class:`ResidualFlowMatching` so the velocity UNet *also* conditions on
the observations directly, learning the posterior ``p(x_next | x_t, f_det, y)``
in a single generation pass — no test-time guidance / optimisation loop.

Two extra input channels are appended to the conditioning group:
    [ ... , f_det_pred (10), obs_value_norm (1), obs_mask (1), time (1) ]
so ``in_chans`` grows by 2 vs the unconditional residual-FM (51 -> 53 at l10).

Training: each sample draws a random observation mask (varied coverage) and
synthesises the column observation of the *target* state through a fixed nominal
pressure-weight column operator ``H_L`` (uniform AK), normalised like the state.
The FM then learns to generate the residual consistent with that obs.

Inference: the obs channels are built from ``batch["obs_mask"]`` /
``batch["obs_values"]`` (already normalised), and generation is a single
unconditional ODE solve — the obs are baked into the conditioning.
"""

from __future__ import annotations

import numpy as np
import torch

from neural_transport.models.flowmatching import compute_ot_coupling
from neural_transport.models.residual_flowmatching import ResidualFlowMatching

# Nominal l10 layer-edge pressures (hPa) -> column pressure weights H_L (sum=1).
_P_BOT_L10 = np.array([966.8, 959.3, 949.5, 926.8, 900.8, 806.9, 619.1, 430.3, 240.4, 73.1])
_P_TOP_L10 = np.array([959.3, 949.5, 926.8, 900.8, 806.9, 619.1, 430.3, 240.4, 73.1, 0.0])
_H_L10 = (_P_BOT_L10 - _P_TOP_L10) / _P_BOT_L10[0]


class AmortizedResidualFlowMatching(ResidualFlowMatching):
    """Residual-FM whose velocity UNet conditions on observations directly."""

    def init_model(
        self,
        det_ckpt: str,
        sigma_res_path: str,
        det_freeze: bool = True,
        obs_coverage_range: tuple[float, float] = (0.0, 0.30),
        column_weights: list | None = None,
        **flowmatching_kwargs,
    ):
        super().init_model(det_ckpt, sigma_res_path, det_freeze=det_freeze, **flowmatching_kwargs)
        self.obs_coverage_range = tuple(obs_coverage_range)
        hl = np.asarray(column_weights, dtype="float32") if column_weights is not None else _H_L10.astype("float32")
        self.register_buffer("column_weights", torch.from_numpy(hl))  # [C]
        self.n_obs_channels = 2  # obs_value_norm, obs_mask

    # ------------------------------------------------------------------ helpers
    def _column_of(self, x_phys):
        """Nominal column XCO2 of a physical profile [B,N,C] -> [B,N]."""
        w = self.column_weights.to(x_phys.device, x_phys.dtype)
        return (x_phys * w).sum(dim=-1)

    def _obs_channels_from_grid(self, y_norm_grid, mask_grid):
        """Stack [obs_value_norm*mask, mask] -> [B, 2, nlat, nlon]."""
        return torch.cat([y_norm_grid * mask_grid, mask_grid], dim=1)

    def _sample_train_obs_channels(self, x_next_phys, batch):
        """Synthesise a randomly-masked column obs of the target -> [B,2,H,W]."""
        target_var = self.target_vars[0]
        B, N, _ = x_next_phys.shape
        y = self._column_of(x_next_phys)  # [B, N] physical column
        offset = batch[f"{target_var}_offset"].view(B, 1)
        scale = batch[f"{target_var}_scale"].view(B, 1)
        y_norm = (y - offset) / scale  # [B, N]
        # Per-sample coverage drawn uniformly in the configured range.
        lo, hi = self.obs_coverage_range
        cov = lo + (hi - lo) * torch.rand(B, 1, device=x_next_phys.device)
        mask = (torch.rand(B, N, device=x_next_phys.device) < cov).to(y_norm.dtype)  # [B,N]
        y_grid = y_norm.reshape(B, self.nlat, self.nlon, 1).permute(0, 3, 1, 2)
        m_grid = mask.reshape(B, self.nlat, self.nlon, 1).permute(0, 3, 1, 2)
        return self._obs_channels_from_grid(y_grid, m_grid)

    def _infer_obs_channels(self, batch, B):
        """Build obs channels from batch obs_mask / obs_values (normalised)."""
        dev = self.column_weights.device
        if "obs_mask" not in batch or "obs_values" not in batch:
            return torch.zeros(B, self.n_obs_channels, self.nlat, self.nlon, device=dev)
        m = batch["obs_mask"].reshape(B, self.nlat, self.nlon, 1).permute(0, 3, 1, 2).to(dev)
        y = batch["obs_values"].reshape(B, self.nlat, self.nlon, 1).permute(0, 3, 1, 2).to(dev)
        y = torch.nan_to_num(y, nan=0.0)
        return self._obs_channels_from_grid(y.float(), m.float())

    # ------------------------------------------------------------------ training
    def training_forward(self, batch):
        target_var = self.target_vars[0]

        det_pred_phys, det_pred_grid = self._det_predict(batch)

        x_next_phys = batch[f"{target_var}_next"]
        sigma_phys = self._sigma_res_for_phys(x_next_phys)
        residual_target = (x_next_phys - det_pred_phys) / sigma_phys
        B, N, C = residual_target.shape
        x_1 = residual_target.reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)

        x_in = self.preprocess_inputs(batch)
        x_in = torch.cat([x_in, det_pred_grid.to(x_in.dtype)], dim=1)
        # Observation conditioning group (random mask + nominal column of target).
        obs_ch = self._sample_train_obs_channels(x_next_phys, batch)
        x_in = torch.cat([x_in, obs_ch.to(x_in.dtype)], dim=1)

        x_0 = torch.randn_like(x_1)
        if self.use_ot_coupling:
            x_0 = compute_ot_coupling(x_0, x_1)

        t = self._sample_time(B, x_in.device)
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        x_t = path_sample.x_t

        nlev_target = self.nlev * len(self.target_vars)
        x_in[:, :nlev_target, :, :] = x_t

        t_expanded = path_sample.t.view(-1, 1, 1, 1).expand(B, 1, self.nlat, self.nlon)
        x_in = torch.cat([x_in, t_expanded], dim=1)

        x_out = self.submodel.model(x_in)

        time_weight = self._compute_time_loss_weight(t)
        time_weight = time_weight if not isinstance(time_weight, float) else None
        return x_out, path_sample.dx_t, time_weight

    # ------------------------------------------------------------------ inference
    def forward(self, batch, *, mode=None):
        if not hasattr(self, "generate_kwargs"):
            self.generate_kwargs = {}
        _mode = mode
        if _mode is None:
            if self.training:
                _mode = "train"
            elif self.generating:
                _mode = "generate"
        if _mode != "generate":
            return super().forward(batch, mode=mode)

        target_var = self.target_vars[0]
        det_pred_phys, det_pred_grid = self._det_predict(batch)

        x_in = self.preprocess_inputs(batch)
        nlev_target = self.nlev * len(self.target_vars)
        all_levels = x_in[:, :nlev_target, :, :]
        obs_ch = self._infer_obs_channels(batch, x_in.shape[0])
        x_in_with_cond = torch.cat([x_in, det_pred_grid.to(x_in.dtype), obs_ch.to(x_in.dtype)], dim=1)

        if "noise" in batch:
            noise = batch["noise"]
            B_, N_, C_ = noise.shape
            x_init = noise.reshape(B_, self.nlat, self.nlon, C_).permute(0, 3, 1, 2)
        else:
            x_init = torch.randn_like(all_levels)
            noise_scale = float(self.generate_kwargs.get("noise_scale", 1.0))
            if noise_scale != 1.0:
                x_init = x_init * noise_scale

        # Amortized conditioning -> pure unconditional ODE solve (no sampler/guidance).
        trajectory = self.inference_forward(
            x_in_with_cond, x_init, masking_config={}, generate_kwargs=self.generate_kwargs
        )

        if self.return_intermediates:
            res_final_grid = trajectory[-1, ...]
        else:
            res_final_grid = trajectory

        B_, C_, H, W = res_final_grid.shape
        res_final = res_final_grid.permute(0, 2, 3, 1).reshape(B_, H * W, C_)
        sigma_phys = self._sigma_res_for_phys(res_final)
        x_next_phys = det_pred_phys + sigma_phys * res_final

        sol = {target_var: x_next_phys}
        if self.return_intermediates:
            T = trajectory.shape[0]
            traj = trajectory.permute(0, 1, 3, 4, 2).reshape(T, B_, H * W, C_)
            sol["trajectory"] = traj
        return sol
