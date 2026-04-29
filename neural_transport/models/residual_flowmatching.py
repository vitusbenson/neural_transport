"""Residual flow matching wrapper (ArchesWeatherGen-style).

Wraps a frozen deterministic backbone f_det (Phase 25c v4 phase 1b) and trains
an FM head g_phi to predict the *residual*

    r = (x_next - f_det(x)) / sigma_res

conditioned on [x_t (residual at FM time t), x_data (current CO2 + winds),
f_det(x), time]. At AR inference, x_next = f_det(x) + sigma_res * g_phi(noise).

The residual is normalized to ~unit variance per (variable, level) by sigma_res
which is precomputed once over the training split (see scripts/compute_sigma_res.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from neural_transport.models.flowmatching import FlowMatching, compute_ot_coupling


class ResidualFlowMatching(FlowMatching):
    """FM head modelling the residual on top of a frozen deterministic backbone.

    Channel layout fed to the velocity UNet (training and inference):
        [x_t (target, residual space), x_data (target_var current),
         forcings..., f_det_pred (normalized), time]

    Args (in addition to FlowMatching's):
        det_ckpt: path to a pytorch-lightning .ckpt with a deterministic
            ``NeuralTransport`` module whose ``self.model`` is the UNet.
        sigma_res_path: path to a ``.npy`` array of shape ``(nlev,)`` (or scalar).
    """

    def init_model(
        self,
        det_ckpt: str,
        sigma_res_path: str,
        det_freeze: bool = True,
        **flowmatching_kwargs,
    ):
        super().init_model(**flowmatching_kwargs)

        # Load deterministic backbone -- a NeuralTransport lit-module whose
        # `.model` attribute is the UNet (or other RegularGridModel).
        from neural_transport.litmodule import NeuralTransport

        ckpt_path = str(Path(det_ckpt).expanduser().resolve())
        det_lit = NeuralTransport.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)
        self.f_det = det_lit.model
        if det_freeze:
            for p in self.f_det.parameters():
                p.requires_grad_(False)
        self.f_det.eval()
        self._det_freeze = det_freeze

        # Load sigma_res. Shape: (C,) per level, or scalar.
        sigma = np.load(str(Path(sigma_res_path).expanduser().resolve())).astype("float32")
        sigma = np.asarray(sigma).reshape(-1)  # always (C,)
        self.register_buffer("sigma_res", torch.from_numpy(sigma))  # [C]

    # Keep f_det frozen even when Lightning calls .train() after validation.
    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_det_freeze", True):
            self.f_det.eval()
        return self

    # ------------------------------------------------------------------ helpers
    def _det_predict(self, batch):
        """Run f_det in inference mode (with `_next` placeholder = current state).

        When ``self.enable_det_grad`` is True (set by window-D-Flow sampler),
        gradient flows from f_det's output back to its inputs — needed to
        chain gradients across AR steps when the input state is a function
        of upstream optimizable noise. f_det parameters stay frozen either way
        (set in init_model via requires_grad_(False)).

        Returns (det_pred_phys [B, N, C], det_pred_norm_grid [B, C, Nlat, Nlon]).
        """
        target_var = self.target_vars[0]
        batch_for_det = dict(batch)
        if f"{target_var}_next" in batch_for_det:
            placeholder = batch_for_det[target_var]
            # Detach only when grad is disabled so that we don't anchor the
            # placeholder to a stale graph during optimisation.
            if not getattr(self, "enable_det_grad", False):
                placeholder = placeholder.detach()
            batch_for_det[f"{target_var}_next"] = placeholder.clone()
        prev_train = self.f_det.training
        self.f_det.eval()
        if getattr(self, "enable_det_grad", False):
            det_preds = self.f_det(batch_for_det)
        else:
            with torch.no_grad():
                det_preds = self.f_det(batch_for_det)
        if prev_train and not getattr(self, "_det_freeze", True):
            self.f_det.train(True)
        det_pred_phys = det_preds[target_var]  # [B, N, C]

        # Normalized + targshifted version for conditioning the velocity UNet.
        offset = batch[f"{target_var}_offset"]
        scale = batch[f"{target_var}_scale"]
        det_pred_norm = (det_pred_phys - offset) / scale
        if getattr(self, "targshift", False) or (hasattr(self.submodel, "targshift") and self.submodel.targshift):
            det_pred_norm = det_pred_norm - det_pred_norm.mean((1, 2), keepdim=True)
        B, N, C = det_pred_norm.shape
        det_pred_grid = det_pred_norm.reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)
        return det_pred_phys, det_pred_grid

    def _sigma_res_for_phys(self, like: torch.Tensor) -> torch.Tensor:
        """Return sigma_res broadcastable against tensor with trailing channel dim."""
        return self.sigma_res.to(like.device, dtype=like.dtype).view(*((1,) * (like.dim() - 1)), -1)

    def _sigma_res_for_grid(self, like: torch.Tensor) -> torch.Tensor:
        """Return sigma_res broadcastable against [B, C, H, W]."""
        return self.sigma_res.to(like.device, dtype=like.dtype).view(1, -1, 1, 1)

    # ------------------------------------------------------------------ training
    def training_forward(self, batch):
        target_var = self.target_vars[0]

        # Deterministic backbone prediction.
        det_pred_phys, det_pred_grid = self._det_predict(batch)

        # Residual target in unit-variance space, [B, N, C].
        x_next_phys = batch[f"{target_var}_next"]
        sigma_phys = self._sigma_res_for_phys(x_next_phys)
        residual_target = (x_next_phys - det_pred_phys) / sigma_phys
        B, N, C = residual_target.shape
        x_1 = residual_target.reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)

        # Standard conditioning channels (target placeholder + forcings, normalized).
        x_in = self.preprocess_inputs(batch)  # [B C_in_in Nlat Nlon]

        # Append det_pred as extra conditioning group (normalized + targshift'd).
        x_in = torch.cat([x_in, det_pred_grid.to(x_in.dtype)], dim=1)

        # Sample x_0 ~ N(0,I) shaped like x_1 (residual space).
        x_0 = torch.randn_like(x_1)
        if self.use_ot_coupling:
            x_0 = compute_ot_coupling(x_0, x_1)

        # Sample timestep and path.
        t = self._sample_time(B, x_in.device)
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        x_t = path_sample.x_t

        # Replace target placeholder slot with x_t (residual at time t).
        nlev_target = self.nlev * len(self.target_vars)
        x_in[:, :nlev_target, :, :] = x_t

        # Append time channel.
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
            # Train / fallback paths share parent behaviour (which dispatches
            # to our overridden training_forward).
            return super().forward(batch, mode=mode)

        # ---- generate ----
        target_var = self.target_vars[0]

        # Deterministic backbone prediction (used for both conditioning and final state).
        det_pred_phys, det_pred_grid = self._det_predict(batch)

        # Build conditioning channels.
        x_in = self.preprocess_inputs(batch)
        nlev_target = self.nlev * len(self.target_vars)
        all_levels = x_in[:, :nlev_target, :, :]
        x_in_with_det = torch.cat([x_in, det_pred_grid.to(x_in.dtype)], dim=1)

        # Initial noise (residual space, unit variance).
        if "noise" in batch:
            noise = batch["noise"]
            B_, N_, C_ = noise.shape
            x_init = noise.reshape(B_, self.nlat, self.nlon, C_).permute(0, 3, 1, 2)
        else:
            x_init = torch.randn_like(all_levels)
            noise_scale = float(self.generate_kwargs.get("noise_scale", 1.0))
            if noise_scale != 1.0:
                x_init = x_init * noise_scale
        B = x_init.shape[0]
        C = x_init.shape[1]

        # Posterior masking config (xco2/column observations) -- exactly like parent.
        if "obs_mask" in batch and "obs_values" in batch:
            obs_var = self.generate_kwargs.get("obs_var", "co2massmix")
            masking_config = self.prepare_masking_config(batch, B, C, obs_var)
            if hasattr(self.submodel, "targshift") and self.submodel.targshift:
                target_var_ = self.target_vars[0]
                batch_norm = (batch[target_var_] - batch[f"{target_var_}_offset"]) / batch[f"{target_var_}_scale"]
                targshift_mean = batch_norm.mean(dim=(1, 2), keepdim=True)
                masking_config["targshift_mean"] = targshift_mean.unsqueeze(-1)
        else:
            masking_config = {}

        trajectory = self.inference_forward(
            x_in_with_det,
            x_init,
            masking_config=masking_config,
            generate_kwargs=self.generate_kwargs,
        )

        # Convert residual to physical x_next:  x_next = det_pred + sigma_res * residual.
        if self.return_intermediates:
            res_final_grid = trajectory[-1, ...]  # [B, C, Nlat, Nlon]
        else:
            res_final_grid = trajectory  # [B, C, Nlat, Nlon]

        # Reshape final residual to [B, N, C] and combine with det_pred_phys.
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
