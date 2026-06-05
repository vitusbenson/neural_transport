# torch
import torch
import torch.nn as nn
from flow_matching.path import AffineProbPath

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver

# neural_transport
from neural_transport.forward_model import XCO2ForwardModel
from neural_transport.inference.masking import (
    apply_masking as _apply_masking_fn,
)
from neural_transport.inference.masking import (
    apply_temporal_weighting as _apply_temporal_weighting_fn,
)
from neural_transport.inference.masking import (
    compute_dt,
    get_temporal_weight,
)
from neural_transport.models import MODELS
from neural_transport.models.regulargrid import RegularGridModel
from neural_transport.tools.spatial import gaussian_smooth_2d as _gaussian_smooth_2d

_SAMPLER_KWARGS_MAP = {
    "flowdps": ["sigma_obs", "spatial_smoothing_sigma", "fresh_noise"],
    "sde": [
        "sigma_obs",
        "spatial_smoothing_sigma",
        "fresh_noise",
        "sigma_max",
        "noise_schedule",
        "n_corrector_steps",
        "corrector_step_size",
        "corrector_snr",
        "use_projection",
    ],
    "fig": [
        "sigma_obs",
        "spatial_smoothing_sigma",
        "k_steps",
        "step_size_c",
        "noise_scale_w",
        "skip_first_last",
    ],
    "ictm": [
        "sigma_obs",
        "spatial_smoothing_sigma",
        "fresh_noise",
        "r_max",
        "r_schedule",
        "n_inner_steps",
        "inner_lr",
    ],
    "mcg": [
        "sigma_obs",
        "spatial_smoothing_sigma",
        "fresh_noise",
        "n_forward_steps",
    ],
    "pcfm": [
        "sigma_obs",
        "spatial_smoothing_sigma",
        "fresh_noise",
        "n_forward_steps",
        "lambda_penalty",
    ],
    "fmps": [
        "sigma_obs",
        "spatial_smoothing_sigma",
        "guidance_strength",
        "r_schedule",
        "svd_rank",
        "spectral_k_low",
        "spectral_k_high",
        "grad_clip_norm",
    ],
    "dflow": [
        "sigma_obs",
        "spatial_smoothing_sigma",
        "n_opt_steps",
        "lr",
        "reg_weight",
        "reg_type",
        "optimizer",
        "use_checkpointing",
    ],
}


class VelocityWrapper(nn.Module):
    def __init__(
        self,
        submodel: nn.Module,
        nlev: int = 1,
        static_inputs: torch.Tensor | None = None,
    ):
        super().__init__()
        self.submodel = submodel
        self.nlev = nlev
        self.static_inputs = static_inputs if static_inputs is not None else torch.empty(0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, _, Nlat, Nlon = x.shape
        # Handle both scalar t and batch t [B]
        if t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1):
            t_expanded = t.view(1, 1, 1, 1).expand(B, 1, Nlat, Nlon)
        else:
            t_expanded = t.view(B, 1, 1, 1).expand(B, 1, Nlat, Nlon)
        if self.static_inputs.numel() > 0:
            static_inputs = self.static_inputs.to(x.device)
            # Order must match training_forward: [x_t, conditioning, time].
            x_in = torch.cat([x, static_inputs, t_expanded], dim=1)  # [B C_total Nlat Nlon]
        else:
            x_in = torch.cat([x, t_expanded], dim=1)
        out = self.submodel.model(x_in)
        return out[:, : self.nlev, :, :]


class MaskedVelocityWrapper(VelocityWrapper):
    def __init__(
        self,
        submodel: nn.Module,
        masking_config: dict,
        forward_model=None,
        nlev: int = 1,
        static_inputs: dict | None = None,
        **generate_kwargs,
    ):
        super().__init__(submodel=submodel, nlev=nlev, static_inputs=static_inputs)
        self.masking_config = masking_config
        # Frequently-accessed shortcuts
        self.obs_mask = masking_config.get("obs_mask", None)
        self.obs_values = masking_config.get("obs_values", None)
        self.ak = masking_config.get("ak", None)
        self.pressure_weights = masking_config.get("pressure_weights", None)
        self.time_grid = masking_config.get("time_grid", None)
        # Soft obs_weight: float [0,1] field (from gaussian blur of binary mask)
        self.obs_weight = masking_config.get("obs_weight", None)

        self.masking_time = generate_kwargs.get("masking_time", None)
        self.t_threshold = generate_kwargs.get("t_threshold", 0.9)
        self.masking_method = generate_kwargs.get("masking_method", "interpolate")
        self.conditioning_mode = generate_kwargs.get("conditioning_mode", "correction")
        self.guidance_scale = generate_kwargs.get("guidance_scale", 1.0)
        self.sigma_obs = generate_kwargs.get("sigma_obs", 1.0)
        self.spatial_smoothing_sigma = generate_kwargs.get("spatial_smoothing_sigma", 0.0)

        self.forward_model = (
            forward_model if forward_model is not None else XCO2ForwardModel.from_masking_config(masking_config)
        )

    def forward(self, x, t):
        if self.conditioning_mode == "velocity_projection":
            return self.forward_velocity_projection(x, t)
        elif self.conditioning_mode == "guidance":
            return self.forward_guidance(x, t)
        elif self.conditioning_mode == "repaint":
            return self.forward_repaint(x, t)
        else:
            return self.forward_correction(x, t)

    def forward_correction(self, x, t):
        """Original approach: modify state, compute correction term."""
        x_masked = self.apply_masking(x, t)
        x_effective = _apply_temporal_weighting_fn(x, x_masked, t, self.masking_time, self.t_threshold)

        dt = compute_dt(t, self.time_grid)

        # For column constraints (ak != None), feed uncorrected x to network.
        # Column corrections shift noise uniformly which is out-of-distribution
        # for the UNet and causes velocity explosion. Direct field corrections
        # (ak=None) produce realistic intermediate states that the UNet handles.
        x_for_network = x if self.ak is not None else x_effective
        dtx = (x_effective - x) / dt + super().forward(x_for_network, t)

        return dtx

    def forward_velocity_projection(self, x, t):
        """Velocity projection: network sees unmodified state, velocity is
        blended between learned velocity and target velocity at observed locations."""
        v = super().forward(x, t)
        x_masked = self.apply_masking(x, t)

        remaining_time = (1.0 - t.view(-1, 1, 1, 1)).clamp(min=1e-3)
        target_v = (x_masked - x) / remaining_time

        mask_weight = get_temporal_weight(t, self.masking_time, self.t_threshold)

        v_obs = mask_weight * target_v + (1.0 - mask_weight) * v
        if self.obs_weight is not None:
            v_projected = self.obs_weight * v_obs + (1.0 - self.obs_weight) * v
        else:
            v_projected = torch.where(self.obs_mask, v_obs, v)

        return v_projected

    def forward_guidance(self, x, t):
        """Soft gradient guidance (DPS-style): add data-fidelity gradient to velocity."""
        v = super().forward(x, t)

        if self.ak is not None:
            h_ak = self.forward_model.effective_kernel(x)
            xco2 = self.forward_model.forward(x)
            if self.obs_weight is not None:
                column_error = self.obs_weight * (xco2 - self.obs_values.detach())
            else:
                obs_safe = torch.where(self.obs_mask, self.obs_values.detach(), torch.zeros_like(xco2))
                column_error = torch.where(self.obs_mask, xco2 - obs_safe, torch.zeros_like(xco2))
            if self.spatial_smoothing_sigma > 0:
                column_error = _gaussian_smooth_2d(column_error, self.spatial_smoothing_sigma)
            guidance = (h_ak / (self.sigma_obs**2)) * column_error
        else:
            if self.obs_weight is not None:
                guidance = self.obs_weight * (x - self.obs_values.detach()) / (self.sigma_obs**2)
            else:
                obs_safe = torch.where(self.obs_mask, self.obs_values.detach(), torch.zeros_like(x))
                guidance = torch.where(self.obs_mask, (x - obs_safe) / (self.sigma_obs**2), torch.zeros_like(x))
            if self.spatial_smoothing_sigma > 0:
                guidance = _gaussian_smooth_2d(guidance, self.spatial_smoothing_sigma)

        mask_weight = get_temporal_weight(t, self.masking_time, self.t_threshold)
        return v - self.guidance_scale * mask_weight * guidance

    def forward_repaint(self, x, t):
        """Repaint: hard replacement of velocity at observed locations."""
        v = super().forward(x, t)

        remaining_time = (1.0 - t.view(-1, 1, 1, 1)).clamp(min=1e-3)

        if self.ak is not None:
            h = self.pressure_weights if self.pressure_weights is not None else 1.0 / x.shape[1]
            h_ak = h * self.ak
            h_ak_sum = h_ak.sum(dim=1, keepdim=True).clamp(min=1e-12)
            xco2_current = self.forward_model.forward(x)
            xco2_target_v = (self.obs_values - xco2_current) / remaining_time
            target_v = xco2_target_v / h_ak_sum
        else:
            target_v = (self.obs_values - x) / remaining_time

        mask_weight = get_temporal_weight(t, self.masking_time, self.t_threshold)
        v_conditioned = mask_weight * target_v + (1 - mask_weight) * v
        if self.obs_weight is not None:
            v_final = self.obs_weight * v_conditioned + (1.0 - self.obs_weight) * v
        else:
            v_final = torch.where(self.obs_mask, v_conditioned, v)
        return v_final

    def apply_masking(self, x, t):
        """Route to appropriate masking method."""
        cfg = {k: v for k, v in self.masking_config.items() if k not in ("obs_mask", "obs_values", "time_grid")}
        return _apply_masking_fn(
            method=self.masking_method,
            x=x,
            t=t,
            obs_mask=self.obs_mask,
            obs_values=self.obs_values,
            forward_model=self.forward_model,
            **cfg,
        )


@torch.no_grad()
def compute_ot_coupling(x_0, x_1, reg=0.05, num_iter=50):
    """Minibatch OT coupling via Sinkhorn. Returns x_0 reordered to match x_1."""
    B = x_0.shape[0]
    x0_flat = x_0.reshape(B, -1)  # [B, D]
    x1_flat = x_1.reshape(B, -1)
    # Cost matrix [B, B]
    C = torch.cdist(x0_flat, x1_flat, p=2) ** 2
    # Sinkhorn iterations
    C_reg = C / (reg * C.max().clamp(min=1e-12))
    K = torch.exp(-C_reg)
    u = torch.ones(B, device=x_0.device)
    for _ in range(num_iter):
        v = 1.0 / (K.T @ u + 1e-12)
        u = 1.0 / (K @ v + 1e-12)
    # Transport plan -> hard assignment
    plan = torch.diag(u) @ K @ torch.diag(v)  # [B, B]
    # For each x_1[j], find best matching x_0[i]: perm[j] = argmax_i plan[i,j]
    perm = plan.argmax(dim=0)  # [B]
    return x_0[perm]


class FlowMatching(RegularGridModel):
    def init_model(
        self,
        submodel="unet",
        model_kwargs={},
        generating=False,
        return_intermediates=False,
        method='midpoint',
        nlev=1,
        step_size=0.01,
        generate_kwargs=None,
        use_ot_coupling=False,
        time_grid_spacing='uniform',
        atol=1e-5,
        rtol=1e-5,
        time_sampling='uniform',
        time_sampling_kwargs=None,
        time_loss_weight=None,
        rollout_aug_sigma=0.0,
        rollout_aug_prob=1.0,
    ):
        # Phase 25c rollout-augmentation knobs (no-op when sigma==0).
        self.rollout_aug_sigma = float(rollout_aug_sigma or 0.0)
        self.rollout_aug_prob = float(rollout_aug_prob)
        self.submodel = MODELS[submodel](**model_kwargs)
        self.return_intermediates = return_intermediates
        self.generating = generating
        self.method = method
        self.nlev = nlev
        self.step_size = step_size
        self.generate_kwargs = generate_kwargs if generate_kwargs is not None else {}
        self.use_ot_coupling = use_ot_coupling
        self.time_grid_spacing = time_grid_spacing
        self.atol = atol
        self.rtol = rtol
        self.time_sampling = time_sampling
        self.time_sampling_kwargs = time_sampling_kwargs or {}
        self.time_loss_weight = time_loss_weight
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.target_vars = self.submodel.target_vars  # Here target_vars[0] is supposed to be "co2massmix"

    def forward(self, batch, *, mode=None):
        """Forward pass with explicit mode dispatch.

        Args:
            batch: Input batch dict.
            mode: Override dispatch mode. One of ``"train"``, ``"generate"``,
                or ``None``.  When ``None`` (default), the mode is inferred
                from ``self.training`` / ``self.generating`` as before.
        """
        if not hasattr(self, "generate_kwargs"):
            self.generate_kwargs = {}

        # Resolve mode: explicit kwarg takes priority over module state.
        _mode = mode
        if _mode is None:
            if self.training:
                _mode = "train"
            elif self.generating:
                _mode = "generate"

        if _mode == "train":
            x_out, dx_t, time_weight = self.training_forward(batch)
            preds = self.postprocess_outputs(x_out, batch, denormalize=False)
            preds["dx_t"] = dx_t.permute(0, 2, 3, 1).reshape(*preds[self.target_vars[0]].shape)  # [B N C]
            if time_weight is not None:
                # Broadcast weight [B,1,1,1] -> [B,N,C] matching preds shape
                B = preds[self.target_vars[0]].shape[0]
                preds["time_loss_weight"] = time_weight.view(B, 1, 1).expand_as(preds[self.target_vars[0]])
            return preds
        elif _mode == "generate":
            x_in = self.preprocess_inputs(batch)
            all_levels = x_in[:, : self.nlev * len(self.target_vars), :, :]  # [B C Nlat Nlon]
            # surface_level = x_in[:, :1, :, :]  # [B 1 Nlat Nlon]
            if "noise" in batch:
                noise = batch["noise"]
                B, N, C = noise.shape
                x_init = noise.reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)  # [B C Nlat Nlon]
            else:
                x_init = torch.randn_like(all_levels, device=x_in.device)
                # Phase 25e: AWG-style initial-noise scaling (rho>1 widens
                # source distribution; counters AR-time underdispersion).
                noise_scale = float(self.generate_kwargs.get("noise_scale", 1.0))
                if noise_scale != 1.0:
                    x_init = x_init * noise_scale
                B, C = x_init.shape[0], x_init.shape[1]
            if "obs_mask" in batch and "obs_values" in batch:
                obs_var = self.generate_kwargs.get("obs_var", "co2massmix")
                masking_config = self.prepare_masking_config(batch, B, C, obs_var)
                # Compute targshift mean for correct physical-space conversion in column obs.
                # batch[target_var] is [B, N, C]; targshift subtracts mean over (N, C) dims.
                # We need the per-sample mean to undo targshift in compute_xco2.
                if hasattr(self.submodel, 'targshift') and self.submodel.targshift:
                    target_var = self.target_vars[0]
                    batch_norm = (batch[target_var] - batch[f"{target_var}_offset"]) / batch[f"{target_var}_scale"]
                    # targshift uses mean((1,2), keepdim=True) on [B, N, C] -> [B, 1, 1]
                    targshift_mean = batch_norm.mean(dim=(1, 2), keepdim=True)  # [B, 1, 1]
                    # Reshape to [B, 1, 1, 1] for broadcasting with x [B, C, Nlat, Nlon]
                    masking_config["targshift_mean"] = targshift_mean.unsqueeze(-1)  # [B, 1, 1, 1]
            else:
                masking_config = {}
            trajectory = self.inference_forward(
                x_in, x_init, masking_config=masking_config, generate_kwargs=self.generate_kwargs
            )
            if self.return_intermediates:
                # trajectory is [T, B, C, Nlat, Nlon]
                x_out = trajectory[-1, ...]
                sol = self.postprocess_outputs(x_out, batch)
                T, B, C, Nlat, Nlon = trajectory.shape
                trajectory = trajectory.permute(0, 1, 3, 4, 2)  # [T B Nlat Nlon C]
                trajectory = trajectory.reshape(T, B, Nlat * Nlon, C)  # [T B Nlat*Nlon C]
                sol["trajectory"] = trajectory
            else:
                # trajectory is [B, C, Nlat, Nlon] (final timestep only)
                x_out = trajectory
                sol = self.postprocess_outputs(x_out, batch)
            return sol
        else:
            return super().forward(batch)

    def _sample_time(self, B, device):
        """Sample timesteps for training with configurable distribution.

        Args:
            B: batch size
            device: torch device

        Returns:
            Tensor of shape [B] with timesteps in (0, 1).
        """
        if self.time_sampling == "logit_normal":
            mean = self.time_sampling_kwargs.get("mean", 0.0)
            std = self.time_sampling_kwargs.get("std", 1.0)
            u = torch.randn(B, device=device) * std + mean
            return torch.sigmoid(u)
        elif self.time_sampling == "beta":
            a = self.time_sampling_kwargs.get("a", 2.0)
            b = self.time_sampling_kwargs.get("b", 5.0)
            return torch.distributions.Beta(a, b).sample((B,)).to(device)
        else:  # uniform
            return torch.rand(B, device=device)

    def _compute_time_loss_weight(self, t):
        """Compute time-dependent loss weighting.

        Args:
            t: [B] tensor of timesteps

        Returns:
            [B, 1, 1, 1] weight tensor, or 1.0 if no weighting.
        """
        if self.time_loss_weight is None:
            return 1.0
        elif self.time_loss_weight == "snr":
            # SNR weighting: w(t) = 1 / (1 - t + eps)^2
            # Higher weight near t=1 where signal-to-noise is higher
            w = 1.0 / (1.0 - t + 1e-4) ** 2
            return w.view(-1, 1, 1, 1)
        elif self.time_loss_weight == "sigma_inv":
            # Inverse sigma weighting: w(t) = 1 / sigma(t)
            # For CondOT: sigma(t) = 1 - t
            w = 1.0 / (1.0 - t + 1e-4)
            return w.view(-1, 1, 1, 1)
        else:
            return 1.0

    # training
    def training_forward(self, batch):
        # sample data [B C Nlat Nlon]
        x_in = self.preprocess_inputs(batch)

        # Optional rollout-augmentation (Phase 25c v0): perturb the prior CO2
        # channels with Gaussian noise to mimic the inference-time distribution
        # shift between GT prior (training) and model-predicted prior (AR
        # rollout). If `rollout_aug_sigma` is None or 0, this is a no-op.
        # If `rollout_aug_prob` < 1, perturb only that fraction of the batch.
        sigma = getattr(self, "rollout_aug_sigma", 0.0) or 0.0
        if sigma > 0.0 and self.training:
            prob = getattr(self, "rollout_aug_prob", 1.0)
            n_target_chans = self.nlev * len(self.target_vars)
            # Slot 0..n_target_chans-1 is the target-time placeholder
            # (`co2massmix_next`), overwritten later by x_t. The genuine prior
            # `co2massmix` lives at channels n_target_chans..2*n_target_chans-1.
            prior_start = n_target_chans
            prior_end = prior_start + n_target_chans
            if prior_end <= x_in.shape[1]:
                B = x_in.shape[0]
                noise = (
                    torch.randn(
                        B,
                        prior_end - prior_start,
                        *x_in.shape[2:],
                        device=x_in.device,
                        dtype=x_in.dtype,
                    )
                    * sigma
                )
                if prob < 1.0:
                    keep = (torch.rand(B, 1, 1, 1, device=x_in.device) < prob).to(x_in.dtype)
                    noise = noise * keep
                x_in = x_in.clone()
                x_in[:, prior_start:prior_end] = x_in[:, prior_start:prior_end] + noise
        # extract target_vars to get x_1 [B C Nlat Nlon] and normalize
        batch_normalized = self.normalize_batch_target_vars(batch)
        x_1_normalized = batch_normalized[f"{self.target_vars[0]}_next"]
        B, _, C = x_1_normalized.shape
        x_1_normalized = x_1_normalized.reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)

        # sample noise  x_0 ~ N(0, I), shaped like x_1 (target-only channels,
        # not full x_in which may include conditioning forcings).
        x_0 = torch.randn_like(x_1_normalized, device=x_in.device)

        # Minibatch OT coupling: reorder x_0 to reduce transport cost
        if self.use_ot_coupling:
            x_0 = compute_ot_coupling(x_0, x_1_normalized)

        # sample time t \in [0,1], [B] -> [B 1 Nlat Nlon]
        B, _, nlat, nlon = x_in.shape
        t = self._sample_time(B, x_in.device)

        # sample path
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1_normalized)

        # sample x_t from the path
        x_t = path_sample.x_t
        # x_t = path_sample.x_t + x_0 (simplified version)
        # this requires target_vars to be first in input_vars and all nlev-dimensional
        x_in[:, : self.nlev * len(self.target_vars), :, :] = x_t  # [B C Nlat Nlon]

        t_expanded = path_sample.t.view(-1, 1, 1, 1).expand(B, 1, nlat, nlon)
        x_in = torch.cat([x_in, t_expanded], dim=1)  # [B C+1 Nlat Nlon]

        x_out = self.submodel.model(x_in)

        dx_t = path_sample.dx_t

        # Compute time-dependent loss weight
        time_weight = self._compute_time_loss_weight(t)
        time_weight = time_weight if not isinstance(time_weight, float) else None

        return x_out, dx_t, time_weight

    def prepare_masking_config(self, batch, B, C, obs_var):
        if "xco2_averaging_kernel" in batch:
            obs_mask = batch["obs_mask"].reshape(B, self.nlat, self.nlon, 1).permute(0, 3, 1, 2)
            obs_values = batch["obs_values"].reshape(B, self.nlat, self.nlon, 1).permute(0, 3, 1, 2)
            ak = batch["xco2_averaging_kernel"].reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)
            xco2_prior = batch["xco2_apriori"].reshape(B, self.nlat, self.nlon, 1).permute(0, 3, 1, 2)
            co2_profile_prior = batch["co2_profile_apriori"].reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)
            if "pressure_weight" in batch:
                pressure_weights = batch["pressure_weight"].reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)
            else:
                pressure_weights = None
        else:
            obs_mask = batch["obs_mask"].reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)
            obs_values = batch["obs_values"].reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)
            ak = None
            xco2_prior = None
            co2_profile_prior = None
            pressure_weights = None

        obs_mean = batch[f"{obs_var}_offset"].view(B, 1, 1, 1)
        obs_std = batch[f"{obs_var}_scale"].view(B, 1, 1, 1)
        target_mean = batch[f"{self.target_vars[0]}_offset"].view(B, 1, 1, 1)
        target_std = batch[f"{self.target_vars[0]}_scale"].view(B, 1, 1, 1)

        # Soft observation weight (float [0,1]) for smooth boundaries
        obs_weight = None
        if "obs_weight" in batch:
            if "xco2_averaging_kernel" in batch:
                obs_weight = batch["obs_weight"].reshape(B, self.nlat, self.nlon, 1).permute(0, 3, 1, 2)
            else:
                obs_weight = batch["obs_weight"].reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)

        masking_config = {
            "obs_mask": obs_mask,  # [B 1 Nlat Nlon]
            "obs_values": obs_values,  # [B 1 Nlat Nlon]
            "obs_mean": obs_mean,  # [B 1 1 1]
            "obs_std": obs_std,  # [B 1 1 1]
            "target_mean": target_mean,  # [B 1 1 1]
            "target_std": target_std,  # [B 1 1 1]
            "ak": ak,  # [B C Nlat Nlon]
            "xco2_prior": xco2_prior,
            "co2_profile_prior": co2_profile_prior,
            "pressure_weights": pressure_weights,  # [B C Nlat Nlon] or None
            "obs_weight": obs_weight,  # [B 1 Nlat Nlon] float or None
        }
        return masking_config

    def return_velocity_wrapper(
        self,
        submodel,
        static_inputs=None,
        masking_config=None,
        generate_kwargs=None,
    ):
        if masking_config is None:
            masking_config = getattr(self, 'masking_config', {})
        obs_mask = masking_config.get("obs_mask", None)
        obs_values = masking_config.get("obs_values", None)

        if generate_kwargs is None:
            generate_kwargs = getattr(self, 'generate_kwargs', {})

        if obs_mask is not None and obs_values is not None:
            forward_model = XCO2ForwardModel.from_masking_config(masking_config)
            return MaskedVelocityWrapper(
                submodel=submodel,
                masking_config=masking_config,
                forward_model=forward_model,
                nlev=self.nlev,
                static_inputs=static_inputs,
                **generate_kwargs,
            )
        else:
            return VelocityWrapper(
                submodel=submodel,
                nlev=self.nlev,
                static_inputs=static_inputs,
            )

    @staticmethod
    def _build_time_grid(steps, device, spacing='uniform'):
        """Build time grid with specified spacing.

        Args:
            steps: number of grid points
            device: torch device
            spacing: 'uniform', 'cosine' (denser at endpoints), or
                     'front_loaded' (denser near t=0)

        Returns:
            Tensor of shape [steps] in [0, 1], monotonically increasing.
        """
        t_lin = torch.linspace(0, 1, steps, device=device)
        if spacing == 'cosine':
            return 0.5 * (1 - torch.cos(torch.pi * t_lin))
        elif spacing == 'front_loaded':
            return t_lin**2
        else:  # uniform
            return t_lin

    # inference_forward
    def inference_forward(self, x_in, x_init, masking_config=None, generate_kwargs=None):
        if generate_kwargs is None:
            generate_kwargs = {}

        refine_start = generate_kwargs.get("refine_start", 1.0)
        steps = generate_kwargs.get("steps", 11)
        spacing = generate_kwargs.get("time_grid_spacing", self.time_grid_spacing)

        # get timesteps for integration [T]
        if refine_start < 1.0:
            coarse = torch.linspace(0, refine_start, steps=steps, device=x_init.device)[:-1]
            fine = torch.linspace(refine_start, 1.0, steps=steps, device=x_init.device)
            time_grid = torch.cat([coarse, fine[1:]])
        else:
            time_grid = self._build_time_grid(steps - 1, x_init.device, spacing)
        masking_config["time_grid"] = time_grid

        # Pass conditioning channels (everything after the target-var slots) as
        # static_inputs so the UNet sees [x_t, conditioning, time].
        nlev_target = self.nlev * len(self.target_vars)
        static_inputs = x_in[:, nlev_target:, :, :] if x_in.shape[1] > nlev_target else None

        # Posterior sampler dispatch via registry
        sampler_name = generate_kwargs.get("sampler", None)
        if sampler_name in _SAMPLER_KWARGS_MAP:
            from neural_transport.inference.samplers import create_sampler

            velocity_model = VelocityWrapper(
                submodel=self.submodel,
                nlev=self.nlev,
                static_inputs=static_inputs,
            )
            sampler_kwargs = {k: generate_kwargs[k] for k in _SAMPLER_KWARGS_MAP[sampler_name] if k in generate_kwargs}
            sampler = create_sampler(sampler_name, velocity_model, masking_config, **sampler_kwargs)
            return sampler.sample(x_init, time_grid, self.return_intermediates)

        # UNet expects normalization parameters
        velocity_model = self.return_velocity_wrapper(
            submodel=self.submodel,
            static_inputs=static_inputs,
            masking_config=masking_config,
            generate_kwargs=generate_kwargs,
        )

        # solve the ODE to get the trajectory
        solver = ODESolver(velocity_model=velocity_model)
        solver_kwargs = dict(
            time_grid=time_grid,
            x_init=x_init,
            method=generate_kwargs.get("method", self.method),
            step_size=self.step_size,
            return_intermediates=self.return_intermediates,
            atol=generate_kwargs.get("atol", self.atol),
            rtol=generate_kwargs.get("rtol", self.rtol),
            enable_grad=generate_kwargs.get("enable_grad", False),
        )
        trajectory = solver.sample(**solver_kwargs)  # [T B C Nlat Nlon]
        return trajectory
