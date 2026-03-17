# torch
import torch
import torch.nn as nn
from flow_matching.path import AffineProbPath

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver

# neural_transport
from neural_transport.configs import DT_FALLBACK
from neural_transport.forward_model import XCO2ForwardModel
from neural_transport.inference.masking import (
    apply_masking as _apply_masking_fn,
)
from neural_transport.inference.masking import (
    apply_temporal_weighting as _apply_temporal_weighting_fn,
)
from neural_transport.inference.masking import (
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
            x_in = torch.cat([x, t_expanded, static_inputs], dim=1)  # [B C_total Nlat Nlon]
        else:
            x_in = torch.cat([x, t_expanded], dim=1)
        out = self.submodel.model(x_in)
        return out[:, : self.nlev, :, :]


class MaskedVelocityWrapper(VelocityWrapper):
    def __init__(
        self,
        submodel: nn.Module,
        masking_config: dict,
        nlev: int = 1,
        static_inputs: dict | None = None,
        **generate_kwargs,
    ):
        super().__init__(submodel=submodel, nlev=nlev, static_inputs=static_inputs)
        self.obs_mask = masking_config.get("obs_mask", None)
        self.obs_values = masking_config.get("obs_values", None)
        self.obs_mean = masking_config.get("obs_mean", None)
        self.obs_std = masking_config.get("obs_std", None)
        self.target_mean = masking_config.get("target_mean", None)
        self.target_std = masking_config.get("target_std", None)
        self.ak = masking_config.get("ak", None)
        self.xco2_prior = masking_config.get("xco2_prior", None)
        self.co2_profile_prior = masking_config.get("co2_profile_prior", None)
        self.pressure_weights = masking_config.get("pressure_weights", None)
        self.targshift_mean = masking_config.get("targshift_mean", None)
        self.time_grid = masking_config.get("time_grid", None)

        self.masking_time = generate_kwargs.get("masking_time", None)
        self.t_threshold = generate_kwargs.get("t_threshold", 0.9)
        self.masking_method = generate_kwargs.get("masking_method", "interpolate")
        self.conditioning_mode = generate_kwargs.get("conditioning_mode", "correction")
        self.guidance_scale = generate_kwargs.get("guidance_scale", 1.0)
        self.sigma_obs = generate_kwargs.get("sigma_obs", 1.0)
        self.spatial_smoothing_sigma = generate_kwargs.get("spatial_smoothing_sigma", 0.0)

        self.forward_model = XCO2ForwardModel.from_masking_config(masking_config)

    def compute_xco2(self, x):
        """OCO-2 forward model: XCO2 = xco2_prior + sum(h * a * (x - x_prior)).

        h = pressure_weights (h_k = dp_k / p_surface), a = averaging kernel.
        Returns XCO2 in normalized observation space.
        Delegates to self.forward_model.forward().
        """
        return self.forward_model.forward(x)

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
        x_effective = self.apply_temporal_weighting(x, x_masked, t)

        dt = self.compute_dt(t)

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
        v = super().forward(x, t)  # learned velocity on UNMODIFIED state

        # Compute target state at observed locations
        x_masked = self.apply_masking(x, t)

        # Target velocity: what velocity would bring x to x_masked at t=1
        remaining_time = (1.0 - t.view(-1, 1, 1, 1)).clamp(min=1e-3)
        target_v = (x_masked - x) / remaining_time

        # Temporal weighting
        mask_weight = self._get_temporal_weight(t)

        # Blend: at observed locations use weighted target velocity
        v_obs = mask_weight * target_v + (1.0 - mask_weight) * v
        v_projected = torch.where(self.obs_mask, v_obs, v)

        return v_projected

    def forward_guidance(self, x, t):
        """Soft gradient guidance (DPS-style): add data-fidelity gradient to velocity."""
        v = super().forward(x, t)  # learned velocity on UNMODIFIED state

        if self.ak is not None:
            h_ak = self.forward_model._get_h_ak_for_x(x)
            xco2 = self.compute_xco2(x)
            # Use torch.where to avoid NaN from obs_values at unobserved locations
            obs_safe = torch.where(self.obs_mask, self.obs_values.detach(), torch.zeros_like(xco2))
            column_error = torch.where(self.obs_mask, xco2 - obs_safe, torch.zeros_like(xco2))
            if self.spatial_smoothing_sigma > 0:
                column_error = _gaussian_smooth_2d(column_error, self.spatial_smoothing_sigma)
            # Jacobian transpose of H(x) = sum_k h_k a_k x_k: gradient is h_k * a_k * error
            guidance = (h_ak / (self.sigma_obs**2)) * column_error
        else:
            obs_safe = torch.where(self.obs_mask, self.obs_values.detach(), torch.zeros_like(x))
            guidance = torch.where(self.obs_mask, (x - obs_safe) / (self.sigma_obs**2), torch.zeros_like(x))
            if self.spatial_smoothing_sigma > 0:
                guidance = _gaussian_smooth_2d(guidance, self.spatial_smoothing_sigma)

        mask_weight = self._get_temporal_weight(t)
        return v - self.guidance_scale * mask_weight * guidance

    def forward_repaint(self, x, t):
        """Repaint: hard replacement of velocity at observed locations."""
        v = super().forward(x, t)

        remaining_time = (1.0 - t.view(-1, 1, 1, 1)).clamp(min=1e-3)

        if self.ak is not None:
            h = self.pressure_weights if self.pressure_weights is not None else 1.0 / x.shape[1]
            h_ak = h * self.ak
            h_ak_sum = h_ak.sum(dim=1, keepdim=True).clamp(min=1e-12)
            xco2_current = self.compute_xco2(x)
            xco2_target_v = (self.obs_values - xco2_current) / remaining_time
            # Apply uniform target velocity across all levels to correctly constrain column.
            # Target velocity per level: xco2_target_v / h_ak_sum (broadcast across levels)
            target_v = xco2_target_v / h_ak_sum
        else:
            target_v = (self.obs_values - x) / remaining_time

        mask_weight = self._get_temporal_weight(t)
        v_final = torch.where(self.obs_mask, mask_weight * target_v + (1 - mask_weight) * v, v)
        return v_final

    def _get_temporal_weight(self, t):
        """Get scalar temporal weight for conditioning."""
        return get_temporal_weight(t, self.masking_time, self.t_threshold)

    def apply_masking(self, x, t):
        """Route to appropriate masking method."""
        return _apply_masking_fn(
            method=self.masking_method,
            x=x,
            t=t,
            obs_mask=self.obs_mask,
            obs_values=self.obs_values,
            forward_model=self.forward_model,
            ak=self.ak,
            pressure_weights=self.pressure_weights,
            target_mean=self.target_mean,
            target_std=self.target_std,
            obs_mean=self.obs_mean,
            obs_std=self.obs_std,
            xco2_prior=self.xco2_prior,
            co2_profile_prior=self.co2_profile_prior,
        )

    def apply_temporal_weighting(self, x, x_masked, t):
        """Apply temporal weighting based on masking_time strategy."""
        return _apply_temporal_weighting_fn(x, x_masked, t, self.masking_time, self.t_threshold)

    def compute_dt(self, t):
        if self.time_grid is not None:
            idx = torch.searchsorted(self.time_grid, t.item())
            if idx == 0:
                dt = self.time_grid[1] - self.time_grid[0]
            elif idx >= len(self.time_grid):
                dt = self.time_grid[-1] - self.time_grid[-2]
            else:
                dt = self.time_grid[idx] - self.time_grid[idx - 1]
        else:
            dt = DT_FALLBACK
        return dt


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
    ):
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

    def forward(self, batch):
        if not hasattr(self, "generate_kwargs"):
            self.generate_kwargs = {}
        if self.training:
            x_out, dx_t, time_weight = self.training_forward(batch)
            preds = self.postprocess_outputs(x_out, batch, denormalize=False)
            preds["dx_t"] = dx_t.permute(0, 2, 3, 1).reshape(*preds[self.target_vars[0]].shape)  # [B N C]
            if time_weight is not None:
                # Broadcast weight [B,1,1,1] -> [B,N,C] matching preds shape
                B = preds[self.target_vars[0]].shape[0]
                preds["time_loss_weight"] = time_weight.view(B, 1, 1).expand_as(preds[self.target_vars[0]])
            return preds
        elif self.generating:
            x_in = self.preprocess_inputs(batch)
            all_levels = x_in[:, : self.nlev * len(self.target_vars), :, :]  # [B C Nlat Nlon]
            # surface_level = x_in[:, :1, :, :]  # [B 1 Nlat Nlon]
            if "noise" in batch:
                noise = batch["noise"]
                B, N, C = noise.shape
                x_init = noise.reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)  # [B C Nlat Nlon]
            else:
                x_init = torch.randn_like(all_levels, device=x_in.device)
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
        # extract target_vars to get x_1 [B C Nlat Nlon] and normalize
        batch_normalized = self.normalize_batch_target_vars(batch)
        x_1_normalized = batch_normalized[f"{self.target_vars[0]}_next"]
        B, _, C = x_1_normalized.shape
        x_1_normalized = x_1_normalized.reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)

        # sample noise  x_0 ~ N(0, I), [B C Nlat Nlon]
        x_0 = torch.randn_like(x_in, device=x_in.device)

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
            return MaskedVelocityWrapper(
                submodel=submodel,
                masking_config=masking_config,
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

        # Posterior sampler dispatch via registry
        sampler_name = generate_kwargs.get("sampler", None)
        if sampler_name in _SAMPLER_KWARGS_MAP:
            from neural_transport.inference.samplers import create_sampler

            velocity_model = VelocityWrapper(submodel=self.submodel, nlev=self.nlev)
            sampler_kwargs = {k: generate_kwargs[k] for k in _SAMPLER_KWARGS_MAP[sampler_name] if k in generate_kwargs}
            sampler = create_sampler(sampler_name, velocity_model, masking_config, **sampler_kwargs)
            return sampler.sample(x_init, time_grid, self.return_intermediates)

        # UNet expects normalization parameters
        velocity_model = self.return_velocity_wrapper(
            submodel=self.submodel,
            static_inputs=None,
            masking_config=masking_config,
            generate_kwargs=generate_kwargs,
            # static_inputs=x_in[:,:self.nlev*len(self.target_vars),:,:],  # [B C Nlat Nlon] (static inputs for conditioning later)
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
        )
        trajectory = solver.sample(**solver_kwargs)  # [T B C Nlat Nlon]
        return trajectory
