from typing import Optional
import inspect

# torch
import torch
import torch.nn as nn

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver

# neural_transport
from neural_transport.models import MODELS
from neural_transport.models.regulargrid import RegularGridModel


class VelocityWrapper(nn.Module):
    def __init__(
        self,
        submodel: nn.Module,
        nlev: int = 1,
        static_inputs: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.submodel = submodel
        self.nlev = nlev
        self.static_inputs = static_inputs if static_inputs is not None else torch.empty(0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, _, Nlat, Nlon = x.shape
        t_expanded = t.view(1, 1, 1, 1).expand(B, 1, Nlat, Nlon)
        if self.static_inputs.numel() > 0:
            static_inputs = self.static_inputs.to(x.device)
            x_in = torch.cat([x, t_expanded, static_inputs], dim=1) # [B C_total Nlat Nlon]
        else:
            x_in = torch.cat([x, t_expanded], dim=1)
        out = self.submodel.model(x_in)
        return out[:, :self.nlev, :, :]


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

    def compute_xco2(self, x):
        """OCO-2 forward model: XCO2 = xco2_prior + sum(h * a * (x - x_prior)).

        h = pressure_weights (h_k = dp_k / p_surface), a = averaging kernel.
        Returns XCO2 in normalized observation space.
        """
        h = self.pressure_weights
        if h is None:
            h = 1.0 / x.shape[1]

        if self.xco2_prior is not None and self.co2_profile_prior is not None:
            # Correct for targshift: when targshift is active, x is shifted by
            # the batch spatial mean, so x * std + mean doesn't recover physical values.
            # We need to add back the targshift_mean to get the true normalized value.
            x_corrected = x + self.targshift_mean if self.targshift_mean is not None else x
            x_phys = x_corrected * self.target_std + self.target_mean
            xco2 = self.xco2_prior + (h * self.ak * (x_phys - self.co2_profile_prior)).sum(dim=1, keepdim=True)
            return (xco2 - self.obs_mean) / self.obs_std
        else:
            h_ak = h * self.ak
            h_ak_sum = h_ak.sum(dim=1, keepdim=True).clamp(min=1e-12)
            # XCO2 is a weighted sum, not a weighted average.
            # When no prior is available, XCO2 = sum(h_k * a_k * x_k).
            # For normalized x, we need to account for the target mean/std:
            # XCO2 = sum(h_k * a_k * (x * std + mean)) = std * sum(h_k * a_k * x) + mean * h_ak_sum
            # In normalized obs space (assuming obs_mean ~ target_mean * h_ak_sum, obs_std ~ target_std):
            # We return the weighted sum plus a correction for the mean contribution.
            return (h_ak * x).sum(dim=1, keepdim=True) + (self.target_mean / self.target_std) * (h_ak_sum - 1.0)

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
            h = self.pressure_weights if self.pressure_weights is not None else 1.0 / x.shape[1]
            h_ak = h * self.ak
            h_ak_sum = h_ak.sum(dim=1, keepdim=True).clamp(min=1e-12)
            xco2 = self.compute_xco2(x)
            # Use torch.where to avoid NaN from obs_values at unobserved locations
            obs_safe = torch.where(self.obs_mask, self.obs_values.detach(), torch.zeros_like(xco2))
            column_error = torch.where(self.obs_mask, xco2 - obs_safe, torch.zeros_like(xco2))
            guidance = column_error / h_ak_sum
        else:
            obs_safe = torch.where(self.obs_mask, self.obs_values.detach(), torch.zeros_like(x))
            guidance = torch.where(self.obs_mask, x - obs_safe, torch.zeros_like(x))

        # TODO: investigate spatial smoothing of the guidance field for column conditioning.
        # The point-wise column error creates spatially discontinuous gradients that may
        # benefit from Gaussian smoothing before being applied to the 3D velocity field.
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
        masking_time = self.masking_time
        t_threshold = self.t_threshold
        if masking_time == "smooth_late_masking":
            return torch.sigmoid((t - t_threshold) * 20.0).view(-1, 1, 1, 1)
        elif masking_time == "step_late_masking":
            return (t >= t_threshold).float().view(-1, 1, 1, 1)
        elif masking_time == "smooth_early_masking":
            return torch.sigmoid((t_threshold - t) * 20.0).view(-1, 1, 1, 1)
        elif masking_time == "step_early_masking":
            return (t < t_threshold).float().view(-1, 1, 1, 1)
        else:
            return 1.0

    def apply_masking(self, x, t):
        """Route to appropriate masking method."""
        masking_fn = getattr(self, f"masking_{self.masking_method}", None)
        if masking_fn is None:
            raise ValueError(f"Unknown masking method: {self.masking_method}")

        sig = inspect.signature(masking_fn)
        if 't' in sig.parameters:
            x_masked = masking_fn(x, t)
        else:
            x_masked = masking_fn(x)
        return x_masked
    
    def apply_temporal_weighting(self, x, x_masked, t):
        """Apply temporal weighting based on masking_time strategy."""
        mask_weight = self._get_temporal_weight(t)
        return mask_weight * x_masked + (1.0 - mask_weight) * x
    
    def compute_dt(self, t):
        if self.time_grid is not None:
            idx = torch.searchsorted(self.time_grid, t.item())
            if idx == 0:
                dt = self.time_grid[1] - self.time_grid[0]
            elif idx >= len(self.time_grid):
                dt = self.time_grid[-1] - self.time_grid[-2]
            else:
                dt = self.time_grid[idx] - self.time_grid[idx-1]
        else:
            dt = 0.1  # fallback
        return dt

    def masking_simple(self, x):
        x_masked = torch.where(self.obs_mask, self.obs_values.detach(), x)
        return x_masked
    
    def masking_interpolate(self, x, t):
        obs_values = t * self.obs_values.detach() + (1.0 - t) * x
        x_masked = torch.where(self.obs_mask, obs_values, x)
        return x_masked
    
    def masking_preserve_global_mean(self, x):
        spatial_dims = (-2, -1)  # x.shape [B, C, Nlat, Nlon]
        obs_values = torch.where(self.obs_mask, self.obs_values, torch.zeros_like(x))

        # Fraction observed (per batch, per channel)
        m = self.obs_mask.float().mean(dim=spatial_dims, keepdim=True)  # shape [B, C, 1, 1]
        
        # Calculate means
        # Global mean before: s_all_before = m * s_mask_before + (1-m) * s_unmask_before
        s_all_before = x.mean(dim=spatial_dims, keepdim=True)  # [B, C, 1, 1]
        # mean of obs values (only over masked cells)
        mask_count = self.obs_mask.float().sum(dim=spatial_dims, keepdim=True)  # [B,C,1,1]
        s_mask_obs = torch.where(
            mask_count > 0,
            (obs_values * self.obs_mask.float()).sum(dim=spatial_dims, keepdim=True) / mask_count,
            s_all_before,
        )
        # Mean over unmasked cells before
        unmask = ~self.obs_mask
        unmask_count = unmask.float().sum(dim=spatial_dims, keepdim=True)
        s_unmask_before = torch.where(
            unmask_count > 0,
            (x * unmask.float()).sum(dim=spatial_dims, keepdim=True) / unmask_count,
            s_all_before,  # degenerate
        )
        # desired new mean on unmasked region to preserve global mean
        denom = (1.0 - m).clamp(min=1e-12)
        s_unmask_new = (s_all_before - m * s_mask_obs) / denom
        # additive correction applied only to unmasked cells:
        a_add = s_unmask_new - s_unmask_before  # shape [B,C,1,1]
        x_masked = torch.where(self.obs_mask, self.obs_values, x + a_add)

        return x_masked

    def masking_preserve_global_mean_and_var(self, x):
        """
        Replace masked cells with observations, and renormalize unmasked region
        so that global mean and variance of x are preserved.
        """
        spatial_dims = (-2, -1)  # x.shape [B, C, Nlat, Nlon]
        obs_values = torch.where(self.obs_mask, self.obs_values, torch.zeros_like(x))

        # Fraction observed (per batch, per channel)
        m = self.obs_mask.float().mean(dim=spatial_dims, keepdim=True)  # [B,C,1,1]

        # Means
        s_all_before = x.mean(dim=spatial_dims, keepdim=True)
        mask_count = self.obs_mask.float().sum(dim=spatial_dims, keepdim=True)
        s_mask_obs = torch.where(
            mask_count > 0,
            (obs_values * self.obs_mask.float()).sum(dim=spatial_dims, keepdim=True) / mask_count,
            s_all_before,
        )

        unmask = ~self.obs_mask
        unmask_count = unmask.float().sum(dim=spatial_dims, keepdim=True)
        s_unmask_before = torch.where(
            unmask_count > 0,
            (x * unmask.float()).sum(dim=spatial_dims, keepdim=True) / unmask_count,
            s_all_before,
        )

        denom = (1.0 - m).clamp(min=1e-12)
        s_unmask_new = (s_all_before - m * s_mask_obs) / denom

        # Mean correction
        a_add = s_unmask_new - s_unmask_before

        # Compute variances
        v_before = x.var(dim=spatial_dims, unbiased=False, keepdim=True)

        v_mask_obs = torch.where(
            mask_count > 0,
            ((obs_values - s_mask_obs) ** 2 * self.obs_mask.float()).sum(dim=spatial_dims, keepdim=True) / mask_count,
            v_before,
        )

        v_unmask = torch.where(
            unmask_count > 0,
            (((x + a_add) - s_unmask_new) ** 2 * unmask.float()).sum(dim=spatial_dims, keepdim=True) / unmask_count,
            v_before,
        )

        # Scale factor for unmasked region to preserve total variance
        denom_var = ((1.0 - m) * v_unmask).clamp(min=1e-12)
        numer_var = (v_before - m * v_mask_obs).clamp(min=0.0)
        s_scale = torch.sqrt(numer_var / denom_var)

        # Apply scaling only to unmasked region
        x_final = torch.where(
            self.obs_mask,
            self.obs_values,
            s_unmask_new + s_scale * (x - s_unmask_before),
        )

        return x_final

    def masking_total_column_average_test(self, x):
        """
        Constrain vertical profile adjusting (multiplicative) column-averaged observations (XCO2).
        
        Args:
            x: [B, C, Nlat, Nlon] - the C-level CO2 field

        ak: [B, C, Nlat, Nlon] - averaging kernel for each level
        xco2_prior: [B, 1, Nlat, Nlon] - prior XCO2 column observations
        co2_profile_prior: [B, C, Nlat, Nlon] - prior CO2 profile
        obs_values: [B, 1, Nlat, Nlon] - XCO2 column observations
        obs_mask: [B, 1, Nlat, Nlon] - spatial mask
        We compute: x_averaged = xco2_prior + sum(ak * (x - co2_profile_prior)) over levels
        Then scale each level: x_new = x * (obs_values / x_averaged)
        """
        if self.ak is None:
            self.ak = torch.ones(x.shape, device=x.device)
        h = self.pressure_weights if self.pressure_weights is not None else 1.0 / x.shape[1]
        x_physical = x * self.target_std + self.target_mean
        x_averaged_physical = self.xco2_prior + (h * self.ak * (x_physical - self.co2_profile_prior)).sum(dim=1, keepdim=True)  # [B 1 Nlat Nlon]
        obs_physical = self.obs_values * self.obs_std + self.obs_mean

        scale_factor = (obs_physical.detach() / x_averaged_physical.clamp(min=1e-12))  # [B 1 Nlat Nlon]
        x_scaled_physical = scale_factor * x_physical  # [B C Nlat Nlon]
        x_scaled = (x_scaled_physical - self.target_mean) / self.target_std

        x_masked = torch.where(
            self.obs_mask,
            x_scaled,
            x
        )
        return x_masked

    def masking_total_column_average_test_basic(self, x):
        if self.ak is None:
            self.ak = torch.ones(x.shape, device=x.device)

        h = self.pressure_weights if self.pressure_weights is not None else 1.0 / x.shape[1]
        h_ak = h * self.ak
        C = x.shape[1]
        xco2 = self.compute_xco2(x)  # [B 1 Nlat Nlon]

        correction = (self.obs_values.detach() / C - h_ak * x).sum(dim=1, keepdim=True)  # [B 1 Nlat Nlon]

        distributed_correction = (1 / h_ak.clamp(min=1e-12)) * correction  # [B C Nlat Nlon]

        x_masked = torch.where(
            self.obs_mask,
            x + distributed_correction,
            x
        )
        return x_masked
    
    def masking_total_column_average_simple(self, x):
        if self.ak is None:
            self.ak = torch.ones(x.shape, device=x.device)

        h = self.pressure_weights if self.pressure_weights is not None else 1.0 / x.shape[1]
        h_ak = h * self.ak
        h_ak_sum = h_ak.sum(dim=1, keepdim=True).clamp(min=1e-12)

        xco2 = self.compute_xco2(x)  # [B 1 Nlat Nlon]
        column_error = self.obs_values.detach() - xco2  # [B 1 Nlat Nlon]

        # Apply uniform correction across all levels to satisfy column constraint.
        # Mathematically: if sum_i(h_i * a_i * delta_x_i) = column_error_physical,
        # and we choose delta_x_i = constant for all i, then:
        # constant = column_error_physical / sum_i(h_i * a_i)
        # In normalized space, this becomes:
        # constant_normalized = column_error_normalized / h_ak_sum (where h_ak_sum is in normalized space)
        # We distribute by level importance: delta_x_i = (h_i * a_i / h_ak_sum) * (column_error / h_ak_sum)
        # which simplifies to: constant across levels divided by h_ak_sum
        distributed_correction = column_error / h_ak_sum  # [B C Nlat Nlon] - uniform per level

        x_masked = torch.where(
            self.obs_mask,
            x + distributed_correction,
            x
        )

        return x_masked

    def masking_total_column_average_add(self, x):
        """
        Constrain vertical profile adjusting (additive) column-averaged observations (XCO2).

        Args:
            x: [B, C, Nlat, Nlon] - the C-level CO2 field
        """

        if self.ak is None:
            self.ak = torch.ones(x.shape, device=x.device)

        h = self.pressure_weights if self.pressure_weights is not None else 1.0 / x.shape[1]

        correction = (h * (self.obs_values.detach() - self.xco2_prior) - (h * self.ak * (x - self.co2_profile_prior))).sum(dim=1, keepdim=True)  # [B 1 Nlat Nlon]

        h_ak = h * self.ak
        distributed_correction = (h_ak / (h_ak ** 2).sum(dim=1, keepdim=True).clamp(min=1e-12)) * correction  # [B C Nlat Nlon]

        x_masked = torch.where(
            self.obs_mask,
            x + distributed_correction,
            x
        )

        return x_masked

    def masking_total_column_average_mult(self, x):
        """
        Constrain vertical profile adjusting (multiplicative) column-averaged observations (XCO2).

        Args:
            x: [B, C, Nlat, Nlon] - the C-level CO2 field
        """

        if self.ak is None:
            self.ak = torch.ones(x.shape, device=x.device)
        h = self.pressure_weights if self.pressure_weights is not None else 1.0 / x.shape[1]
        x_physical = x * self.target_std + self.target_mean
        obs_physical = self.obs_values * self.obs_std + self.obs_mean

        xco2_physical = self.xco2_prior + (h * self.ak * (x_physical - self.co2_profile_prior)).sum(dim=1, keepdim=True)  # [B 1 Nlat Nlon]

        h_ak = h * self.ak
        correction = (self.xco2_prior * h / h_ak.sum(dim=1, keepdim=True).clamp(min=1e-12) + x_physical - self.co2_profile_prior) * (obs_physical / xco2_physical - 1)  # [B C Nlat Nlon]

        x_scaled_physical = x_physical + correction  # [B C Nlat Nlon]
        x_scaled = (x_scaled_physical - self.target_mean) / self.target_std

        x_masked = torch.where(
            self.obs_mask,
            x_scaled,
            x
        )

        return x_masked
    
    def masking_total_column_average_simple_unitary(self, x):
        if self.ak is None:
            self.ak = torch.ones(x.shape, device=x.device)

        h = self.pressure_weights if self.pressure_weights is not None else 1.0 / x.shape[1]
        h_ak = h * self.ak
        h_ak_sum = h_ak.sum(dim=1, keepdim=True).clamp(min=1e-12)
        xco2 = self.compute_xco2(x)  # [B 1 Nlat Nlon]

        # Apply uniform correction across all levels to satisfy column constraint.
        # Same logic as masking_total_column_average_simple:
        # distributed_correction should be the same for all C levels
        column_error = self.obs_values.detach() - xco2  # [B 1 Nlat Nlon]
        distributed_correction = column_error / h_ak_sum  # [B 1 Nlat Nlon]

        x_masked = torch.where(
            self.obs_mask,
            x + distributed_correction,
            x
        )
        return x_masked

    def masking_total_column_average_simple_diag(self, x):
        if self.ak is None:
            self.ak = torch.ones(x.shape, device=x.device)

        h = self.pressure_weights if self.pressure_weights is not None else 1.0 / x.shape[1]
        h_ak = h * self.ak
        h_ak_sum = h_ak.sum(dim=1, keepdim=True).clamp(min=1e-12)
        xco2 = self.compute_xco2(x)  # [B 1 Nlat Nlon]

        # Apply uniform correction across all levels to satisfy column constraint.
        # Same logic as masking_total_column_average_simple
        column_error = self.obs_values.detach() - xco2  # [B 1 Nlat Nlon]
        distributed_correction = column_error / h_ak_sum  # [B 1 Nlat Nlon]

        x_masked = torch.where(
            self.obs_mask,
            x + distributed_correction,
            x
        )
        return x_masked

    def masking_total_column_average_simple_invdiag(self, x):
        if self.ak is None:
            self.ak = torch.ones(x.shape, device=x.device)

        h = self.pressure_weights if self.pressure_weights is not None else 1.0 / x.shape[1]
        h_ak = h * self.ak
        h_ak_sum = h_ak.sum(dim=1, keepdim=True).clamp(min=1e-12)
        xco2 = self.compute_xco2(x)  # [B 1 Nlat Nlon]

        # Apply uniform correction across all levels to satisfy column constraint.
        # Same logic as masking_total_column_average_simple
        column_error = self.obs_values.detach() - xco2  # [B 1 Nlat Nlon]
        distributed_correction = column_error / h_ak_sum  # [B 1 Nlat Nlon]

        x_masked = torch.where(
            self.obs_mask,
            x + distributed_correction,
            x
        )
        return x_masked


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
            ):

        self.submodel = MODELS[submodel](**model_kwargs)
        self.return_intermediates = return_intermediates
        self.generating = generating
        self.method = method
        self.nlev = nlev
        self.step_size = step_size
        self.generate_kwargs = generate_kwargs if generate_kwargs is not None else {}
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.target_vars = self.submodel.target_vars # Here target_vars[0] is supposed to be "co2massmix"

    def forward(self, batch):
        if not hasattr(self, "generate_kwargs"):
            self.generate_kwargs = {}
        if self.training:
            x_out, dx_t = self.training_forward(batch)
            preds = self.postprocess_outputs(x_out, batch, denormalize=False)
            preds["dx_t"] = dx_t.permute(0, 2, 3, 1).reshape(*preds[self.target_vars[0]].shape)  # [B N C]
            return preds
        elif self.generating:
            x_in = self.preprocess_inputs(batch)
            all_levels = x_in[:, :self.nlev*len(self.target_vars), :, :]  # [B C Nlat Nlon]
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
                x_in, x_init,
                masking_config=masking_config,
                generate_kwargs=self.generate_kwargs
            )
            x_out = trajectory[-1,...]
            if self.return_intermediates:
                sol = self.postprocess_outputs(x_out, batch)
                T, B, C, Nlat, Nlon = trajectory.shape
                trajectory = trajectory.permute(0, 1, 3, 4, 2)  # [T B Nlat Nlon C]
                trajectory = trajectory.reshape(T, B, Nlat*Nlon, C)  # [T B Nlat*Nlon C]
                sol["trajectory"] = trajectory
            return sol
        else:
            return super().forward(batch)

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

        # sample time t \in [0,1], [B] -> [B 1 Nlat Nlon]
        B, _, nlat, nlon = x_in.shape
        t = torch.rand(B, device=x_in.device)

        # sample path
        path_sample = self.path.sample(
            t=t,
            x_0=x_0,
            x_1=x_1_normalized
        )

        # sample x_t from the path
        x_t = path_sample.x_t
        # x_t = path_sample.x_t + x_0 (simplified version)
        # this requires target_vars to be first in input_vars and all nlev-dimensional
        x_in[:, :self.nlev*len(self.target_vars), :, :] = x_t  # [B C Nlat Nlon] 

        t_expanded = path_sample.t.view(-1, 1, 1, 1).expand(B, 1, nlat, nlon)
        x_in = torch.cat([x_in, t_expanded], dim=1)  # [B C+1 Nlat Nlon]

        x_out = self.submodel.model(x_in)

        dx_t = path_sample.dx_t

        # # access scheduler for affine path
        # scheduler_out = self.path.scheduler(t)
        # d_sigma_t = scheduler_out.d_sigma_t.view(-1, 1, 1, 1)
        # d_alpha_t = scheduler_out.d_alpha_t.view(-1, 1, 1, 1)

        # x_out = (x_out - d_sigma_t * x_0) / d_alpha_t # to adjust for loss function definition

        return x_out, dx_t #, x_1_normalized # return {self.target_vars[0]: x_out}

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
            "obs_mask": obs_mask,        # [B 1 Nlat Nlon]
            "obs_values": obs_values,    # [B 1 Nlat Nlon]
            "obs_mean": obs_mean,        # [B 1 1 1]
            "obs_std": obs_std,          # [B 1 1 1]
            "target_mean": target_mean,  # [B 1 1 1]
            "target_std": target_std,    # [B 1 1 1]
            "ak": ak,                    # [B C Nlat Nlon]
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

    # inference_forward
    def inference_forward(
            self,
            x_in, x_init,
            masking_config=None,
            generate_kwargs=None
            ):
        if generate_kwargs is None:
            generate_kwargs = {}

        refine_start = generate_kwargs.get("refine_start", 1.0)
        steps = generate_kwargs.get("steps", 11)

        # get timesteps for integration [T]
        if refine_start < 1.0:
            coarse = torch.linspace(0, refine_start, steps=steps, device=x_init.device)[:-1]
            fine = torch.linspace(refine_start, 1.0, steps=steps, device=x_init.device)
            time_grid = torch.cat([coarse, fine[1:]])
        else:
            time_grid = torch.linspace(0, 1, steps=steps-1, device=x_init.device) 
        masking_config["time_grid"] = time_grid

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
        trajectory = solver.sample(time_grid=time_grid,
                            x_init=x_init, method=self.method,
                            step_size=self.step_size,
                            return_intermediates=self.return_intermediates
        )  # [T B C Nlat Nlon]
        return trajectory
