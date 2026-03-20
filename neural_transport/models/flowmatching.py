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
from neural_transport.tools.guidance import XCO2Guidance
from neural_transport.tools.developement import _print_stats_torch


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
        self.time_grid = masking_config.get("time_grid", None)

        self.masking_time = generate_kwargs.get("masking_time", None)
        self.t_threshold = generate_kwargs.get("t_threshold", 0.9)
        self.masking_method = generate_kwargs.get("masking_method", "interpolate")
        
        # DPS guidance parameters
        self.use_dps_guidance = generate_kwargs.get("use_dps_guidance", False)
        self.guidance_scale = generate_kwargs.get("guidance_scale", 1.0)
        self.guidance_start_t = generate_kwargs.get("guidance_start_t", 0.0)
        self.guidance_end_t = generate_kwargs.get("guidance_end_t", 1.0)
        self.guidance_loss_type = generate_kwargs.get("loss_type", "mse")
        
        # Initialize guidance if requested
        if self.use_dps_guidance:
            self.guidance = XCO2Guidance(
                obs_mask=self.obs_mask,
                obs_values=self.obs_values,
                averaging_kernel=self.ak,
                guidance_scale=self.guidance_scale,
                loss_type=self.guidance_loss_type,
            )
            print("\nDEBUG DPS guidance enabled")
        else:
            self.guidance = None

        print("\nDEBUG MaskedVelocityWrapper init")
        obs_valid = self.obs_values[~torch.isnan(self.obs_values)]
        if obs_valid.numel() > 0:
            print(f"  obs_values valid stats: min={obs_valid.min():.6f}, max={obs_valid.max():.6f}")
        print(f"  obs_mean: {self.obs_mean.flatten()[0]:.6f}")
        print(f"  obs_std: {self.obs_std.flatten()[0]:.6f}")
        print(f"  target_mean: {self.target_mean.flatten()[0]:.6f}")
        print(f"  target_std: {self.target_std.flatten()[0]:.6f}")
        if self.use_dps_guidance:
            print(f"  DPS guidance_scale: {self.guidance_scale}")
            print(f"  DPS guidance_start_t: {self.guidance_start_t}")
            print(f"  DPS guidance_end_t: {self.guidance_end_t}")

    def forward(self, x, t):
        if torch.isnan(x).any():
            print(f"\nDEBUG MaskedVelocityWrapper.forward: INPUT x has NaN at t={t}")
            print(f"  x NaN count: {torch.isnan(x).sum()}")
        x_masked = self.apply_masking(x, t)
        if torch.isnan(x_masked).any():
            print(f"\nDEBUG MaskedVelocityWrapper.forward: x_masked has NaN at t={t}")
            print(f"  x_masked NaN count: {torch.isnan(x_masked).sum()}")
            print(f"  masking_method: {self.masking_method}")

        x_effective = self.apply_temporal_weighting(x, x_masked, t)

        dt = self.compute_dt(t)
        print(f"  t={t.item()}")
        print(f"  dt={dt}")
        
        # f(x,t) base velocity
        v_base = super().forward(x_effective, t)
        
        if self.use_dps_guidance:
            v_base = self.apply_dps_guidance(
                x=x_effective,
                t=t,
                v=v_base,
            )
        
        # dxt = (x_effective - x)/dt + f(x_effective, t)
        dtx = (x_effective - x) / dt + v_base
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(dtx[0].mean(dim=0).detach().cpu().numpy())
        # plt.subplot(1,2,2)
        # plt.imshow(((x_effective - x) / dt)[0].mean(dim=0).detach().cpu().numpy())
        # plt.colorbar()
        # plt.title(f"dtx at t={t}")
        # plt.savefig(f"dtx_t{int(t.item()*100)}.png")
        # plt.close()
        # dxt = f(x,t)
        # dxt = torch.where(self.obs_mask, self.obs_values - x, super().forward(x, t))
        if torch.isnan(dtx).any():
            print(f"\nDEBUG MaskedVelocityWrapper.forward: OUTPUT dtx has NaN at t={t}")
            print(f"  dtx NaN count: {torch.isnan(dtx).sum()}")
            print(f"  (x_effective - x)/dt stats: min={(x_effective - x).min()/dt:.6f}, max={(x_effective - x).max()/dt:.6f}")

        return dtx

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
        masking_time = self.masking_time
        t_threshold = self.t_threshold
        if masking_time == "smooth_late_masking":
            mask_weight = torch.sigmoid((t - t_threshold) * 20.0).view(-1, 1, 1, 1)
        elif masking_time == "step_late_masking":
            mask_weight = (t >= t_threshold).float().view(-1, 1, 1, 1)
        elif masking_time == "smooth_early_masking":
            mask_weight = torch.sigmoid((t_threshold - t) * 20.0).view(-1, 1, 1, 1)
        elif masking_time == "step_early_masking":
            mask_weight = (t < t_threshold).float().view(-1, 1, 1, 1)
        else:
            mask_weight = 1.0
        
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

    def apply_dps_guidance(self, x, t, v):
        t_val = t.item()
        if self.guidance_start_t <= t_val <= self.guidance_end_t:
            # Create fresh copy for gradient computation
            x_guided = x.detach().clone()
            x_guided.requires_grad_(True)

            # Recompute velocity with gradients
            v_guided = super().forward(x_guided, t)
            
            # Estimate denoised prediction: x(1) ≈ x(t) + (1-t)*v(x,t)
            x_denoised = x_guided + (1 - t_val) * v_guided
            
            # Get guidance gradient
            grad = self.guidance.get_gradient(x_guided, x_denoised, retain_graph=False)
            
            # Apply guidance: steer velocity toward observations
            v_base = v - (1 - t_val) * grad.detach()
            print(f"\nDEBUG  DPS guidance applied: grad_norm={grad.norm().item():.6f}")
        return v_base

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
        print("\nDEBUG masking_total_column_average_test")
        print(f"  x shape: {x.shape}, has NaN: {torch.isnan(x).any()}")
        print(f"    x stats: min={x.min():.6f}, max={x.max():.6f}")
        print(f"    x norm: {x.norm(dim=(2,3)).mean()}")
        if self.ak is None:
            self.ak = torch.ones(x.shape, device=x.device)
        x_physical = x * self.target_std + self.target_mean
        print(f"  x_physical shape: {x_physical.shape}, has NaN: {torch.isnan(x_physical).any()}")
        print(f"    x_physical stats: min={x_physical.min():.6f}, max={x_physical.max():.6f}")
        print(f"    x_physical norm: {x_physical.norm(dim=(2,3)).mean()}")
        if torch.isnan(x_physical).any():
            print(f"  target_std has NaN: {torch.isnan(self.target_std).any()}")
            print(f"  target_mean has NaN: {torch.isnan(self.target_mean).any()}")
        x_averaged_physical = self.xco2_prior + (self.ak * (x_physical - self.co2_profile_prior)).sum(dim=1, keepdim=True)  # [B 1 Nlat Nlon]
        print("  Using prior correction")
        print(f"    xco2_prior shape {self.xco2_prior.shape}, has NaN: {torch.isnan(self.xco2_prior).any()}")
        print(f"    co2_profile_prior shape {self.co2_profile_prior.shape}, has NaN: {torch.isnan(self.co2_profile_prior).any()}")
        print(f"  x_averaged_physical shape {x_averaged_physical.shape}, has NaN: {torch.isnan(x_averaged_physical).any()}")
        x_ap_valid = x_averaged_physical[~torch.isnan(x_averaged_physical)]
        if x_ap_valid.numel() > 0:
            print(f"    x_averaged_physical valid stats: min={x_ap_valid.min():.6f}, max={x_ap_valid.max():.6f}")
        obs_physical = self.obs_values * self.obs_std + self.obs_mean
        print(f"  obs_physical shape {obs_physical.shape}, has NaN: {torch.isnan(obs_physical).any()}")
        if torch.isnan(obs_physical).any():
            print(f"    obs_values has NaN: {torch.isnan(self.obs_values).any()}")
            print(f"    obs_std has NaN: {torch.isnan(self.obs_std).any()}")
            print(f"    obs_mean has NaN: {torch.isnan(self.obs_mean).any()}")
        op_valid = obs_physical[~torch.isnan(obs_physical)]
        if op_valid.numel() > 0:
            print(f"    obs_physical valid stats: min={op_valid.min():.6f}, max={op_valid.max():.6f}")
        scale_factor = (obs_physical.detach() / x_averaged_physical.clamp(min=1e-12))  # [B 1 Nlat Nlon]
        print(f"  scale_factor shape {scale_factor.shape}, has NaN: {torch.isnan(scale_factor).any()}")
        if not torch.isnan(scale_factor).any():
            valid_sf = scale_factor[self.obs_mask]
            print(f"    scale_factor shape {valid_sf.shape}, has NaN on obs_mask: {torch.isnan(valid_sf).any()}")
            print(f"      scale_factor stats: min={valid_sf.min():.6f}, max={valid_sf.max():.6f}")
            if valid_sf.numel() > 0:
                print(f"      scale_factor[obs_mask] stats: min={valid_sf.min():.6f}, max={valid_sf.max():.6f}")
        x_scaled_physical = scale_factor * x_physical  # [B C Nlat Nlon]
        print(f"  x_scaled_physical shape {x_scaled_physical.shape}, has NaN: {torch.isnan(x_scaled_physical).any()}")
        x_sp_valid = x_scaled_physical[~torch.isnan(x_scaled_physical)]
        if x_sp_valid.numel() > 0:
            print(f"    x_scaled_physical valid stats: min={x_sp_valid.min():.6f}, max={x_sp_valid.max():.6f}")
        x_scaled = (x_scaled_physical - self.target_mean) / self.target_std
        print(f"  x_scaled shape {x_scaled.shape}, has NaN: {torch.isnan(x_scaled).any()}")
        x_s_valid = x_scaled[~torch.isnan(x_scaled)]
        if x_s_valid.numel() > 0:
            print(f"    x_scaled valid stats: min={x_s_valid.min():.6f}, max={x_s_valid.max():.6f}")
        x_masked = torch.where(
            self.obs_mask,
            x_scaled,
            x
        )
        print(f"  x_masked (final) shape {x_masked.shape}, has NaN: {torch.isnan(x_masked).any()}")
        print(f"    x_masked (final) stats: min={x_masked.min():.6f}, max={x_masked.max():.6f}")
        print(f"    x_masked (final) norm: {x_masked.norm(dim=(2,3)).mean()}")
        return x_masked

    def masking_total_column_average_test_basic(self, x):
        print("\nDEBUG masking_total_column_average_test_basic")
        print(f"  x shape: {x.shape}, has NaN: {torch.isnan(x).any()}")
        print(f"    x stats: min={x.min():.6f}, max={x.max():.6f}")
        print(f"    x norm: {x.norm(dim=(2,3)).mean()}")
        if self.ak is None:
            self.ak = torch.ones(x.shape, device=x.device)
        
        C = x.shape[1]
        xco2 = self.obs_values.sum(dim=1, keepdim=True)  # [B 1 Nlat Nlon]
        print(f"  xco2 shape: {xco2.shape}, has NaN: {torch.isnan(xco2).any()}")
        xco2_valid = xco2[~torch.isnan(xco2)]
        if xco2_valid.numel() > 0:
            print(f"    xco2 valid stats: min={xco2_valid.min():.6f}, max={xco2_valid.max():.6f}")

        correction = 1/C * (xco2 / C - self.ak * x).sum(dim=1, keepdim=True)  # [B 1 Nlat Nlon]
        print(f"  correction shape: {correction.shape}, has NaN: {torch.isnan(correction).any()}")
        correction_valid = correction[~torch.isnan(correction)]
        if correction_valid.numel() > 0:
            print(f"    correction valid stats: min={correction_valid.min():.6f}, max={correction_valid.max():.6f}")

        distributed_correction = 1/self.ak * correction  # [B C Nlat Nlon]
        print(f"  distributed_correction shape: {distributed_correction.shape}, has NaN: {torch.isnan(distributed_correction).any()}")
        distributed_correction_valid = distributed_correction[~torch.isnan(distributed_correction)]
        if distributed_correction_valid.numel() > 0:
            print(f"    distributed_correction valid stats: min={distributed_correction_valid.min():.6f}, max={distributed_correction_valid.max():.6f}")

        x_masked = torch.where(
            self.obs_mask,
            x + distributed_correction,
            x
        )
        print(f"  x_masked shape {x_masked.shape}, has NaN: {torch.isnan(x_masked).any()}")
        print(f"    x_masked stats: min={x_masked.min():.6f}, max={x_masked.max():.6f}")
        print(f"    x_masked norm: {x_masked.norm(dim=(2,3)).mean()}")
        return x_masked
    
    def masking_total_column_average_simple(self, x):
        print("\nDEBUG masking_total_column_average_simple")
        print(f"  x shape: {x.shape}, has NaN: {torch.isnan(x).any()}")
        print(f"    x stats: min={x.min():.6f}, max={x.max():.6f}")
        print(f"    x norm: {x.norm(dim=(2,3)).mean()}")
        if self.ak is None:
            self.ak = torch.ones(x.shape, device=x.device)
        
        ak_sum = self.ak.sum(dim=1, keepdim=True).clamp(min=1e-12)
        xco2 = (self.ak * x).sum(dim=1, keepdim=True) / ak_sum  # [B 1 Nlat Nlon]
        print(f"  xco2 shape: {xco2.shape}, has NaN: {torch.isnan(xco2).any()}")
        xco2_valid = xco2[~torch.isnan(xco2)]
        if xco2_valid.numel() > 0:
            print(f"    xco2 valid stats: min={xco2_valid.min():.6f}, max={xco2_valid.max():.6f}")
                
        column_error = self.obs_values.detach() - xco2  # [B 1 Nlat Nlon]
        
        ak_normalized = self.ak / ak_sum  # [B C Nlat Nlon]
        distributed_correction = ak_normalized * column_error  # [B C Nlat Nlon]
        print(f"  distributed_correction shape: {distributed_correction.shape}, has NaN: {torch.isnan(distributed_correction).any()}")
        distributed_correction_valid = distributed_correction[~torch.isnan(distributed_correction)]
        if distributed_correction_valid.numel() > 0:
            print(f"    distributed_correction valid stats: min={distributed_correction_valid.min():.6f}, max={distributed_correction_valid.max():.6f}")

        x_masked = torch.where(
            self.obs_mask,
            x + distributed_correction,
            x
        )
        print(f"  x_masked shape {x_masked.shape}, has NaN: {torch.isnan(x_masked).any()}")
        print(f"    x_masked stats: min={x_masked.min():.6f}, max={x_masked.max():.6f}")
        print(f"    x_masked norm: {x_masked.norm(dim=(2,3)).mean()}")
        return x_masked

    def masking_total_column_average_add(self, x):
        """
        Constrain vertical profile adjusting (additive) column-averaged observations (XCO2).
        
        Args:
            x: [B, C, Nlat, Nlon] - the C-level CO2 field
        """      
        print("\nDEBUG masking_total_column_average_add")
        print(f"  x shape: {x.shape}, has NaN: {torch.isnan(x).any()}")
        print(f"    x stats: min={x.min():.6f}, max={x.max():.6f}")
        print(f"    x norm: {x.norm(dim=(2,3)).mean()}")
        if self.ak is None:
            self.ak = torch.ones(x.shape, device=x.device)
        
        C = x.shape[1]

        correction = (1 / C * (self.obs_values.detach() - self.xco2_prior) - (self.ak * (x - self.co2_profile_prior))).sum(dim=1, keepdim=True)  # [B 1 Nlat Nlon]
        print(f"  correction shape: {correction.shape}, has NaN: {torch.isnan(correction).any()}")
        correction_valid = correction[~torch.isnan(correction)]
        if correction_valid.numel() > 0:
            print(f"    correction valid stats: min={correction_valid.min():.6f}, max={correction_valid.max():.6f}")
        distributed_correction = 1/self.ak * correction  # [B C Nlat Nlon]
        print(f"  distributed_correction shape: {distributed_correction.shape}, has NaN: {torch.isnan(distributed_correction).any()}")
        distributed_correction_valid = distributed_correction[~torch.isnan(distributed_correction)]
        if distributed_correction_valid.numel() > 0:
            print(f"    distributed_correction valid stats: min={distributed_correction_valid.min():.6f}, max={distributed_correction_valid.max():.6f}")

        x_masked = torch.where(
            self.obs_mask,
            x + distributed_correction,
            x
        )
        print(f"  x_masked shape {x_masked.shape}, has NaN: {torch.isnan(x_masked).any()}")
        print(f"    x_masked stats: min={x_masked.min():.6f}, max={x_masked.max():.6f}")
        print(f"    x_masked norm: {x_masked.norm(dim=(2,3)).mean()}")

        return x_masked

    def masking_total_column_average_mult(self, x):
        """
        Constrain vertical profile adjusting (multiplicative) column-averaged observations (XCO2).
        
        Args:
            x: [B, C, Nlat, Nlon] - the C-level CO2 field
        """
        print("\nDEBUG masking_total_column_average_mult")
        print(f"  x shape: {x.shape}, has NaN: {torch.isnan(x).any()}")
        print(f"    x stats: min={x.min():.6f}, max={x.max():.6f}")
        print(f"    x norm: {x.norm(dim=(2,3)).mean()}")
        if self.ak is None:
            self.ak = torch.ones(x.shape, device=x.device)
        x_physical = x * self.target_std + self.target_mean
        print(f"  x_physical shape: {x_physical.shape}, has NaN: {torch.isnan(x_physical).any()}")
        print(f"    x_physical stats: min={x_physical.min():.6f}, max={x_physical.max():.6f}")
        print(f"    x_physical norm: {x_physical.norm(dim=(2,3)).mean()}")
        obs_physical = self.obs_values * self.obs_std + self.obs_mean
        C = x.shape[1]
        print(f"  obs_physical shape {obs_physical.shape}, has NaN: {torch.isnan(obs_physical).any()}")
        if torch.isnan(obs_physical).any():
            print(f"    obs_values has NaN: {torch.isnan(self.obs_values).any()}")
            print(f"    obs_std has NaN: {torch.isnan(self.obs_std).any()}")
            print(f"    obs_mean has NaN: {torch.isnan(self.obs_mean).any()}")
        op_valid = obs_physical[~torch.isnan(obs_physical)]
        if op_valid.numel() > 0:
            print(f"    obs_physical valid stats: min={op_valid.min():.6f}, max={op_valid.max():.6f}")

        xco2_physical = self.xco2_prior + (self.ak * (x_physical - self.co2_profile_prior)).sum(dim=1, keepdim=True)  # [B 1 Nlat Nlon]

        correction = (self.xco2_prior / (C * self.ak) + x_physical - self.co2_profile_prior) * (obs_physical / xco2_physical - 1)  # [B C Nlat Nlon]

        x_scaled_physical = x_physical + correction  # [B C Nlat Nlon]
        x_scaled = (x_scaled_physical - self.target_mean) / self.target_std

        x_masked = torch.where(
            self.obs_mask,
            x_scaled,
            x
        )
        print(f"  x_masked shape {x_masked.shape}, has NaN: {torch.isnan(x_masked).any()}")
        print(f"    x_masked stats: min={x_masked.min():.6f}, max={x_masked.max():.6f}")
        print(f"    x_masked norm: {x_masked.norm(dim=(2,3)).mean()}")

        return x_masked
    
    def masking_total_column_average_simple_unitary(self, x):
        print("\nDEBUG masking_total_column_average_simple_unitary")
        print(f"  x shape: {x.shape}, has NaN: {torch.isnan(x).any()}")
        print(f"    x stats: min={x.min():.6f}, max={x.max():.6f}")
        print(f"    x norm: {x.norm(dim=(2,3)).mean()}")
        if self.ak is None:
            self.ak = torch.ones(x.shape, device=x.device)

        ak_sum = self.ak.sum(dim=1, keepdim=True).clamp(min=1e-12)
        xco2 = (self.ak * x).sum(dim=1, keepdim=True)  # [B 1 Nlat Nlon]
        print(f"  xco2 shape: {xco2.shape}, has NaN: {torch.isnan(xco2).any()}")
        xco2_valid = xco2[~torch.isnan(xco2)]
        if xco2_valid.numel() > 0:
            print(f"    xco2 valid stats: min={xco2_valid.min():.6f}, max={xco2_valid.max():.6f}")

        column_error = (self.obs_values.detach() - xco2) / ak_sum  # [B 1 Nlat Nlon]

        unitary = torch.ones(x.shape, device=x.device)  # [B C Nlat Nlon]
        distributed_correction = unitary * column_error  # [B C Nlat Nlon]
        print(f"  distributed_correction shape: {distributed_correction.shape}, has NaN: {torch.isnan(distributed_correction).any()}")
        distributed_correction_valid = distributed_correction[~torch.isnan(distributed_correction)]
        if distributed_correction_valid.numel() > 0:
            print(f"    distributed_correction valid stats: min={distributed_correction_valid.min():.6f}, max={distributed_correction_valid.max():.6f}")

        x_masked = torch.where(
            self.obs_mask,
            x + distributed_correction,
            x
        )
        print(f"  x_masked shape {x_masked.shape}, has NaN: {torch.isnan(x_masked).any()}")
        print(f"    x_masked stats: min={x_masked.min():.6f}, max={x_masked.max():.6f}")
        print(f"    x_masked norm: {x_masked.norm(dim=(2,3)).mean()}")
        return x_masked

    def masking_total_column_average_simple_diag(self, x):
        print("\nDEBUG masking_total_column_average_simple_diag")
        print(f"  x shape: {x.shape}, has NaN: {torch.isnan(x).any()}")
        print(f"    x stats: min={x.min():.6f}, max={x.max():.6f}")
        print(f"    x norm: {x.norm(dim=(2,3)).mean()}")
        if self.ak is None:
            self.ak = torch.ones(x.shape, device=x.device)

        ak2_sum = (self.ak ** 2).sum(dim=1, keepdim=True).clamp(min=1e-12)
        xco2 = (self.ak * x).sum(dim=1, keepdim=True)  # [B 1 Nlat Nlon]
        print(f"  xco2 shape: {xco2.shape}, has NaN: {torch.isnan(xco2).any()}")
        xco2_valid = xco2[~torch.isnan(xco2)]
        if xco2_valid.numel() > 0:
            print(f"    xco2 valid stats: min={xco2_valid.min():.6f}, max={xco2_valid.max():.6f}")

        column_error = (self.obs_values.detach() - xco2) / ak2_sum  # [B 1 Nlat Nlon]

        distributed_correction = self.ak * column_error  # [B C Nlat Nlon]
        print(f"  distributed_correction shape: {distributed_correction.shape}, has NaN: {torch.isnan(distributed_correction).any()}")
        distributed_correction_valid = distributed_correction[~torch.isnan(distributed_correction)]
        if distributed_correction_valid.numel() > 0:
            print(f"    distributed_correction valid stats: min={distributed_correction_valid.min():.6f}, max={distributed_correction_valid.max():.6f}")

        x_masked = torch.where(
            self.obs_mask,
            x + distributed_correction,
            x
        )
        print(f"  x_masked shape {x_masked.shape}, has NaN: {torch.isnan(x_masked).any()}")
        print(f"    x_masked stats: min={x_masked.min():.6f}, max={x_masked.max():.6f}")
        print(f"    x_masked norm: {x_masked.norm(dim=(2,3)).mean()}")
        return x_masked

    def masking_total_column_average_simple_invdiag(self, x):
        print("\nDEBUG masking_total_column_average_simple_invdiag")
        print(f"  x shape: {x.shape}, has NaN: {torch.isnan(x).any()}")
        print(f"    x stats: min={x.min():.6f}, max={x.max():.6f}")
        print(f"    x norm: {x.norm(dim=(2,3)).mean()}")
        if self.ak is None:
            self.ak = torch.ones(x.shape, device=x.device)

        xco2 = (self.ak * x).sum(dim=1, keepdim=True)  # [B 1 Nlat Nlon]
        print(f"  xco2 shape: {xco2.shape}, has NaN: {torch.isnan(xco2).any()}")
        xco2_valid = xco2[~torch.isnan(xco2)]
        if xco2_valid.numel() > 0:
            print(f"    xco2 valid stats: min={xco2_valid.min():.6f}, max={xco2_valid.max():.6f}")

        column_error = self.obs_values.detach() - xco2  # [B 1 Nlat Nlon]

        distributed_correction = (1/self.ak) * column_error  # [B C Nlat Nlon]
        print(f"  distributed_correction shape: {distributed_correction.shape}, has NaN: {torch.isnan(distributed_correction).any()}")
        distributed_correction_valid = distributed_correction[~torch.isnan(distributed_correction)]
        if distributed_correction_valid.numel() > 0:
            print(f"    distributed_correction valid stats: min={distributed_correction_valid.min():.6f}, max={distributed_correction_valid.max():.6f}")

        x_masked = torch.where(
            self.obs_mask,
            x + distributed_correction,
            x
        )
        print(f"  x_masked shape {x_masked.shape}, has NaN: {torch.isnan(x_masked).any()}")
        print(f"    x_masked stats: min={x_masked.min():.6f}, max={x_masked.max():.6f}")
        print(f"    x_masked norm: {x_masked.norm(dim=(2,3)).mean()}")
        return x_masked


class GuidedVelocityWrapper(VelocityWrapper):
    """Velocity wrapper with DPS guidance for flow matching."""
    
    def __init__(
        self,
        submodel: nn.Module,
        guidance: XCO2Guidance,
        nlev: int = 1,
        static_inputs: Optional[torch.Tensor] = None,
        guidance_start_t: float = 0.0,  # Start applying guidance at this t
        guidance_end_t: float = 1.0,    # Stop applying guidance at this t
    ):
        # Pass arguments to parent VelocityWrapper
        super().__init__(submodel=submodel, nlev=nlev, static_inputs=static_inputs)
        # Add guidance-specific attributes
        self.guidance = guidance
        self.guidance_start_t = guidance_start_t
        self.guidance_end_t = guidance_end_t
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute velocity with guidance.
        
        Args:
            x: [B, C, Nlat, Nlon] - current state
            t: scalar - flow time
            
        Returns:
            v: [B, C, Nlat, Nlon] - velocity with guidance
        """

        # Get base velocity from model
        v_base = super().forward(x, t)  # [B, C, Nlat, Nlon]

        # Apply guidance if within time range
        t_val = t.item()
        if self.guidance_start_t <= t_val <= self.guidance_end_t:
            # Enable gradients for x
            x_guided = x.detach().clone()
            x_guided.requires_grad_(True)

            v_guided = super().forward(x_guided, t)[:, :self.nlev, :, :]

            # Compute denoised prediction: x_denoised ≈ x + (1-t) * v
            # For flow matching: x(t) = t*x_1 + (1-t)*x_0, so x_1 ≈ x + (1-t)*v
            x_denoised = x_guided + (1 - t_val) * v_guided

            # Get guidance gradient
            grad = self.guidance.get_gradient(x_guided, x_denoised, retain_graph=False)

            # Apply guidance: v = v_base - (1-t) * grad
            # The (1-t) factor scales guidance strength with time
            v = v_base - (1 - t_val) * grad.detach()
        else:
            v = v_base

        return v


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
            if "obs_mask" in batch and "obs_values" in batch:
                obs_var = self.generate_kwargs.get("obs_var", "co2massmix")
                masking_config = self.prepare_masking_config(batch, B, C, obs_var)
            else:
                masking_config = {}
            trajectory = self.inference_forward(
                x_in,
                x_init,
                batch=batch,
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
        # print("\nDEBUG prepare_masking_config:")
        if "xco2_averaging_kernel" in batch:
            obs_mask = batch["obs_mask"].reshape(B, self.nlat, self.nlon, 1).permute(0, 3, 1, 2)
            obs_values = batch["obs_values"].reshape(B, self.nlat, self.nlon, 1).permute(0, 3, 1, 2)
            ak = batch["xco2_averaging_kernel"].reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)
            xco2_prior = batch["xco2_apriori"].reshape(B, self.nlat, self.nlon, 1).permute(0, 3, 1, 2)
            co2_profile_prior = batch["co2_profile_apriori"].reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)
        else:
            obs_mask = batch["obs_mask"].reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)
            obs_values = batch["obs_values"].reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)
            ak = None
            xco2_prior = None
            co2_profile_prior = None

        obs_mean = batch[f"{obs_var}_offset"].view(B, 1, 1, 1)
        obs_std = batch[f"{obs_var}_scale"].view(B, 1, 1, 1)
        target_mean = batch[f"{self.target_vars[0]}_offset"].view(B, 1, 1, 1)
        target_std = batch[f"{self.target_vars[0]}_scale"].view(B, 1, 1, 1)

        masking_config = {
            "obs_mask": obs_mask,        # [B 1 Nlat Nlon]  or tests [B C Nlat Nlon]
            "obs_values": obs_values,    # [B 1 Nlat Nlon]  or tests [B C Nlat Nlon]
            "obs_mean": obs_mean,        # [B 1 1 1]
            "obs_std": obs_std,          # [B 1 1 1]
            "target_mean": target_mean,  # [B 1 1 1]
            "target_std": target_std,    # [B 1 1 1]
            "ak": ak,                    # [B C Nlat Nlon]  or tests None
            "xco2_prior": xco2_prior,    # [B 1 Nlat Nlon]  or tests None
            "co2_profile_prior": co2_profile_prior,  # [B C Nlat Nlon] or tests None
        }
        return masking_config

    def compute_xco2(
            self,
            x: torch.Tensor,
            batch: dict,
            masking_config: dict,
            generate_kwargs: dict,
    ) -> torch.Tensor:
        """Compute XCO2 from the current state, by first denormalizing with training data, using the averaging kernel and normalizing with conditioning data."""
        B, N, C = batch[self.target_vars[0]].shape
        x = x.permute(0, 2, 3, 1).reshape(B, N, -1)
        x_physical = self.denormalize_tensor(x, batch)
        x_physical = x_physical.reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)

        ak = masking_config["ak"]  # [B C Nlat Nlon]
        xco2_prior = masking_config["xco2_prior"]  # [B 1 Nlat Nlon]
        co2_profile_prior = masking_config["co2_profile_prior"]  # [B C Nlat Nlon]
        xco2 = xco2_prior + torch.sum(ak * (x_physical - co2_profile_prior), dim=1, keepdim=True)  # [B 1 Nlat Nlon]
        print("\nDEBUG compute_xco2")
        _print_stats_torch("x_physical", x_physical)
        _print_stats_torch("ak", ak)
        _print_stats_torch("co2_profile_prior", co2_profile_prior)
        _print_stats_torch("xco2", xco2)

        target_vars_2d = generate_kwargs["generate_data_kwargs"]["target_vars"]
        xco2 = xco2.permute(0, 2, 3, 1).reshape(B, N, -1)
        xco2_normalized = self.normalize_observations(xco2, batch, target_var=target_vars_2d[0], targshift=False)
        xco2_normalized = xco2_normalized.reshape(B, self.nlat, self.nlon, 1).permute(0, 3, 1, 2)

        return xco2_normalized  # [B 1 Nlat Nlon]

    def return_velocity_wrapper(
            self,
            submodel,
            static_inputs=None,
            masking_config=None,
            generate_kwargs=None,
            ):
        """Return velocity wrapper with optional masking and/or DPS guidance."""

        if masking_config is None:
            masking_config = getattr(self, 'masking_config', {})
        obs_mask = masking_config.get("obs_mask", None)
        obs_values = masking_config.get("obs_values", None)

        if generate_kwargs is None:
            generate_kwargs = getattr(self, 'generate_kwargs', {})
        
        posterior_method = generate_kwargs.get("posterior_method", None)

        if obs_mask is not None and obs_values is not None and posterior_method is None:
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
            x_in,
            x_init,
            batch=None,
            masking_config=None,
            generate_kwargs=None
            ):
        if generate_kwargs is None:
            generate_kwargs = {}

        refine_start = generate_kwargs.get("refine_start", 1.0)
        steps = generate_kwargs.get("steps", 11)
        posterior_method = generate_kwargs.get("posterior_method", None)
        mask_source = generate_kwargs.get("mask_source", "oco2")

        # get timesteps for integration [T]
        if refine_start < 1.0:
            coarse = torch.linspace(0, refine_start, steps=steps, device=x_init.device)[:-1]
            fine = torch.linspace(refine_start, 1.0, steps=steps, device=x_init.device)
            time_grid = torch.cat([coarse, fine[1:]])
        else:
            time_grid = torch.linspace(0, 1, steps=steps-1, device=x_init.device) 
        print("\nDEBUG inference_forward:")
        print(f"  time_grid: {time_grid}")
        print(f"  time_grid diffs: {torch.diff(time_grid)}")
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

        if posterior_method is None:
            trajectory = solver.sample(time_grid=time_grid,
                                x_init=x_init, method=self.method,
                                step_size=self.step_size,
                                return_intermediates=self.return_intermediates
            )  # [T B C Nlat Nlon]
        elif posterior_method == "dflow":
            dflow_optimizer = generate_kwargs.get("dflow_optimizer", None)
            if dflow_optimizer == "adam":
                x_0 = torch.nn.Parameter(x_init, requires_grad=True)  # [B C Nlat Nlon]
                optimizer_x_0 = torch.optim.Adam([x_0], lr=1e-2)
                with torch.enable_grad():
                    for i in range(100):
                        optimizer_x_0.zero_grad()
                        ### DEBUG
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        ### END DEBUG
                        trajectory = solver.sample(time_grid=time_grid,
                                                    x_init=x_0,
                                                    method=self.method,
                                                    step_size=self.step_size,
                                                    return_intermediates=self.return_intermediates,
                                                    enable_grad=True,
                        )  # [T B C Nlat Nlon]
                        x_final = trajectory[-1,...]  # [B C Nlat Nlon]
                        if mask_source == "oco2":
                            x_final = self.compute_xco2(x_final, batch, masking_config, generate_kwargs)
                        x_final = x_final * masking_config["obs_mask"]
                        x_target = torch.where(masking_config["obs_mask"], masking_config["obs_values"], torch.zeros_like(masking_config["obs_values"]))
                        loss = torch.nn.functional.mse_loss(x_final, x_target)
                        loss.backward()
                        optimizer_x_0.step()
                        if i % 10 == 0:
                            print(f"Refinement step {i}, loss: {loss.item():.6f}")
                    print("\nDEBUG memory after solver:")
                    print("  allocated:", torch.cuda.memory_allocated()/1e9, "GB")
                    print("  reserved:", torch.cuda.memory_reserved()/1e9, "GB")
                    print("  peak allocated:", torch.cuda.max_memory_allocated()/1e9, "GB")
            elif dflow_optimizer == "lbfgs":
                max_iter_outer = generate_kwargs.get("lbfgs_max_iter_outer", 10)
                max_iter_inner = generate_kwargs.get("lbfgs_max_iter_inner", 20)
                convergence_threshold = generate_kwargs.get("lbfgs_convergence_threshold", 1e-3)
                reg_loss_weight = generate_kwargs.get("lbfgs_reg_loss_weight", 1e-2)
                reg_loss_type = generate_kwargs.get("lbfgs_reg_loss", "norm_diff")
                x_0 = torch.nn.Parameter(x_init.clone().contiguous())  # [B C Nlat Nlon]
                optimizer_x_0 = torch.optim.LBFGS([x_0], max_iter=max_iter_outer, lr=0.5, line_search_fn='strong_wolfe')  # Use L-BFGS optimizer for better convergence
                log_state = {}
                with torch.enable_grad():
                    for i in range(max_iter_inner):
                        def closure():
                            optimizer_x_0.zero_grad()
                            trajectory = solver.sample(time_grid=time_grid,
                                                        x_init=x_0,
                                                        method=self.method,
                                                        step_size=self.step_size,
                                                        return_intermediates=self.return_intermediates,
                                                        enable_grad=True,
                            )  # [T B C Nlat Nlon]
                            x_final = trajectory[-1,...]  # [B C Nlat Nlon]
                            if mask_source == "oco2":
                                x_final = self.compute_xco2(x_final, batch, masking_config, generate_kwargs)
                            x_final = x_final * masking_config["obs_mask"]
                            x_target = torch.where(masking_config["obs_mask"], masking_config["obs_values"], torch.zeros_like(masking_config["obs_values"]))
                            obs_loss = torch.nn.functional.mse_loss(x_final, x_target)
                            reg_loss = self.lbfgs_reg_loss(x_0, x_init, reg_loss_type)
                            loss = obs_loss + reg_loss_weight * reg_loss
                            loss.backward()
                            log_state["obs_loss"] = obs_loss.detach().item()
                            log_state["reg_loss_weighted"] = (reg_loss_weight * reg_loss).detach().item()
                            return loss
                        
                        loss = optimizer_x_0.step(closure)
                        if loss.item() < convergence_threshold:
                            print(f"Converged at step {i} with loss {loss.item():.3f}")
                            break
                        if i % 10 == 0:
                            print(f"Refinement step {i}, loss: {loss.item():.6f}")
                            print(f"  obs_loss: {log_state['obs_loss']:.6f}, reg_loss_weighted: {log_state['reg_loss_weighted']:.6f}")
                with torch.no_grad():
                    trajectory = solver.sample(time_grid=time_grid,
                                                            x_init=x_0,
                                                            method=self.method,
                                                            step_size=self.step_size,
                                                            return_intermediates=self.return_intermediates,
                                                            enable_grad=False,
                                )  # [T B C Nlat Nlon]
        print("\nDEBUG inference_forward:")
        print(f"  trajectory has NaN: {torch.isnan(trajectory).any()}")
        print(f"  trajectory[-1] stats: min={trajectory[-1].min()}, max={trajectory[-1].max()}")
        return trajectory
    
    def lbfgs_reg_loss(self, x_0, x_init, reg_loss_type="norm_diff"):
        if reg_loss_type is None:
            return torch.tensor(0.0, device=x_0.device) # No regularization
        elif reg_loss_type == "norm_diff":
            B = x_0.shape[0]
            x_0_flat = x_0.reshape(B, -1)
            x_init_flat = x_init.reshape(B, -1)
            norm_diff = (torch.norm(x_0_flat, dim=1) - torch.norm(x_init_flat, dim=1))**2
            return norm_diff.mean()  # Encourage the norm of x_0 to be close to the norm of x_init, which can help with optimization stability. This allows x_0 to deviate from x_init in direction but not in magnitude, which can be beneficial since the flow is designed to transform noise into data and the noise typically has a certain expected norm.
        elif reg_loss_type == "l2":
            return torch.norm(x_0)**2  # L2 regularization on x_0 to prevent it from growing too large, which can help with optimization stability. This encourages the solution to stay close to the initial noise level, which can be beneficial since the flow is designed to transform noise into data.
        elif reg_loss_type == "chi_prior":
            B = x_0.shape[0]
            d = x_0[0].numel()

            x_flat = x_0.reshape(B, -1)
            r = torch.norm(x_flat, dim=1)
            eps = 1e-6
            r = torch.clamp(r, min=eps)

            reg = (d - 1) * torch.log(r) + 0.5 * r**2
            return reg.mean()  # This is the negative log-likelihood of the chi distribution with d degrees of freedom, which is the distribution of the norm of a Gaussian vector in d dimensions. It encourages the norm of x_0 to be close to sqrt(d), which is the expected norm for a Gaussian vector. The eps is added to avoid log(0) when r is very small.
        else:
            raise ValueError(f"Unknown reg_loss_type: {reg_loss_type}")
