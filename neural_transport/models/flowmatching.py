from typing import Optional

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
        self.dt = masking_config.get("dt", None)

        self.masking_time = generate_kwargs.get("masking_time", None)
        self.t_threshold = generate_kwargs.get("t_threshold", 0.9)
        self.masking_method = generate_kwargs.get("masking_method", "interpolate")

        print("\nDEBUG MaskedVelocityWrapper init")
        obs_valid = self.obs_values[~torch.isnan(self.obs_values)]
        if obs_valid.numel() > 0:
            print(f"  obs_values valid stats: min={obs_valid.min():.6f}, max={obs_valid.max():.6f}")
        print(f"  obs_mean: {self.obs_mean.flatten()[0]:.6f}")
        print(f"  obs_std: {self.obs_std.flatten()[0]:.6f}")
        print(f"  target_mean: {self.target_mean.flatten()[0]:.6f}")
        print(f"  target_std: {self.target_std.flatten()[0]:.6f}")

    def forward(self, x, t):
        if torch.isnan(x).any():
            print(f"\nDEBUG MaskedVelocityWrapper.forward: INPUT x has NaN at t={t}")
            print(f"  x NaN count: {torch.isnan(x).sum()}")
        if self.masking_method == "simple":
            x_masked = self.masking_simple(x)
        elif self.masking_method == "interpolate":
            x_masked = self.masking_interpolate(x, t)
        elif self.masking_method == "preserve_global_mean":
            x_masked = self.masking_preserve_global_mean(x)
        elif self.masking_method == "preserve_global_mean_and_var":
            x_masked = self.masking_preserve_global_mean_and_var(x)
        elif self.masking_method == "total_column_average_test":
            x_masked = self.masking_total_column_average_test(x)
        elif self.masking_method == "total_column_average_add":
            x_masked = self.masking_total_column_average_add(x)
        elif self.masking_method == "total_column_average_mult":
            x_masked = self.masking_total_column_average_mult(x)
        elif self.masking_method == "total_column_average_test_basic":
            x_masked = self.masking_total_column_average_test_basic(x)

        if torch.isnan(x_masked).any():
            print(f"\nDEBUG MaskedVelocityWrapper.forward: x_masked has NaN at t={t}")
            print(f"  x_masked NaN count: {torch.isnan(x_masked).sum()}")
            print(f"  masking_method: {self.masking_method}")
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
        x_effective = mask_weight * x_masked + (1.0 - mask_weight) * x

        # dxt = (x_effective - x)/dt + f(x_effective, t)
        dtx = (x_effective - x) / self.dt + super().forward(x_effective, t)
        # dxt = f(x,t)
        # dxt = torch.where(self.obs_mask, self.obs_values - x, super().forward(x, t))
        if torch.isnan(dtx).any():
            print(f"\nDEBUG MaskedVelocityWrapper.forward: OUTPUT dtx has NaN at t={t}")
            print(f"  dtx NaN count: {torch.isnan(dtx).sum()}")
            print(f"  (x_effective - x)/dt stats: min={(x_effective - x).min()/self.dt:.6f}, max={(x_effective - x).max()/self.dt:.6f}")
    
        return dtx

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
        # print("\nDEBUG prepare_masking_config:")
        # print(f"  obs_var: {obs_var}")
        # print(f"  batch keys: {list(batch.keys())}")
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
            "obs_mask": obs_mask,        # [B 1 Nlat Nlon]
            "obs_values": obs_values,    # [B 1 Nlat Nlon]
            "obs_mean": obs_mean,        # [B 1 1 1]
            "obs_std": obs_std,          # [B 1 1 1]
            "target_mean": target_mean,  # [B 1 1 1]
            "target_std": target_std,    # [B 1 1 1]
            "ak": ak,                    # [B C Nlat Nlon]
            "xco2_prior": xco2_prior,
            "co2_profile_prior": co2_profile_prior,
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

        # get timesteps for integration [T]
        if refine_start < 1.0:
            coarse = torch.linspace(0, refine_start, steps=11, device=x_init.device)[:-1]
            fine = torch.linspace(refine_start, 1.0, steps=11, device=x_init.device)
            time_grid = torch.cat([coarse, fine[1:]])
        else:
            time_grid = torch.linspace(0, 1, steps=10, device=x_init.device) 
        dt = (time_grid[-1] - time_grid[0]) / (len(time_grid) - 1)
        masking_config["dt"] = dt

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
        print("\nDEBUG inference_forward:")
        print(f"  trajectory has NaN: {torch.isnan(trajectory).any()}")
        print(f"  trajectory[-1] stats: min={trajectory[-1].min()}, max={trajectory[-1].max()}")
        return trajectory
