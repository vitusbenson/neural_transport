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
        submodel,
        nlev=1,
        static_inputs=None,
        obs_mask=None,
        obs_values=None,
        dt=None,
        ak=None,
        **generate_kwargs,
    ):
        super().__init__(submodel=submodel, nlev=nlev, static_inputs=static_inputs)
        self.obs_mask = obs_mask
        self.obs_values = obs_values
        self.dt = dt
        self.ak = ak

        self.masking_time = generate_kwargs.get("masking_time", None)
        self.t_threshold = generate_kwargs.get("t_threshold", 0.9)
        self.masking_method = generate_kwargs.get("masking_method", "interpolate")

    def forward(self, x, t):
        if self.masking_method == "simple":
            x_masked = self.masking_simple(x)
        elif self.masking_method == "interpolate":
            x_masked = self.masking_interpolate(x, t)
        elif self.masking_method == "preserve_global_mean":
            x_masked = self.masking_preserve_global_mean(x)
        elif self.masking_method == "preserve_global_mean_and_var":
            x_masked = self.masking_preserve_global_mean_and_var(x)
        elif self.masking_method == "total_column_average":
            x_masked = self.masking_total_column_average(x, ak=self.ak)

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
    
    def masking_total_column_average(self, x, ak=None):
        if ak is None:
            ak = torch.ones(x.shape, device=x.device)
        x_averaged = (ak * x).sum(dim=1, keepdim=True)  # [B 1 Nlat Nlon]
        x_masked = torch.where(self.obs_mask, self.obs_values.detach()/x_averaged.clamp(min=1e-12) * x, x)
        return x_masked


class FlowMatching(RegularGridModel):
    def init_model(
            self,
            submodel="unet",
            model_kwargs={},
            return_intermediates=False,
            method='midpoint',
            nlev=1,
            step_size=0.01,
            generate_kwargs=None,
            ):
        
        self.submodel = MODELS[submodel](**model_kwargs)
        self.return_intermediates = return_intermediates
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
            preds["dx_t"] = dx_t.permute(0, 2, 3, 1).reshape(*preds[self.target_vars[0]].shape) # [B N C]
            return preds
        elif self.return_intermediates:
            x_in = self.preprocess_inputs(batch)
            all_levels = x_in[:, :self.nlev*len(self.target_vars), :, :]  # [B C Nlat Nlon]
            # surface_level = x_in[:, :1, :, :]  # [B 1 Nlat Nlon]
            if "noise" in batch:
                noise = batch["noise"]
                B, N, C = noise.shape
                x_init = noise.reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2) # [B C Nlat Nlon]
            else:
                x_init = torch.randn_like(all_levels, device=x_in.device)
            if "obs_mask" in batch and "obs_values" in batch:
                obs_mask = batch["obs_mask"].reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2) # [B C Nlat Nlon]
                obs_values = batch["obs_values"].reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)
                if "xco2_averaging_kernel" in batch: # Do I really want averaging kernel to go through preprocess_inputs or should it be before? But if not preprocessed, need to adjust shapes
                    ak = batch["xco2_averaging_kernel"]
                else:
                    ak = None
            else:
                obs_mask = None
                obs_values = None
            trajectory = self.inference_forward(
                x_in, x_init,
                obs_mask, obs_values,
                ak=ak,
                generate_kwargs=self.generate_kwargs
            )
            x_out = trajectory[-1,...]
            sol = self.postprocess_outputs(x_out, batch)
            T, B, C, Nlat, Nlon = trajectory.shape
            trajectory = trajectory.permute(0, 1, 3, 4, 2) # [T B Nlat Nlon C]
            trajectory = trajectory.reshape(T, B, Nlat*Nlon, C) # [T B Nlat*Nlon C]
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

    def return_velocity_wrapper(
            self, submodel,
            obs_mask=None, obs_values=None,
            static_inputs=None,
            dt=None,
            ak=None,
            generate_kwargs=None,
            ):
        if generate_kwargs is None:
            generate_kwargs = {}

        if obs_mask is not None and obs_values is not None:
            return MaskedVelocityWrapper(
                submodel=submodel,
                nlev=self.nlev,
                static_inputs=static_inputs,
                obs_mask=obs_mask,
                obs_values=obs_values,
                dt=dt,
                ak=ak,
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
            obs_mask, obs_values,
            ak=None,
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

        # UNet expects normalization parameters
        velocity_model = self.return_velocity_wrapper(
            submodel=self.submodel,
            static_inputs=None,
            obs_mask=obs_mask,
            obs_values=obs_values,
            dt=dt,
            ak=ak,
            generate_kwargs=generate_kwargs,
            # static_inputs=x_in[:,:self.nlev*len(self.target_vars),:,:],  # [B C Nlat Nlon] (static inputs for conditioning later)
        )

        # solve the ODE to get the trajectory
        solver = ODESolver(velocity_model=velocity_model)
        trajectory = solver.sample(time_grid=time_grid,
                            x_init=x_init, method=self.method,
                            step_size=self.step_size,
                            return_intermediates=self.return_intermediates
        ) # [T B C Nlat Nlon]
        return trajectory
