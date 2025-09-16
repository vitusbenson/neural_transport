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
# from neural_transport.models.unet import UNet

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
        x_in = torch.cat(
            [x, t_expanded] + [self.static_inputs], dim=1
        ) # [B C_total Nlat Nlon]
        out = self.submodel.model(x_in)
        return out[:, :self.nlev, :, :]
    

class FlowMatching(RegularGridModel):
    def init_model(
            self,
            submodel="unet",
            model_kwargs={},
            return_intermediates=False,
            method='midpoint',
            nlev=1,
            step_size=0.01,
            ):
        
        self.submodel = MODELS[submodel](**model_kwargs)
        self.return_intermediates = return_intermediates
        self.method = method
        self.nlev = nlev
        self.step_size = step_size
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.target_vars = self.submodel.target_vars # Here target_vars[0] is supposed to be "co2massmix"
        # self.velocity_model = VelocityWrapper(self.submodel, self.target_vars)
    
    def forward(self, batch):
        if self.training:
            x_out, dx_t = self.training_forward(batch)
            preds = self.postprocess_outputs(x_out, batch, denormalize=False)
            preds["dx_t"] = dx_t.permute(0, 2, 3, 1).reshape(*preds[self.target_vars[0]].shape) # [B N C]
            return preds
        elif self.return_intermediates:
            x_in = self.preprocess_inputs(batch)
            trajectory = self.model(x_in)
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

    def return_velocity_wrapper(self, submodel, static_inputs):
        return VelocityWrapper(
            submodel=submodel,
            nlev=self.nlev,
            static_inputs=static_inputs,
        )

    # inference_forward
    def model(self, x_in):
        # sample noise to get x_init [B C Nlat Nlon]
        all_levels = x_in[:, :self.nlev*len(self.target_vars), :, :]  # [B C Nlat Nlon]
        x_init = torch.randn_like(all_levels, device=x_in.device)
        #surface_level = x_in[:, :1, :, :]  # [B 1 Nlat Nlon]
        #x_init = torch.randn(*surface_level.shape, device=batch[self.target_vars].device)

        # get timesteps for integration [T]
        time_grid = torch.linspace(0, 1, steps=10, device=x_init.device)

        # UNet expects normalization parameters
        velocity_model = self.return_velocity_wrapper(
            submodel=self.submodel,
            static_inputs=x_in[:, self.nlev*len(self.target_vars):, :, :],  # [B C Nlat Nlon] (static inputs)
        )

        # solve the ODE to get the trajectory
        solver = ODESolver(velocity_model=velocity_model)
        trajectory = solver.sample(time_grid=time_grid,
                            x_init=x_init, method=self.method,
                            step_size=self.step_size,
                            return_intermediates=self.return_intermediates
        ) # [T B C Nlat Nlon]
        return trajectory
    
    #def inference_obs_forward(self, batch):
    #    return sol