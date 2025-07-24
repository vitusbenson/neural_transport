from typing import Dict, Optional

# torch
import torch
import torch.nn as nn


# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver

# neural_transport
from neural_transport.models import MODELS
# from neural_transport.models.unet import UNet

class VelocityWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        target_vars: str,
        offset: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        delta_offset: Optional[torch.Tensor] = None,
        delta_scale: Optional[torch.Tensor] = None,
        static_batch: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.model = model
        self.target_vars = target_vars

        self.offset = offset
        self.scale = scale
        self.delta_offset = delta_offset
        self.delta_scale = delta_scale

        self.static_batch = static_batch or {}

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_vf = {
            self.target_vars: x,
            "flow_time": t,
            f"{self.target_vars}_offset": self.offset,
            f"{self.target_vars}_scale": self.scale,
            f"{self.target_vars}_delta_offset": self.delta_offset,
            f"{self.target_vars}_delta_scale": self.delta_scale,
        }

        batch_vf.update(self.static_batch)
        out = self.model(batch_vf)
        return out[self.target_vars]
    

class FlowMatching(nn.Module):
    def __init__(
            self,
            model="unet",
            model_kwargs={},
            input_vars=[],
            return_intermediates=False,
            method='midpoint',
            nlev=1,
            step_size=0.01
            ):
        super().__init__()

        required_input_vars = ["x_t", "flow_time"]
        input_vars = model_kwargs.get("input_vars", [])
        for var in reversed(required_input_vars):
            if var not in input_vars:
                input_vars.insert(0, var)

        model_kwargs = model_kwargs.copy()
        model_kwargs["input_vars"] = input_vars
        model_kwargs["model_kwargs"]["in_chans"] += 1 + nlev * 1 # + 1 for flow_time, + nlev for co2massmix

        self.model = MODELS[model](**model_kwargs) # UNet(**sub_kwargs)
        self.return_intermediates = return_intermediates
        self.method = method
        self.step_size = step_size
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.target_vars = self.model.target_vars[0] # Here target_vars[0] is supposed to be "co2massmix"

    def forward(self, batch):
        if self.training:
            return self.training_forward(batch)
        else:
            return self.inference_forward(batch)


    def training_forward(self, batch):
        # extract target_vars to get x_1 and normalize, [B N C]
        x_1 = batch[f"{self.target_vars}_next"]
        batch_normalized = self.model.normalize_batch_target_vars(batch)
        x_1_normalized = batch_normalized[f"{self.target_vars}_next"]

        # sample noise  x_0 ~ N(0, I), [B N C]
        x_0 = torch.randn_like(x_1, device=x_1.device)

        # sample time t \in [0,1], [B] -> [B N 1]
        B, N, _ = x_1.shape
        t = torch.rand(B, device=x_1.device)
        t_expanded = t[:, None, None].expand(B, N, 1)  # expand [B N 1] to match x_1 shape
        batch["flow_time"] = t_expanded
        batch["flow_time_offset"] = torch.zeros_like(t_expanded)
        batch["flow_time_scale"] = torch.ones_like(t_expanded)

        # sample path
        path_sample = self.path.sample(
            t=t,
            x_0=x_0,
            x_1=x_1_normalized
        )

        # compute flow matching loss and add denormalized x_1
        # access scheduler for affine path
        scheduler_out = self.path.scheduler(t)
        d_sigma_t = scheduler_out.d_sigma_t.view(-1, 1, 1)
        d_alpha_t = scheduler_out.d_alpha_t.view(-1, 1, 1)

        batch["x_t"] = (path_sample.x_t - d_sigma_t * x_0) / d_alpha_t
        # path_sample.x_t + x_0 (simplified version)
        batch["x_t_offset"] = torch.zeros_like(batch["x_t"])
        batch["x_t_scale"] = batch["x_t"].std(dim=(1, 2), keepdim=True)
        ### why dim=(1, 2) everywhere? why not over the entire batch?
        preds = self.model(batch)

        return preds # [B N C]

    def return_velocity_wrapper(self, offset, scale, delta_offset, delta_scale, batch):
        return VelocityWrapper(
            self.model,
            self.target_vars,
            offset=offset,
            scale=scale,
            delta_offset=delta_offset,
            delta_scale=delta_scale,
            static_batch=batch
        )

    def inference_forward(self, batch):
        # sample noise to get x0 [B N C]
        all_levels = batch[self.target_vars] # [B N C]
        x_init = torch.randn(*all_levels.shape, device=batch[self.target_vars].device)
        #surface_level = batch[self.target_vars][:,:,0:1] # [B N 1]
        #x_init = torch.randn(*surface_level.shape, device=batch[self.target_vars].device)

        # get timesteps for integration [T]
        time_grid = torch.linspace(0, 1, steps=10, device=x_init.device)

        # UNet expects normalization parameters
        velocity_model = self.return_velocity_wrapper(
            batch[f"{self.target_vars}_offset"],
            batch[f"{self.target_vars}_scale"],
            batch[f"{self.target_vars}_delta_offset"],
            batch[f"{self.target_vars}_delta_scale"],
            batch
        )

        # solve the ODE to get the trajectory
        solver = ODESolver(velocity_model=velocity_model)
        sol = solver.sample(time_grid=time_grid,
                                 x_init=x_init, method=self.method,
                                 step_size=self.step_size,
                                 return_intermediates=self.return_intermediates
                                )
        # denormalize sol
        return sol
    
    #def inference_obs_forward(self, batch):
    #    return sol