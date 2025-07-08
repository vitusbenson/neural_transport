# torch
import torch
import torch.nn as nn

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver

# neural_transport
from neural_transport.models import MODELS


class FlowMatching(nn.Module):
    def __init__(
            self,
            model="unet",
            model_kwargs={},
            return_intermediates=False,
            method='midpoint',
            step_size=0.01
            ):
        super().__init__()
        self.model = MODELS[model](**model_kwargs)
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.solver = ODESolver(velocity_model=self.model)
        self.method = method
        self.step_size = step_size
        self.return_intermediates = return_intermediates


    def forward(self, batch):
        if self.training:
            ### how to enter into the inference phase? self.training = False
            return self.training_forward(batch)
        else:
            return self.inference_forward(batch)


    def training_forward(self, batch):
        # extract target_vars to get x1 and normalize # B N C
        x_1 = batch[f"{self.model.target_vars[0]}_next"]
        ### Here target_vars[0] is supposed to be co2massmix
        batch_normalized = self.normalize_batch(self, batch)
        x_1_normalized = batch_normalized[f"{self.model.target_vars[0]}_next"]
    
        # sample noise [B N C]
        x_0 = torch.randn_like(x_1, device=x_1.device)

        # sample time [B]
        t = torch.rand(x_1.shape[0], device=x_1.device)

        # sample path
        path_sample = self.path.sample(
            t=t,
            x_0=x_0,
            x_1=x_1_normalized
        )

        # compute flow matching loss and add denormalized x_1
        preds = self.model(path_sample.x_t, path_sample.t) + x_1
        # loss_input_2 = path_sample.dx_t
        return preds # B N C



    def inference_forward(self, batch):
        # sample noise to get x0 # B N C
        x_init = torch.randn((batch[self.model.hparams.target_vars[0]].shape[0], 2))

        # get timesteps for integration T
        time_grid = torch.linspace(0, 1, steps=10)

        # solve the ODE to get the trajectory
        sol = self.solver.sample(time_grid=time_grid,
                                 x_init=x_init, method=self.method,
                                 step_size=self.step_size,
                                 return_intermediates=self.return_intermediates
                                )
        # denormalize sol
        return sol
    
    def inference_obs_forward(self, batch):
        return sol