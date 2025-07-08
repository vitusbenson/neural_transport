# import torch.nn as nn
import torch.nn as nn

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver


class FlowMatching(nn.Module):
    def __init__(
            self,
            return_intermediates=False,
            step_size=0.01,
            method='midpoint'
            ):
        super().__init__()
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.model = UNet()
        self.solver = ODESolver(velocity_model=self.model)
        self.method = method
        self.step_size = step_size
        self.return_intermediates = return_intermediates
    

    def forward(self, batch):
        if self.training:
            return self.training_forward(batch)
    
        else:
            return self.inference_forward(batch)


    def training_forward(self, batch):
        # extrahiere target_vars to get x1 and normalize # B N C
        # sample noise to get x0 # B N C

        # sample time t # B

        # sample path

        # compute flow matching loss and add denormalized x_1
    
        return preds # B N C



    def inference_forward(self, batch):
        # sample noise to get x0 # B N C
        # get timesteps for integration T
        # solve the ODE to get the trajectory
        sol = self.solver.sample(time_grid=T, 
                                 x_init=x_init, method=self.method, 
                                 step_size=self.step_size, 
                                 return_intermediates=self.return_intermediates
                                )  # sample from the model
        # denormalize sol
        return sol