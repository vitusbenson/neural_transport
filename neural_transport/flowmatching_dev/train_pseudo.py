import torch

# neural network for flow matching
from neural_transport.models.unet import UNet

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver



if torch.cuda.is_available():
    device = 'cuda:0'
    print('Using gpu')
else:
    device = 'cpu'
    print('Using cpu.')


# training arguments
lr = 1e-4
batch_size = 64
iterations = 10000

# vector field model initialization
vf = UNet()

# instantiate an affine conditional flow
flow = AffineProbPath(scheduler=CondOTScheduler())

# initialize optimizer
optim = torch.optim.Adam(vf.parameters(), lr=lr)

# training loop for vector field
for i in range(iterations):
    optim.zero_grad()

    # sample data
    x_1 = 
    x_0 = torch.randn_like(x_1).to(device)

    # sample time
    t = torch.rand(x_1.shape[0]).to(device) 

    # sample flow
    flow_sample = flow.sample(
        x_0=x_0,
        x_1=x_1,
        t=t,
        vf=vf,
    )

    # flow matching l2 loss
    loss = torch.pow( vf(flow_sample.x_t,flow_sample.t) - flow_sample.dx_t, 2).mean() 

    # optimizer step
    loss.backward() # backward
    optim.step() # update
    

# use ODESolver to obtain the flow from vector field
solver = ODESolver(velocity_model=vf)

# then sample the flow to obtain the trajectory and finally x_1
sol = solver.sample(
    x_init=x_0,  # initial condition
    step_size=0.01,  # step size for the ODE solver
    time_grid=torch.tensor([0.0, 1.0]),  # time grid for the ODE solver
    method='euler',  # method for the ODE solver
    enable_grad=True,  # enable gradients for backpropagation
)