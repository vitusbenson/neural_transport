# %%

# neural_transport
from neural_transport.models import MODELS
from neural_transport.models.regulargrid import RegularGridModel
from neural_transport.models.unet import UNet
from neural_transport.datasets.grids import (
    LATLON_PROTOTYPE_COORDS,
    VERTICAL_LAYERS_PROTOTYPE_COORDS,
)
from neural_transport.datamodule import CarbonDataModule

# plotting
import matplotlib.pyplot as plt

# torch
import pytorch_lightning as pl
import torch
import torch.nn as nn

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver

# %%
torch.set_float32_matmul_precision("high")
pl.seed_everything(42)
# %%
TARGET_VARS = ["co2massmix"]
FORCING_VARS_2D = [
    "blh",
    "cell_area",
    "co2flux_anthro",
    "co2flux_land",
    "co2flux_ocean",
    "orography",
    "tisr",
]
FORCING_VARS_3D = [
    "airmass",
    "gph_bottom",
    "gph_top",
    "p_bottom",
    "p_top",
    "q",
    "t",
    "u",
    "v",
]
grid = "latlon5.625"
vertical_levels = "l10"
freq = "6h"

nlev = len(VERTICAL_LAYERS_PROTOTYPE_COORDS[vertical_levels]["level"])

FORCING_VARS = FORCING_VARS_2D + FORCING_VARS_3D
LEN_ALL_TARGET_VARS = nlev * len(TARGET_VARS)
LEN_ALL_FORCING_VARS = len(FORCING_VARS_2D) + nlev * len(FORCING_VARS_3D)
LEN_ALL_VARS = LEN_ALL_TARGET_VARS + LEN_ALL_FORCING_VARS

lat = LATLON_PROTOTYPE_COORDS[grid]["lat"]
lon = LATLON_PROTOTYPE_COORDS[grid]["lon"]

MODEL_DIMS = {
    "XS": dict(embed_dim=64),
    "S": dict(embed_dim=128),
    "M": dict(embed_dim=256),
    "L": dict(embed_dim=512),
    "XL": dict(embed_dim=1024),
}

MODEL_SIZE = "S"

model_kwargs = dict(
    model_kwargs=dict(
        in_chans=LEN_ALL_VARS,
        out_chans=LEN_ALL_TARGET_VARS,
        embed_dim=MODEL_DIMS[MODEL_SIZE]["embed_dim"],
        act="leakyrelu",
        norm="batch",
        enc_filters=[[7], [3, 3], [3, 3], [3, 3]],
        dec_filters=[[3, 3], [3, 3], [3, 3], [3, 3]],
        in_interpolation="bilinear",
        out_interpolation="nearest-exact",
        out_clip=None,
    ),
    input_vars=TARGET_VARS + FORCING_VARS,
    target_vars=TARGET_VARS,
    nlat=len(lat),
    nlon=len(lon),
    predict_delta=True,
    add_surfflux=True,
    dt=60 * 60 * 6,
    massfixer="scale",
    targshift=True,
)

model = UNet(**model_kwargs)

# %%
N_GPUS = 1
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_PRED = 32

grid = "latlon5.625"
vertical_levels = "l10"
freq = "6h"

data_kwargs = dict(
    data_path="/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker",
    dataset="carbontracker",
    grid=grid,
    vertical_levels=vertical_levels,
    freq=freq,
    n_timesteps=1,
    batch_size_train=BATCH_SIZE_TRAIN // N_GPUS,
    batch_size_pred=BATCH_SIZE_PRED,
    num_workers=32 * N_GPUS,
    val_rollout_n_timesteps=31,
    target_vars=["co2massmix", "airmass"],
    forcing_vars=[
        "gph_bottom",
        "gph_top",
        "p_bottom",
        "p_top",
        "q",
        "t",
        "u",
        "v",
        "blh",
        "cell_area",
        "co2flux_anthro",
        "co2flux_land",
        "co2flux_ocean",
        "orography",
        "tisr",
    ],
    compute=False,
    # time_interval=["1990-01-01", "2014-12-31"],
)


# %%
dset = CarbonDataModule(**data_kwargs)

# %%
dset

# %%
dir(dset)

# %%
dset.setup('fit')

# %%
dir(dset)

# %%
dset.val_dataset

# %%
dl = dset.train_dataloader()

# %%
dl

# %%
batch = next(iter(dl))
# %%
batch = {k: v[:, 0,...] if isinstance(v, torch.Tensor) else v for k, v in batch.items() }
# %%
batch

# %%
batch.keys()

# %%
batch['co2massmix'].shape

# %%
class Cheating:
    def __init__(self):
        pass

# %%
self = Cheating()

# %%
self.model = model

# %%
x_1 = batch[f"{self.model.target_vars[0]}_next"]

# %%
x_1.shape
# %%
plt.imshow(x_1[0,:,0].reshape(32, 64).numpy()[::-1])

# %%
min_val = x_1.min()
max_val = x_1.max()

print(min_val, max_val)

# %%
def normalize_batch(self, batch):
    
    batch_normalized = {}
    for v in self.input_vars:
        x_in_curr = (batch[v] - batch[f"{v}_offset"]) / batch[f"{v}_scale"]
        if self.targshift and (v in self.target_vars):
            batch_normalized[v] = x_in_curr - x_in_curr.mean((1, 2), keepdim=True)
        else:
            batch_normalized[v] = x_in_curr

    return batch_normalized

def preprocess_inputs(self, batch):

    batch_normalized = normalize_batch(self, batch)

    x_in = torch.cat(list(batch_normalized.values()), dim=-1)

    B, N, C = x_in.shape

    x_in = x_in.reshape(B, self.in_nlat, self.in_nlon, C).permute(
        0, 3, 1, 2
    )  # b c h w

    if self.horizontal_interpolation == "multiscale_encoder":
        x_in = self.multiscale_encoder(x_in)

    elif self.horizontal_interpolation is not None:
        x_in = nn.functional.interpolate(
            x_in,
            size=(self.nlat, self.nlon),
            align_corners=True,
            mode=self.horizontal_interpolation,
        )

    # if not x_in.isfinite().all():
    #     print("x_in not finite", x_in.min(), x_in.mean(), x_in.max())

    return x_in
# %%
batch_normalized = normalize_batch(self, batch)
x_1_normalized = batch_normalized[f"{self.model.target_vars[0]}_next"]
min_normalized = x_1_normalized.min()
max_normalized = x_1_normalized.max()

print(min_normalized, max_normalized)

plt.imshow(x_1_normalized[0,:,0].reshape(32, 64).numpy()[::-1])

# %%
x_0 = torch.randn_like(x_1, device=x_1.device)
plt.imshow(x_0[0,:,0].reshape(32, 64).numpy()[::-1])


# %%
t = torch.rand(x_1.shape[0], device=x_1.device)
# %%
path = AffineProbPath(scheduler=CondOTScheduler())
path_sample = path.sample(
            t=t,
            x_0=x_0,
            x_1=x_1_normalized
        )

# %%
