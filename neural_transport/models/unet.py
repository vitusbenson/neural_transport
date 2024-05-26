import torch.nn as nn

from neural_transport.models.regulargrid import RegularGridModel

ACTIVATIONS = {
    "none": nn.Identity,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": nn.LeakyReLU,
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "hardsigmoid": nn.Hardsigmoid,
    "hardtanh": nn.Hardtanh,
    "swish": nn.SiLU,
}


def get_norm(norm, n_in, n_groups=8):

    if norm == "batch":
        return nn.BatchNorm2d(n_in)
    elif norm == "group":
        return nn.GroupNorm(n_groups, n_in)
    else:
        return nn.Identity()


class PeriodicPadding(nn.Module):

    def __init__(self, n_pad):
        super().__init__()
        self.n_pad = n_pad

    def forward(self, x):

        x = nn.functional.pad(
            x, (0, 0, self.n_pad, self.n_pad), mode="circular"
        )  # torch.cat([x[:, :, -self.n_pad:, :], x, x[:, :, :self.n_pad, :]], dim = 2)

        x = nn.functional.pad(x, (self.n_pad, self.n_pad), mode="constant", value=0)

        return x


class ResBlock(nn.Module):

    def __init__(self, n_in, embed_dim, act="leakyrelu", norm="batch", filter_size=3):
        super().__init__()

        n_pad = (filter_size - 1) // 2

        self.pad = PeriodicPadding(n_pad)

        self.conv = nn.Conv2d(
            n_in, embed_dim, filter_size, stride=1, padding=0, bias=(norm is None)
        )

        self.act = ACTIVATIONS[act]()

        self.norm = get_norm(norm, embed_dim)

    def forward(self, x):

        skip = x

        x = self.pad(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)

        if skip.shape == x.shape:
            x = x + skip

        return x


class UNet(RegularGridModel):

    def init_model(
        self,
        in_chans=193,
        out_chans=19,
        embed_dim=128,
        act="leakyrelu",
        norm="batch",
        enc_filters=[[7], [3, 3], [3, 3], [3, 3]],
        dec_filters=[[3, 3], [3, 3], [3, 3], [3, 3]],
    ):

        self.in_chans = in_chans
        self.out_chans = out_chans

        depth = len(enc_filters)
        assert depth == len(dec_filters)

        self.resc_lat = ((self.nlat // (2**depth)) + 1) * (2**depth)
        self.resc_lon = ((self.nlon // (2**depth)) + 1) * (2**depth)

        self.embed_dim = embed_dim

        enc_stages = []
        for i, filters in enumerate(enc_filters):
            enc_stage = []
            if i > 0:
                enc_stage.append(nn.MaxPool2d(2, 2))
            for j, filter_size in enumerate(filters):
                n_in = in_chans if (i == 0) and (j == 0) else embed_dim
                enc_stage.append(
                    ResBlock(
                        n_in, embed_dim, act=act, norm=norm, filter_size=filter_size
                    )
                )
            enc_stages.append(nn.Sequential(*enc_stage))

        self.enc_stages = nn.ModuleList(enc_stages)

        dec_stages = []
        for i, filters in enumerate(dec_filters):
            dec_stage = []
            for j, filter_size in enumerate(filters):
                dec_stage.append(
                    ResBlock(
                        embed_dim,
                        embed_dim,
                        act=act,
                        norm=norm,
                        filter_size=filter_size,
                    )
                )

            if i != (len(dec_filters) - 1):
                dec_stage.append(nn.Upsample(scale_factor=2))
            dec_stages.append(nn.Sequential(*dec_stage))

        self.dec_stages = nn.ModuleList(dec_stages)

        self.readout = ResBlock(
            embed_dim, out_chans, act="none", norm="none", filter_size=1
        )

    def model(self, x_in):

        x = nn.functional.interpolate(
            x_in,
            size=(self.resc_lat, self.resc_lon),
            align_corners=True,
            mode="bilinear",
        )

        skips = []
        for stage in self.enc_stages:
            x = stage(x)
            skips.append(x)

        x = self.dec_stages[0](x)

        for stage, skip in zip(self.dec_stages[1:], skips[::-1][1:]):
            x = stage(x + skip)

        x = nn.functional.interpolate(
            x, size=(self.nlat, self.nlon), mode="nearest-exact"
        )

        x_out = self.readout(x)

        return x_out