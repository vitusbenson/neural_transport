import torch
import torch.nn as nn


class Tanh3x(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x / 2) * 3


ACTIVATIONS = {
    "none": nn.Identity,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": nn.LeakyReLU,
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "tanh3x": Tanh3x,
    "hardsigmoid": nn.Hardsigmoid,
    "hardtanh": nn.Hardtanh,
    "swish": nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out, layer_norm=True, act="swish"):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(n_in, n_hid, bias=True),
            ACTIVATIONS[act](),
            nn.Linear(n_hid, n_out, bias=(not layer_norm)),
        )

        self.norm = nn.LayerNorm(n_out) if layer_norm else nn.Identity()

    def forward(self, x):
        return self.norm(self.mlp(x))


class MultiScaleModule(nn.Module):

    def __init__(self):
        super().__init__()

    def register_position_features(self, in_shape, out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape

        window_size_lat = in_shape[0] // out_shape[0]
        window_size_lon = in_shape[1] // out_shape[1]

        self.window_size_lat = window_size_lat
        self.window_size_lon = window_size_lon

        lats = torch.linspace(
            -90 + 90 / in_shape[0], 90 - 90 / in_shape[0], in_shape[0]
        )
        lons = torch.linspace(0, 360 - 360 / in_shape[1], in_shape[1])

        lat_scaled, lon_scaled = torch.meshgrid(
            torch.deg2rad(90 - lats) / torch.pi,
            torch.deg2rad(lons) / (2 * torch.pi),
        )

        cos_lat = torch.cos(torch.pi * lat_scaled)
        sin_lon = torch.sin(2 * torch.pi * lon_scaled)
        cos_lon = torch.cos(2 * torch.pi * lon_scaled)

        latstep = torch.abs(torch.diff(lats).mean())
        lonstep = torch.abs(torch.diff(lons).mean())
        area = torch.stack(
            in_shape[1]
            * [
                torch.abs(
                    torch.sin(torch.deg2rad(lats + latstep / 2))
                    - torch.sin(torch.deg2rad(lats - latstep / 2))
                )
                * lonstep
                / 180
                * 100  # arbitrary scaling
            ],
            dim=-1,
        )

        feats = torch.stack(
            [lat_scaled, lon_scaled, cos_lat, sin_lon, cos_lon, area], dim=-1
        )

        window_feats = feats.reshape(
            -1,
            in_shape[0] // window_size_lat,
            window_size_lat,
            in_shape[1] // window_size_lon,
            window_size_lon,
        ).permute(0, 1, 3, 2, 4)

        relative_lats = torch.linspace(
            -1 + 1 / window_size_lat, 1 - 1 / window_size_lat, window_size_lat
        )
        relative_lons = torch.linspace(
            -1 + 1 / window_size_lon, 1 - 1 / window_size_lon, window_size_lon
        )

        relative_lats, relative_lons = torch.meshgrid(
            relative_lats,
            relative_lons,
        )

        window_feats = torch.cat(
            [
                window_feats,
                relative_lats.expand(
                    1,
                    in_shape[0] // window_size_lat,
                    in_shape[1] // window_size_lon,
                    window_size_lat,
                    window_size_lon,
                ),
                relative_lons.expand(
                    1,
                    in_shape[0] // window_size_lat,
                    in_shape[1] // window_size_lon,
                    window_size_lat,
                    window_size_lon,
                ),
            ],
            dim=0,
        )

        self.n_position_feats = window_feats.shape[0]

        self.register_buffer("position_feats", window_feats)

    def forward(self, x):
        raise NotImplementedError


class MultiScaleEncoder(MultiScaleModule):

    def __init__(
        self,
        in_shape,
        out_shape,
        in_chans,
        out_chans,
        layer_norm=True,
        act="swish",
        mlp_ratio=4,
    ):
        super().__init__()

        self.register_position_features(in_shape, out_shape)

        self.mlp = MLP(
            in_chans + self.n_position_feats,
            mlp_ratio * out_chans,
            out_chans,
            layer_norm=layer_norm,
            act=act,
        )

    def forward(self, x):

        B, C, H, W = x.shape
        # Transform into windows
        x_window = x.reshape(
            B,
            C,
            H // self.window_size_lat,
            self.window_size_lat,
            W // self.window_size_lon,
            self.window_size_lon,
        ).permute(0, 1, 2, 4, 3, 5)
        # Stack Positional features
        x_feats = torch.cat(
            [
                x_window,
                self.position_feats.expand(B, -1, -1, -1, -1, -1).type_as(x),
            ],
            dim=1,
        )  # b c h w x y
        # Apply MLP
        x_embed = self.mlp(x_feats.permute(0, 2, 3, 4, 5, 1)).permute(0, 5, 1, 2, 3, 4)
        # Sum over Windows

        x_coarse = x_embed.sum(dim=(4, 5))

        return x_coarse


class MultiScaleDecoder(MultiScaleModule):

    def __init__(
        self,
        in_shape,
        out_shape,
        in_chans,
        out_chans,
        layer_norm=True,
        act="swish",
        mlp_ratio=4,
    ):
        super().__init__()

        self.register_position_features(in_shape, out_shape)

        self.mlp = MLP(
            in_chans + self.n_position_feats,
            mlp_ratio * out_chans,
            out_chans,
            layer_norm=layer_norm,
            act=act,
        )

    def forward(self, x):

        # Repeat into Window Shape
        # Stack Positional features
        # Apply MLP

        B, C, H, W = x.shape

        # Repeat into Window Shape

        x_repeated = (
            x.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(B, C, H, W, self.window_size_lat, self.window_size_lon)
        )

        # Stack Positional features
        x_feats = torch.cat(
            [
                x_repeated,
                self.position_feats.expand(B, -1, -1, -1, -1, -1).type_as(x),
            ],
            dim=1,
        )  # b c h w x y
        # Apply MLP
        x_embed = self.mlp(x_feats.permute(0, 2, 3, 4, 5, 1)).permute(0, 5, 1, 2, 3, 4)

        # Merge Windows

        B, C, H, W, Hw, Ww = x_embed.shape

        x_fine = x_embed.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H * Hw, W * Ww)

        return x_fine
