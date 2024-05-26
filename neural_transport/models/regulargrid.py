import torch
import torch.nn as nn

from neural_transport.tools.conversion import *


class RegularGridModel(nn.Module):

    def __init__(
        self,
        model_kwargs={},
        input_vars=[],
        target_vars=[],
        nlat=45,
        nlon=72,
        predict_delta=False,
        add_surfflux=False,
        dt=3600,
    ):
        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.input_vars = input_vars
        self.target_vars = target_vars

        self.predict_delta = predict_delta
        self.add_surfflux = add_surfflux
        self.dt = dt

        self.init_model(**model_kwargs)

    def init_model(self, **model_kwargs):
        raise NotImplementedError

    def model(self):
        raise NotImplementedError

    def forward(self, batch):
        x_in = self.preprocess_inputs(batch)
        x_out = self.model(x_in)
        preds = self.postprocess_outputs(x_out, batch)
        return preds

    def preprocess_inputs(self, batch):

        x_in = torch.cat(
            [
                (batch[v] - batch[f"{v}_offset"]) / batch[f"{v}_scale"]
                for v in self.input_vars
            ],
            dim=-1,
        )

        B, N, C = x_in.shape

        x_in = x_in.reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)  # b c h w

        return x_in

    def postprocess_outputs(self, x_out, batch):

        B, N, _ = batch[self.target_vars[0]].shape

        x_out = x_out.permute(0, 2, 3, 1).reshape(B, N, -1)

        x_grid_offset = torch.cat(
            [(batch[f"{v}_offset"]).expand_as(batch[v]) for v in self.target_vars],
            dim=-1,
        )
        x_grid_scale = torch.cat(
            [(batch[f"{v}_scale"]).expand_as(batch[v]) for v in self.target_vars],
            dim=-1,
        )

        x_out_prev = torch.cat(
            [batch[v] for v in self.target_vars],
            dim=-1,
        )
        x_grid_delta_offset = torch.cat(
            [
                (batch[f"{v}_delta_offset"]).expand_as(batch[v])
                for v in self.target_vars
            ],
            dim=-1,
        )
        x_grid_delta_scale = torch.cat(
            [(batch[f"{v}_delta_scale"]).expand_as(batch[v]) for v in self.target_vars],
            dim=-1,
        )

        if self.predict_delta:
            x_out_resc = x_out * x_grid_delta_scale + x_grid_delta_offset

            x_out_next = x_out_prev + x_out_resc
        else:
            x_out_next = x_out * x_grid_scale + x_grid_offset

        preds = {}
        i = 0
        for v in self.target_vars:
            C = batch[v].shape[-1]
            preds[v] = x_out_next[..., i : i + C]
            i += C

        for molecule in ["co2", "ch4"]:
            if (
                f"{molecule}density" in preds
                and (f"{molecule}massmix" not in preds)
                # v.endswith("density")
                # and v != "airdensity"
                # and v.replace("density", "massmix") not in self.target_vars
            ):
                if self.add_surfflux and (f"{molecule}flux_land" in batch):
                    surfflux_as_massmixsource_prev = mass_to_density(
                        (
                            batch[f"{molecule}flux_land"]
                            + batch[f"{molecule}flux_ocean"]
                            + batch[f"{molecule}flux_subt"]
                        )
                        * batch["cell_area"]
                        * self.dt,
                        batch["volume"][..., :1],
                    )  # b n h
                    preds[f"{molecule}density"][..., :1] = (
                        preds[f"{molecule}density"][..., :1]
                        + surfflux_as_massmixsource_prev
                    )

                preds[f"{molecule}massmix"] = density_to_massmix(
                    preds[f"{molecule}density"],
                    (
                        batch["airdensity_next"]
                        if not "airdensity" in self.target_vars
                        else preds["airdensity"]
                    ),
                    ppm=True,
                )
            if (
                f"{molecule}massmix" in preds
                and (f"{molecule}density" not in preds)
                # v.endswith("massmix")
                # and v.replace("massmix", "density") not in self.target_vars
            ):

                if self.add_surfflux and (f"{molecule}flux_land" in batch):
                    surfflux_as_massmixsource_prev = mass_to_massmix(
                        (
                            batch[f"{molecule}flux_land"]
                            + batch[f"{molecule}flux_ocean"]
                            + batch[f"{molecule}flux_subt"]
                        )
                        * batch["cell_area"]
                        * self.dt,
                        batch["airdensity"][..., :1],
                        batch["volume"][..., :1],
                        ppm=True,
                    )  # b n h
                    preds[f"{molecule}massmix"][..., :1] = (
                        preds[f"{molecule}massmix"][..., :1]
                        + surfflux_as_massmixsource_prev
                    )

                preds[f"{molecule}density"] = massmix_to_density(
                    preds[f"{molecule}massmix"],
                    (
                        batch["airdensity_next"]
                        if not "airdensity" in self.target_vars
                        else preds["airdensity"]
                    ),
                    ppm=True,
                )

        return preds
