import torch
import torch.nn as nn

from neural_transport.models.aroma import AromaDecoder, AromaEncoder
from neural_transport.models.layers import (
    ACTIVATIONS,
    MultiScaleDecoder,
    MultiScaleEncoder,
)
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
        massfixer=None,
        dt=3600,
        molecules=["co2"],
        targshift=False,
        vert_pos_embed=False,
        vert_pos_embed_kwargs=None,
        horizontal_interpolation=None,
        in_nlat=None,
        in_nlon=None,
        multiscale_kwargs=dict(in_chans=5, out_chans=128, layer_norm=True, act="swish"),
    ):
        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.input_vars = input_vars
        self.target_vars = target_vars

        self.in_nlat = in_nlat or nlat
        self.in_nlon = in_nlon or nlon
        self.horizontal_interpolation = horizontal_interpolation

        self.predict_delta = predict_delta
        self.add_surfflux = add_surfflux
        self.dt = dt
        self.massfixer = massfixer
        self.molecules = molecules
        self.targshift = targshift
        self.vert_pos_embed = vert_pos_embed

        if self.vert_pos_embed:
            self.aroma_encoder = AromaEncoder(**vert_pos_embed_kwargs)
            self.aroma_decoder = AromaDecoder(**vert_pos_embed_kwargs)
            self.aroma_decoder.pos_embed = self.aroma_encoder.pos_embed

        if self.horizontal_interpolation == "multiscale_encoder":
            self.multiscale_encoder = MultiScaleEncoder(
                in_shape=(self.in_nlat, self.in_nlon),
                out_shape=(self.nlat, self.nlon),
                **multiscale_kwargs,
            )
            self.multiscale_decoder = MultiScaleDecoder(
                in_shape=(self.in_nlat, self.in_nlon),
                out_shape=(self.nlat, self.nlon),
                **multiscale_kwargs,
            )

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

        batch_normalized = {}
        for v in self.input_vars:
            x_in_curr = (batch[v] - batch[f"{v}_offset"]) / batch[f"{v}_scale"]
            if self.targshift and (v in self.target_vars):
                batch_normalized[v] = x_in_curr - x_in_curr.mean((1, 2), keepdim=True)
            else:
                batch_normalized[v] = x_in_curr

        if self.vert_pos_embed:
            x_in = self.aroma_encoder(batch_normalized)
        else:
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

    def postprocess_outputs(self, x_out, batch):

        B, N, _ = batch[self.target_vars[0]].shape

        if self.horizontal_interpolation == "multiscale_encoder":
            x_out = self.multiscale_decoder(x_out)
        elif self.horizontal_interpolation is not None:
            x_out = nn.functional.interpolate(
                x_out,
                size=(self.in_nlat, self.in_nlon),
                align_corners=True,
                mode=self.horizontal_interpolation,
            )

        x_out = x_out.permute(0, 2, 3, 1).reshape(B, N, -1)

        if self.vert_pos_embed:
            x_out = self.aroma_decoder(x_out, batch)

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

        # if not x_out.isfinite().all():
        #     print("x_out not finite", x_out.min(), x_out.mean(), x_out.max())

        if self.predict_delta:
            x_out_resc = x_out * x_grid_delta_scale + x_grid_delta_offset

            x_out_next = x_out_prev + x_out_resc
        else:
            x_out_next = x_out * x_grid_scale + x_grid_offset

        # if not x_out_next.isfinite().all():
        #     print(
        #         "x_out_next not finite",
        #         x_out_next.min(),
        #         x_out_next.mean(),
        #         x_out_next.max(),
        #     )

        preds = {}
        i = 0
        for v in self.target_vars:
            C = batch[v].shape[-1]
            preds[v] = x_out_next[..., i : i + C]
            i += C

        for molecule in self.molecules:

            # mass_pred_pre_fixer = (
            #     (preds[f"{molecule}massmix"] / 1e6) * batch["airmass_next"]
            # ).sum((1, 2), keepdim=True)

            if self.massfixer and (not self.training):

                mass_pred = (preds[f"{molecule}massmix"]) * batch["airmass_next"]

                mass_old = (batch[f"{molecule}massmix"]) * batch["airmass"]
                if not self.add_surfflux:
                    surfflux_as_masssource = (
                        (
                            batch[f"{molecule}flux_land"]
                            + batch[f"{molecule}flux_ocean"]
                            + batch[f"{molecule}flux_anthro"]
                        )
                        * batch["cell_area"]
                        * self.dt
                        / 1e6  # / 1e12  # PgCO2
                    )
                    B, N, C = mass_old.shape
                    mass_old = mass_old + (
                        surfflux_as_masssource.sum((1, 2), keepdim=True) / (N * C)
                    )
                if self.massfixer == "shift":

                    preds[f"{molecule}massmix"] = (
                        (
                            mass_pred
                            - mass_pred.mean((1, 2), keepdim=True)
                            + mass_old.mean((1, 2), keepdim=True)
                        )
                    ) / batch["airmass_next"]

                elif self.massfixer == "scale":

                    preds[f"{molecule}massmix"] = (
                        (
                            mass_pred
                            * mass_old.mean((1, 2), keepdim=True)
                            / mass_pred.mean((1, 2), keepdim=True)
                        )
                    ) / batch["airmass_next"]

            # mass_pred_after_fixer = (
            #     (preds[f"{molecule}massmix"] / 1e6) * batch["airmass_next"]
            # ).sum((1, 2), keepdim=True)

            if self.add_surfflux:

                surfflux_as_massmixsource_prev = (
                    (
                        batch[f"{molecule}flux_land"]
                        + batch[f"{molecule}flux_ocean"]
                        + batch[f"{molecule}flux_anthro"]
                    )
                    * batch["cell_area"]
                    * self.dt
                    / 1e6  # / 1e12  # PgCO2
                ) / batch["airmass_next"][
                    ..., :1
                ]  # * 1e6

                preds[f"{molecule}massmix"][..., :1] = (
                    preds[f"{molecule}massmix"][..., :1]
                    + surfflux_as_massmixsource_prev
                )

            ### NOTE: Roughly 0.5% Mass Error remains !!!
            ### THIS IS IN THE DATA ALREADY :/ don't know why.

            # virtual_pred = batch[f"{molecule}massmix"].clone()
            # virtual_pred = virtual_pred * batch["airmass"] / batch["airmass_next"]
            # virtual_pred[..., :1] = (
            #     virtual_pred[..., :1] + surfflux_as_massmixsource_prev
            # )

            # mass_pred = (
            #     (preds[f"{molecule}massmix"] / 1e6) * batch["airmass_next"]
            # ).sum((1, 2), keepdim=True)
            # mass_virtual = ((virtual_pred / 1e6) * batch["airmass_next"]).sum(
            #     (1, 2), keepdim=True
            # )
            # mass_targ = (
            #     (batch[f"{molecule}massmix_next"] / 1e6) * batch["airmass_next"]
            # ).sum((1, 2), keepdim=True)
            # mass_old = ((batch[f"{molecule}massmix"] / 1e6) * batch["airmass"]).sum(
            #     (1, 2), keepdim=True
            # )

            # rmse_mass = ((mass_targ - mass_pred) ** 2).mean() ** 0.5
            # rmse_pre = ((mass_targ - mass_pred_pre_fixer) ** 2).mean() ** 0.5
            # rmse_zero = ((mass_old - mass_pred_after_fixer) ** 2).mean() ** 0.5
            # rmse_post = ((mass_targ - mass_pred_after_fixer) ** 2).mean() ** 0.5
            # rmse_virtual = ((mass_targ - mass_virtual) ** 2).mean() ** 0.5

            # rmse_delta = ((mass_targ - mass_old) ** 2).mean() ** 0.5

            # print(
            #     f"RMSE in {molecule} mass: {rmse_mass:.5f}, RMSE Delta: {rmse_delta:.5f}, RMSE PreFixer: {rmse_pre:.5f}, RMSE PostFixer: {rmse_post:.5f}, RMSE Virtual {rmse_virtual:.5f}, RMSE Zero {rmse_zero:.5f}"
            # )
            # breakpoint()

        return preds
