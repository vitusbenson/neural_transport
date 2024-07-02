import torch
import torch.nn as nn

from neural_transport.tools.conversion import *

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
    ):
        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.input_vars = input_vars
        self.target_vars = target_vars

        self.predict_delta = predict_delta
        self.add_surfflux = add_surfflux
        self.dt = dt
        self.massfixer = massfixer
        self.molecules = molecules
        self.targshift = targshift

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

        to_cat = []
        for v in self.input_vars:
            x_in_curr = (batch[v] - batch[f"{v}_offset"]) / batch[f"{v}_scale"]
            if self.targshift and (v in self.target_vars):
                to_cat.append(x_in_curr - x_in_curr.mean((1, 2), keepdim=True))
            else:
                to_cat.append(x_in_curr)

        x_in = torch.cat(to_cat, dim=-1)

        B, N, C = x_in.shape

        x_in = x_in.reshape(B, self.nlat, self.nlon, C).permute(0, 3, 1, 2)  # b c h w

        # if not x_in.isfinite().all():
        #     print("x_in not finite", x_in.min(), x_in.mean(), x_in.max())

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

            # if (
            #     f"{molecule}density" in preds
            #     and (f"{molecule}massmix" not in preds)
            #     # v.endswith("density")
            #     # and v != "airdensity"
            #     # and v.replace("density", "massmix") not in self.target_vars
            # ):

            #     if self.add_surfflux and (f"{molecule}flux_land" in batch):
            #         surfflux_as_densitysource_prev = mass_to_density(
            #             (
            #                 batch[f"{molecule}flux_land"]
            #                 + batch[f"{molecule}flux_ocean"]
            #                 + batch[f"{molecule}flux_subt"]
            #             )
            #             * batch["cell_area"]
            #             * self.dt,
            #             batch["volume"][..., :1],
            #         )  # b n h
            #         preds[f"{molecule}density"][..., :1] = (
            #             preds[f"{molecule}density"][..., :1]
            #             + surfflux_as_densitysource_prev
            #         )

            #     preds[f"{molecule}massmix"] = density_to_massmix(
            #         preds[f"{molecule}density"],
            #         (
            #             batch["airdensity_next"]
            #             if not "airdensity" in self.target_vars
            #             else preds["airdensity"]
            #         ),
            #         ppm=True,
            #     )
            # if (
            #     f"{molecule}massmix" in preds
            #     and (f"{molecule}density" not in preds)
            #     # v.endswith("massmix")
            #     # and v.replace("massmix", "density") not in self.target_vars
            # ):
            #     if self.massfixer:
            #         cell_area_weights = batch["cell_area"] / batch["cell_area"].sum(
            #             (1,), keepdim=True
            #         )
            #         pl_weights = batch["pressure_height"] / batch[
            #             "pressure_height"
            #         ].sum((2,), keepdim=True)
            #         mean_pred_mass = (
            #             preds[f"{molecule}massmix"] * cell_area_weights * pl_weights
            #         ).sum((1, 2), keepdim=True)
            #         mean_targ_mass = (
            #             batch[f"{molecule}massmix"] * cell_area_weights * pl_weights
            #         ).sum((1, 2), keepdim=True)

            #         if self.massfixer == "shift":
            #             preds[f"{molecule}massmix"] = (
            #                 preds[f"{molecule}massmix"]
            #                 - mean_pred_mass
            #                 + mean_targ_mass
            #             )
            #         elif self.massfixer == "scale":
            #             preds[f"{molecule}massmix"] = (
            #                 preds[f"{molecule}massmix"]
            #                 * mean_targ_mass
            #                 / mean_pred_mass
            #             )

            #         new_pred_mass = (
            #             preds[f"{molecule}massmix"] * cell_area_weights * pl_weights
            #         ).sum((1, 2), keepdim=True)

            #         print(
            #             f"Mass error in {molecule} mass: {mean_targ_mass - new_pred_mass}"
            #         )

            #         preds[f"{molecule}massmix"] = batch[f"{molecule}massmix"].clone()

            #     if self.add_surfflux and (f"{molecule}flux_land" in batch):

            #         surfflux_as_massmixsource_prev = molemix_to_massmix(
            #             (
            #                 batch[f"{molecule}flux_land"]
            #                 + batch[f"{molecule}flux_ocean"]
            #                 + batch[f"{molecule}flux_subt"]
            #             )
            #             * batch["cell_area"]
            #             * self.dt
            #             / 1e12
            #             / 3.664
            #             / 2.124
            #         )
            #         next_targ_mass = (
            #             batch[f"{molecule}massmix_next"]
            #             * cell_area_weights
            #             * pl_weights
            #         ).sum((1, 2), keepdim=True)
            #         # targ_mass2 = batch[f"{molecule}massmix_next"].clone()
            #         # targ_mass2[..., :1] = (
            #         #     targ_mass2[..., :1] + surfflux_as_massmixsource_prev
            #         # )
            #         # flux_targ_mass = (targ_mass2 * cell_area_weights * pl_weights).sum(
            #         #     (1, 2), keepdim=True
            #         # )

            #         # breakpoint()
            #         # surfflux_as_massmixsource_prev = mass_to_massmix(
            #         #     (
            #         #         batch[f"{molecule}flux_land"]
            #         #         + batch[f"{molecule}flux_ocean"]
            #         #         + batch[f"{molecule}flux_subt"]
            #         #     )
            #         #     * batch["cell_area"]
            #         #     * self.dt,
            #         #     batch["airdensity"][..., :1],
            #         #     batch["volume"][..., :1],
            #         #     ppm=True,
            #         # )  # b n h
            #         preds[f"{molecule}massmix"][..., :1] = (
            #             preds[f"{molecule}massmix"][..., :1]
            #             + surfflux_as_massmixsource_prev
            #             / pl_weights[..., :1]
            #             / cell_area_weights
            #         )

            #         flux_pred_mass = (
            #             preds[f"{molecule}massmix"] * cell_area_weights * pl_weights
            #         ).sum((1, 2), keepdim=True)

            #         print(
            #             f"Mass error w/ flux in {molecule} mass: {next_targ_mass - flux_pred_mass}"
            #         )

            #         print(
            #             f"Delta CO2 targ {next_targ_mass - mean_targ_mass}, Delta Co2 flux {surfflux_as_massmixsource_prev.sum()}"
            #         )

            #     preds[f"{molecule}density"] = massmix_to_density(
            #         preds[f"{molecule}massmix"],
            #         (
            #             batch["airdensity_next"]
            #             if not "airdensity" in self.target_vars
            #             else preds["airdensity"]
            #         ),
            #         ppm=True,
            #     )

        return preds
