from torch_advection import Advection

from neural_transport.models.sfno import *
from neural_transport.tools.conversion import *


class HybridSFNO(RegularGridModel):

    def init_model(
        self,
        embed_dim=256,
        num_layers=8,
        operator_type="driscoll-healy",
        scale_factor=1,
        in_chans=193,
        out_chans=19,
    ) -> None:

        super().__init__()

        self.sfnonet = SphericalFourierNeuralOperatorNet(
            embed_dim=embed_dim,
            num_layers=num_layers,
            operator_type=operator_type,
            scale_factor=scale_factor,
            img_size=(self.nlat, self.nlon),
            in_chans=in_chans,
            out_chans=out_chans,
        )

        hard_thresholding_fraction = 1.0
        modes_lat = int(self.nlat * hard_thresholding_fraction)
        modes_lon = int((self.nlon // 2) * hard_thresholding_fraction)
        modes_lat = modes_lon = min(modes_lat, modes_lon)
        self.adv = Advection(
            self.nlat,
            self.nlon,
            lmax=modes_lat,
            mmax=modes_lon,
            grid="equiangular",
            norm="ortho",
            csphase=True,
        )

        self.earth_radius = 6371 * 1000  # metres

    def model(self, x_in):

        x_out = self.sfnonet(x_in)

        return x_out

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

        u_wind = (
            batch["u"]
            .reshape(B, self.nlat, self.nlon, batch["u"].shape[-1])
            .permute(0, 3, 1, 2)
        )  # b c h w
        v_wind = (
            batch["v"]
            .reshape(B, self.nlat, self.nlon, batch["v"].shape[-1])
            .permute(0, 3, 1, 2)
        )  # b c h w

        preds = {}
        i = 0
        for v in self.target_vars:
            C = batch[v].shape[-1]
            preds[v] = x_out_next[..., i : i + C]
            i += C

            x = (
                batch[v]
                .reshape(B, self.nlat, self.nlon, batch[v].shape[-1])
                .permute(0, 3, 1, 2)
            )  # b c h w

            dx_dt = (
                self.adv(x, u_wind, v_wind).permute(0, 2, 3, 1).reshape(B, N, -1)
                / self.earth_radius
            )

            preds[v] = preds[v] + dx_dt * self.dt

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
