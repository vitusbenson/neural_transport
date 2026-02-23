import torch
import torch.nn as nn

from neural_transport.models.layers import (
    MultiScaleDecoder,
    MultiScaleEncoder,
)

class RegularGridModel(nn.Module):

    def __init__(
        self,
        model_kwargs={},
        input_vars=[],
        target_vars=[],
        nlat=45,
        nlon=72,
        nlev=None,
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
        self.nlev = nlev
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

    def normalize_batch(self, batch):
        batch_normalized = {}
        vars_to_normalize = self.input_vars
        for v in vars_to_normalize:
            if v not in batch:
                print(f"WARNING: skipping {v}, missing in batch")
                continue

            try:
                offset = batch[f"{v}_offset"]
                scale = batch[f"{v}_scale"]
            except KeyError as e:
                raise KeyError(
                    f"Missing offset/scale for input variable '{v}' during normalization: {e}."
                    f"Expected keys: {v}_offset, {v}_scale. "
                    f"Available keys: {list(batch.keys())}"
                )
            
            x_in_curr = (batch[v] - offset) / scale

            if self.targshift and (v in self.target_vars):
                batch_normalized[v] = x_in_curr - x_in_curr.mean((1, 2), keepdim=True)
            else:
                batch_normalized[v] = x_in_curr

        return batch_normalized

    def preprocess_inputs(self, batch):

        batch_normalized = self.normalize_batch(batch)

        x_in = torch.cat(list(batch_normalized.values()), dim=-1)

        B, _, C = x_in.shape

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

        return x_in

    def normalize_batch_target_vars(self, batch):
        
        batch_normalized = {}
        for v in self.target_vars:
            for suffix in ['', '_next']:
                key = f"{v}{suffix}"
                if key in batch:
                    mean = batch[f"{v}_offset"]
                    std = batch[f"{v}_scale"]
                    x_in_curr = (batch[key] - mean) / std
                if self.targshift:
                    batch_normalized[key] = x_in_curr - x_in_curr.mean((1, 2), keepdim=True)
                else:
                    batch_normalized[key] = x_in_curr

        return batch_normalized
    

    def normalize_observations(self, obs_values, batch, target_var, targshift=None):
        mean = batch[f"{target_var}_offset"]
        std = batch[f"{target_var}_scale"]

        obs_mask = batch["obs_mask"]
        mask = obs_mask.bool()

        obs_norm = torch.where(
            mask,
            (obs_values - mean) / std,
            obs_values,
        )

        ### Implement targshift with observations stats from larger set, e.g. 16-day window
        if targshift is None:
            targshift = self.targshift
        if targshift:
            mean = torch.nanmean((batch[target_var] - mean) / std, dim=(1, 2), keepdim=True)
            obs_norm = torch.where(mask, obs_norm - mean, obs_norm)

        return obs_norm

    
    def denormalize_tensor(self, x_out, batch):
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
            has_next_keys = all(f"{v}_next" in batch for v in self.target_vars)
            if self.targshift and has_next_keys:
                x_out_next = torch.cat(
                    [batch[f"{v}_next"] for v in self.target_vars],
                    dim=-1,
                )
                x_out_next_normalized = (x_out_next - x_grid_offset) / x_grid_scale
                x_out_next_normalized_mean = x_out_next_normalized.mean((1, 2), keepdim=True)
                x_out_next = (x_out + x_out_next_normalized_mean) * x_grid_scale + x_grid_offset
            else:
                x_out_next = x_out * x_grid_scale + x_grid_offset

        return x_out_next

    def postprocess_outputs(self, x_out, batch, denormalize=True):

        if self.horizontal_interpolation == "multiscale_encoder":
            x_out = self.multiscale_decoder(x_out)
        elif self.horizontal_interpolation is not None:
            x_out = nn.functional.interpolate(
                x_out,
                size=(self.in_nlat, self.in_nlon),
                align_corners=True,
                mode=self.horizontal_interpolation,
            )

        B, N, _ = batch[self.target_vars[0]].shape
        x_out = x_out.permute(0, 2, 3, 1).reshape(B, N, -1)
        
        if denormalize:
            x_out_next = self.denormalize_tensor(x_out, batch)
        else:
            x_out_next = x_out
        
        preds = {}
        i = 0
        for v in self.target_vars:
            C = batch[v].shape[-1]
            preds[v] = x_out_next[..., i : i + C]
            i += C

        for molecule in self.molecules:

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

        return preds
