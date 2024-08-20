import torch
import torch.nn as nn
import torch_harmonics


class MAE(nn.Module):
    def __init__(self, weights={}):
        super().__init__()

        self.vars = list(weights.keys())

        for variable, weight in weights.items():
            self.register_buffer(
                f"weights_{variable}", torch.from_numpy(weight.astype("float32"))
            )  # N, C

    def forward(self, preds, batch):
        loss = 0
        losses = {}
        for v in self.vars:
            se = (preds[v] - batch[f"{v}_next"]).abs()
            wmse = torch.mean(se * getattr(self, f"weights_{v}"))
            losses[f"Loss_Vari/weighted_{v}"] = wmse
            losses[f"Loss_Vari/unweighted{v}"] = torch.mean(se)
            # if getattr(self, f"weights_{v}").shape[-1] > se.shape[-1]:
            #     print(
            #         "shape mismatch loss", v, se.shape, getattr(self, f"weights_{v}").shape
            #     )
            # if self.training and not wmse.requires_grad:
            #     print("no grad", v, preds[v].requires_grad, preds[v].shape)
            # if not torch.isfinite(wmse).all():
            #     print(
            #         v,
            #         wmse,
            #         preds[v].min(),
            #         preds[v].max(),
            #         batch[v].min(),
            #         batch[v].max(),
            #         getattr(self, f"weights_{v}").min(),
            #         getattr(self, f"weights_{v}").max(),
            #     )

            loss = loss + wmse

        return loss, losses


class MSE(nn.Module):
    def __init__(
        self,
        weights={},
        massconserve_weight=0,
        spectral_power_weight=0,
        nlat=32,
        nlon=64,
        cutoff=None,
        scale_by_spectral_power=True,
    ):
        super().__init__()

        self.vars = list(weights.keys())

        for variable, weight in weights.items():
            self.register_buffer(
                f"weights_{variable}", torch.from_numpy(weight.astype("float32"))
            )  # N, C
        self.massconserve_weight = massconserve_weight

        self.spectral_power_weight = spectral_power_weight
        self.scale_by_spectral_power = scale_by_spectral_power
        if self.spectral_power_weight > 0:
            self.cutoff = cutoff or nlat
            self.sht = torch_harmonics.RealSHT(nlat, nlon, grid="equiangular")

    def forward(self, preds, batch):
        loss = 0
        losses = {}
        for v in self.vars:
            se = (preds[v] - batch[f"{v}_next"]) ** 2
            wmse = torch.mean(se * getattr(self, f"weights_{v}"))
            losses[f"Loss_Vari/weighted_{v}"] = wmse
            losses[f"Loss_Vari/unweighted_{v}"] = torch.mean(se)
            # if getattr(self, f"weights_{v}").shape[-1] > se.shape[-1]:
            #     print(
            #         "shape mismatch loss", v, se.shape, getattr(self, f"weights_{v}").shape
            #     )
            # if self.training and not wmse.requires_grad:
            #     print("no grad", v, preds[v].requires_grad, preds[v].shape)
            # if not torch.isfinite(wmse).all():
            #     print(
            #         v,
            #         wmse,
            #         preds[v].min(),
            #         preds[v].max(),
            #         batch[v].min(),
            #         batch[v].max(),
            #         getattr(self, f"weights_{v}").min(),
            #         getattr(self, f"weights_{v}").max(),
            #     )

            loss = loss + wmse

            if ("massmix" in v) and (self.massconserve_weight > 0):

                mass_pred = (preds[v] / 1e6) * batch["airmass_next"]
                mass_targ = (batch[f"{v}_next"] / 1e6) * batch["airmass_next"]

                se = (mass_pred.sum([-1, -2]) - mass_targ.sum([-1, -2])) ** 2
                mse = torch.mean(se)
                wmse = mse * self.massconserve_weight

                losses[f"Loss_Mass/unweighted_{v}"] = mse
                losses[f"Loss_Mass/weighted_{v}"] = wmse

                loss = loss + wmse

            if self.spectral_power_weight > 0:
                B, T, N, C = preds[v].shape
                sh_pred = torch.view_as_real(
                    self.sht(
                        preds[v]
                        .permute(0, 1, 3, 2)
                        .reshape(B, T, C, self.sht.nlat, self.sht.nlon)
                    )[..., : self.cutoff, : self.cutoff]
                )

                sh_targ = torch.view_as_real(
                    self.sht(
                        batch[f"{v}_next"]
                        .permute(0, 1, 3, 2)
                        .reshape(B, T, C, self.sht.nlat, self.sht.nlon)
                    )[..., : self.cutoff, : self.cutoff]
                )

                power_pred = (sh_pred**2).sum([-1, -2])
                power_targ = (sh_targ**2).sum([-1, -2])

                se = (
                    ((power_pred - power_targ) ** 2) / power_targ**2
                    if self.scale_by_spectral_power
                    else ((power_pred - power_targ) ** 2)
                )
                mse = torch.mean(se)
                wmse = mse * self.spectral_power_weight

                losses[f"Loss_SpectralPower/unweighted_{v}"] = mse
                losses[f"Loss_SpectralPower/weighted_{v}"] = wmse

                loss = loss + wmse

        return loss, losses


LOSSES = {
    "mse": MSE,
    "mae": MAE,
}
