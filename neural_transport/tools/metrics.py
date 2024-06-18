import torch
import torch.nn as nn
from torchmetrics.functional import pearson_corrcoef, r2_score

from neural_transport.tools.conversion import *


class PixelwiseMetric(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.vars = list(weights.keys())

        for variable, weight in weights.items():
            self.register_buffer(
                f"weights_{variable}", torch.from_numpy(weight.astype("float32"))
            )  # N, C

    @property
    def name(self):
        raise NotImplementedError

    def compute_metric(self, pred, targ, weights):
        raise NotImplementedError

    def forward(self, preds, batch):
        metrics = {}
        for v in self.vars:
            # if getattr(self, f"weights_{v}").shape[-1] > preds[v].shape[-1]:
            #     print("shape mismatch metric", v, preds[v].shape, getattr(self, f"weights_{v}").shape)

            if "_delta" in v:
                pred_delta = (
                    preds[v.replace("_delta", "")] - batch[v.replace("_delta", "")]
                )
                targ_delta = (
                    batch[v.replace("_delta", "_next")] - batch[v.replace("_delta", "")]
                )
                metric = self.compute_metric(
                    pred_delta, targ_delta, getattr(self, f"weights_{v}")
                )
            else:
                metric = self.compute_metric(
                    preds[v], batch[v], getattr(self, f"weights_{v}")
                )

            if metric.numel() > 1:
                for i, m in enumerate(metric):
                    metrics[f"{self.name}_{v}/l{i}"] = m
            metrics[f"{self.name}_{v}/all"] = metric.mean()

        return metrics


class RMSE(PixelwiseMetric):
    @property
    def name(self):
        return "rmse"

    def compute_metric(self, pred, targ, weights):
        se = (pred - targ) ** 2
        wmse = torch.mean(se * weights, dim=(0, 1, 2))
        return wmse


class RRMSE(PixelwiseMetric):
    @property
    def name(self):
        return "rrmse"

    def compute_metric(self, pred, targ, weights):
        se = (pred - targ) ** 2
        wmse = torch.mean(se * weights, dim=(0, 1, 2)) / (targ * weights).abs().mean()
        return wmse


class RelAbsBias(PixelwiseMetric):
    @property
    def name(self):
        return "rabsbias"

    def compute_metric(self, pred, targ, weights):
        weight_sum = weights.expand_as(pred).sum(dim=(0, 1, 2))
        mean_pred = (pred * weights).sum(dim=(0, 1, 2)) / weight_sum
        mean_targ = (targ * weights).sum(dim=(0, 1, 2)) / weight_sum
        return (mean_pred - mean_targ).abs() / mean_targ


class R2(PixelwiseMetric):
    @property
    def name(self):
        return "r2"

    def compute_metric(self, pred, targ, weights):
        B, T, N, C = pred.shape
        return pearson_corrcoef(pred.reshape(-1, C), targ.reshape(-1, C)) ** 2


class NSE(PixelwiseMetric):
    @property
    def name(self):
        return "nse"

    def compute_metric(self, pred, targ, weights):
        B, T, N, C = pred.shape
        return (
            r2_score(pred.reshape(-1, C), targ.reshape(-1, C), multioutput="raw_values")
            ** 2
        )


class Mass_RMSE(nn.Module):
    def __init__(self, molecule="co2", weights=None):
        super().__init__()
        self.molecule = molecule

        if weights is not None:
            self.register_buffer(
                f"weights", torch.from_numpy(weights.astype("float32"))
            )
        else:
            self.weights = None

    def forward(self, preds, batch):
        mass_pred = density_to_mass(
            preds[f"{self.molecule}density"], batch["volume_next"]
        )
        mass_targ = density_to_mass(
            batch[f"{self.molecule}density_next"], batch["volume_next"]
        )
        if self.weights is not None:
            mass_pred = mass_pred * self.weights
            mass_targ = mass_targ * self.weights
        se = (mass_pred.sum([-1, -2]) - mass_targ.sum([-1, -2])) ** 2
        rmse = torch.sqrt(torch.mean(se))
        return {
            f"mass_rmse_{self.molecule}": rmse,
            f"mass_rrmse_{self.molecule}": rmse / mass_targ.sum([-1, -2]).mean(),
        }

class Mass_RMSEv2(nn.Module):
    def __init__(self, molecule = "co2"):

        super().__init__()
        self.molecule = molecule

    def forward(self, preds, batch):

        mass_pred = (preds[f"{self.molecule}massmix"] / 1e6) * batch["airmass_next"]
        mass_targ = (batch[f"{self.molecule}massmix_next"] / 1e6) * batch["airmass_next"]

        se = (mass_pred.sum([-1, -2]) - mass_targ.sum([-1, -2])) ** 2
        rmse = torch.sqrt(torch.mean(se))
        return {
            f"mass_rmse_{self.molecule}": rmse,
            f"mass_rrmse_{self.molecule}": rmse / mass_targ.sum([-1, -2]).mean(),
        }

METRICS = {
    "rmse": RMSE,
    "r2": R2,
    "nse": NSE,
    "rabsbias": RelAbsBias,
    "rrmse": RRMSE,
    "mass_rmse": Mass_RMSE,
    "mass_rmsev2": Mass_RMSEv2,
}


class ManyMetrics(nn.Module):
    def __init__(self, metrics=[dict(name="mass_rmse", kwargs=dict(molecule="co2"))]):
        super().__init__()

        self.metrics = nn.ModuleList(
            [METRICS[m["name"]](**m["kwargs"]) for m in metrics]
        )

    def forward(self, preds, batch):
        metrics = {}
        for m in self.metrics:
            metrics.update(m(preds, batch))
        return metrics
