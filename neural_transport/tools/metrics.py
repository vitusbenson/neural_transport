import numpy as np
import torch
import xarray as xr
from torch import nn
from torchmetrics.functional import pearson_corrcoef, r2_score

from neural_transport.tools.conversion import (
    density_to_mass,
)


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
                "weights", torch.from_numpy(weights.astype("float32"))
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


def crps(preds, tests) -> tuple[xr.DataArray | np.ndarray, float]:
    """
    Compute the Continuous Ranked Probability Score (CRPS)
    for ensemble predictions vs. ground truth.

    Parameters
    ----------
    preds : xarray.DataArray or np.ndarray
        Ensemble predictions. Expected shape:
          - with levels: (sample, lat, lon, level)
          - without levels: (sample, lat, lon)
    tests : xarray.DataArray, torch.Tensor, or np.ndarray
        Ground truth / test samples. Shape:
          - with levels: (level, lat, lon)
          - without levels: (lat, lon)

    Returns
    -------
    crps_map : xarray.DataArray or np.ndarray
        CRPS per gridpoint (lat x lon) or (lat x lon x level) if levels present.
    crps_mean : float
        Mean CRPS averaged over all gridpoints.
    """
    if isinstance(preds, xr.DataArray):
        preds = preds.values
    if isinstance(tests, xr.DataArray):
        tests = tests.values
    if torch.is_tensor(tests):
        tests = tests.cpu().numpy()

    if tests.ndim == 3:
        tests = np.moveaxis(tests, 0, -1)  # [lat, lon, level]
        lat_t, lon_t, C_t = tests.shape
        tests_flat = tests.reshape(-1, C_t)  # [N, C]
    elif tests.ndim == 2:
        lat_t, lon_t = tests.shape
        C_t = None
        tests_flat = tests.reshape(-1)  # [N,]

    if preds.ndim == 4:
        # preds: [samples, lat, lon, C]
        S, lat, lon, C = preds.shape
        assert C == C_t, f"Level mismatch: {C} vs {C_t}"
        assert lat == lat_t and lon == lon_t, f"Spatial mismatch: {lat}x{lon} vs {lat_t}x{lon_t}"
        preds_flat = preds.reshape(S, -1, C)  #  [samples, N, C]

    elif preds.ndim == 3:
        # preds: [samples, lat, lon]
        S, lat, lon = preds.shape
        preds_flat = preds.reshape(S, lat*lon)  # [samples, N]
    
    # Compute CRPS across ensemble dimension
    # term1 = mean(|x_i - obs|)
    term1 = np.mean(np.abs(preds_flat - tests_flat[None, ...]), axis=0)  # [N, (C)]
    # term2 = 0.5 * mean(|x_i - x_j|)
    diffs = np.abs(preds_flat[:, None, ...] - preds_flat[None, :, ...])  # [2xsamples, N, (C)]
    term2 = 0.5 * np.mean(diffs, axis=(0, 1))  # [N, (C)]

    crps_flat = term1 - term2  # [N, (C)]
    crps_map = crps_flat.reshape(lat, lon, C_t) if preds.ndim == 4 else crps_flat.reshape(lat, lon)  # [lat, lon, (level)]
    crps_mean = float(np.mean(crps_map, axis=(0, 1)))  # average over lat, lon
    return crps_map, crps_mean


def compute_error_maps(gen_samples, gt):
    """
    Compute spatial bias, RMSE, and ensemble spread maps.

    Parameters
    ----------
    gen_samples : xr.DataArray, shape [n_samples, lat, lon]
        Generated ensemble samples.
    gt : xr.DataArray, shape [lat, lon]
        Ground truth field.

    Returns
    -------
    bias_map, rmse_map, mean_map, spread_map : np.ndarray, shape [lat, lon]
    """
    # Convert xarray to numpy
    gen_samples = gen_samples.values
    gt = gt.values

    mean_map = np.mean(gen_samples, axis=0)

    mask = np.isfinite(gt) & np.isfinite(mean_map)
    gt_masked = np.where(mask, gt, np.nan)
    gen_samples_masked = np.where(mask, gen_samples, np.nan)

    bias_map = np.nanmean(gen_samples_masked, axis=0) - gt_masked
    rmse_map = np.sqrt(np.nanmean((gen_samples_masked - gt_masked)**2, axis=0))
    spread_map = np.nanstd(gen_samples_masked, axis=0)
    mean_map = np.where(mask, mean_map, np.nan)

    return bias_map, rmse_map, mean_map, spread_map


def compute_error_scalars(bias_map, rmse_map, mean_map, spread_map, weights=None):
    """
    Compute scalar error metrics from spatial maps.

    Parameters
    ----------
    bias_map, rmse_map, mean_map, spread_map : np.ndarray, shape [lat, lon]
        Spatial error maps.
    weights : np.ndarray, shape [lat, lon], optional

    Returns
    -------
    bias_scalar, rmse_scalar, mean_scalar, spread_scalar : float
        Mean values of the error metrics.
    """
    if weights is not None:
        weights = weights / np.sum(weights)
        bias_scalar = np.sum(bias_map * weights)
        rmse_scalar = np.sum(rmse_map * weights)
        mean_scalar = np.sum(mean_map * weights)
        spread_scalar = np.sum(spread_map * weights)
    else:
        bias_scalar = np.mean(bias_map)
        rmse_scalar = np.mean(rmse_map)
        mean_scalar = np.mean(mean_map)
        spread_scalar = np.mean(spread_map)

    return bias_scalar, rmse_scalar, mean_scalar, spread_scalar
