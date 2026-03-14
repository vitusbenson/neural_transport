"""XCO2 forward model: single source of truth for H(x) = column-averaged CO2.

Consolidates the duplicated forward model from:
- MaskedVelocityWrapper.compute_xco2() (flowmatching.py)
- FlowDPSSampler.compute_xco2() (posterior_samplers.py)
- compute_xco2_column() (metrics.py) — kept as-is (NumPy, physical space)
"""

import numpy as np
import torch
from torch import Tensor


class XCO2ForwardModel:
    """OCO-2 XCO2 forward model: H(x) = xco2_prior + sum(h * a * (x - x_prior)).

    Not an nn.Module — plain class, no learnable parameters.
    h_ak is computed eagerly in __init__ and cached.

    Parameters
    ----------
    pressure_weights : Tensor or None
        Pressure layer weights h_k. Shape [B, C, Nlat, Nlon] or broadcastable.
        If None, uses uniform 1/nlev (requires nlev param).
    ak : Tensor or None
        Averaging kernel a_k. Shape [B, C, Nlat, Nlon] or broadcastable.
    xco2_prior : Tensor or None
        Prior column-averaged XCO2 in physical space. Shape [B, 1, Nlat, Nlon].
    co2_profile_prior : Tensor or None
        Prior CO2 profile in physical space. Shape [B, C, Nlat, Nlon].
    obs_mean, obs_std : Tensor or None
        Observation normalization parameters. Shape [B, 1, 1, 1].
    target_mean, target_std : Tensor or None
        Target (CO2 profile) normalization parameters. Shape [B, 1, 1, 1].
    targshift_mean : Tensor or None
        Per-sample spatial mean subtracted by targshift. Shape [B, 1, 1, 1].
    nlev : int or None
        Number of vertical levels. Required when pressure_weights is None.
    """

    def __init__(
        self,
        pressure_weights,
        ak,
        xco2_prior=None,
        co2_profile_prior=None,
        obs_mean=None,
        obs_std=None,
        target_mean=None,
        target_std=None,
        targshift_mean=None,
        nlev=None,
    ):
        self.pressure_weights = pressure_weights
        self.ak = ak
        self.xco2_prior = xco2_prior
        self.co2_profile_prior = co2_profile_prior
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.target_mean = target_mean
        self.target_std = target_std
        self.targshift_mean = targshift_mean
        self.nlev = nlev

        # Eagerly compute h_ak if possible
        self._h_ak = self._compute_h_ak()

    def _compute_h_ak(self):
        """Compute h * ak tensor. Returns None if pressure_weights is None and nlev is None."""
        h = self.pressure_weights
        if h is None and self.nlev is not None:
            h = 1.0 / self.nlev
        elif h is None:
            return None

        if self.ak is not None:
            return h * self.ak
        else:
            # When ak is None, h_ak is just h (scalar or tensor)
            return h

    @property
    def h_ak(self) -> Tensor:
        """h * ak tensor [B, C, Nlat, Nlon] or scalar-like."""
        if self._h_ak is not None:
            return self._h_ak
        raise ValueError("h_ak not available: pressure_weights and nlev are both None")

    @property
    def h_ak_sq_sum(self) -> Tensor:
        """sum_k (h_k * a_k)^2. Shape [B, 1, Nlat, Nlon]."""
        return (self.h_ak**2).sum(dim=1, keepdim=True)

    @property
    def has_priors(self) -> bool:
        """Whether prior information is available for the full forward model."""
        return self.xco2_prior is not None and self.co2_profile_prior is not None

    def _get_h_ak_for_x(self, x):
        """Get h_ak, handling the case where pressure_weights is None (uniform)."""
        if self._h_ak is not None:
            return self._h_ak
        # Fallback: uniform weights, need x shape
        nlev = x.shape[1]
        h = 1.0 / nlev
        if self.ak is not None:
            return h * self.ak
        else:
            return torch.full((1, nlev, 1, 1), 1.0 / nlev, device=x.device, dtype=x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        """H(x): [B,C,Nlat,Nlon] -> [B,1,Nlat,Nlon].

        With priors: XCO2 = xco2_prior + sum(h * a * (x_phys - co2_profile_prior)),
        normalized to observation space.

        Without priors (fallback): weighted sum with mean/std correction.
        """
        h = self.pressure_weights
        if h is None:
            h = 1.0 / x.shape[1]

        if self.has_priors:
            x_corrected = x + self.targshift_mean if self.targshift_mean is not None else x
            x_phys = x_corrected * self.target_std + self.target_mean
            xco2 = self.xco2_prior + (h * self.ak * (x_phys - self.co2_profile_prior)).sum(dim=1, keepdim=True)
            return (xco2 - self.obs_mean) / self.obs_std
        else:
            h_ak = self._get_h_ak_for_x(x)
            h_ak_sum = h_ak.sum(dim=1, keepdim=True).clamp(min=1e-12)
            result = (h_ak * x).sum(dim=1, keepdim=True) + (self.target_mean / self.target_std) * (h_ak_sum - 1.0)
            if self.targshift_mean is not None:
                result = result + self.targshift_mean * h_ak_sum
            return result

    def forward_numpy(self, x: np.ndarray, levels_axis: int = -1) -> np.ndarray:
        """Simple sum(h*a*x) along levels_axis. No normalization (physical space).

        Equivalent to metrics.compute_xco2_column() for the simple case.

        Parameters
        ----------
        x : np.ndarray
            CO2 field with levels along levels_axis.
        levels_axis : int
            Axis corresponding to vertical levels.

        Returns
        -------
        np.ndarray with levels_axis summed out.
        """
        pw = self.pressure_weights
        ak = self.ak
        if pw is None:
            raise ValueError("forward_numpy requires pressure_weights to be set")
        if ak is None:
            raise ValueError("forward_numpy requires ak to be set")

        # Convert tensors to numpy if needed
        if hasattr(pw, 'detach'):
            pw = pw.detach().cpu().numpy()
        if hasattr(ak, 'detach'):
            ak = ak.detach().cpu().numpy()

        pw = np.asarray(pw)
        ak = np.asarray(ak)

        return (pw * ak * x).sum(axis=levels_axis)

    def jacobian_transpose(self, col_error: Tensor) -> Tensor:
        """J^T * error: [B,1,Nlat,Nlon] -> [B,C,Nlat,Nlon].

        For H(x) = sum_k h_k a_k x_k, the Jacobian transpose is h_k * a_k * error.
        """
        return self.h_ak * col_error

    def project(self, x_hat, obs_values, obs_mask, sigma, spatial_smoothing_sigma=0.0):
        """Pseudoinverse projection onto column measurement manifold.

        x_hat_k += (h_k * a_k) * (y - H(x_hat)) / (||h*a||^2 + sigma^2)

        Only modifies at observed locations (via obs_mask).

        Parameters
        ----------
        x_hat : Tensor [B, C, Nlat, Nlon]
            Current estimate.
        obs_values : Tensor [B, 1, Nlat, Nlon]
            Observed XCO2 values (normalized).
        obs_mask : Tensor [B, 1, Nlat, Nlon]
            Boolean mask of observed locations.
        sigma : float
            Observation noise for regularization.
        spatial_smoothing_sigma : float
            Gaussian smoothing of column error (0 = none).

        Returns
        -------
        Tensor [B, C, Nlat, Nlon] — projected estimate.
        """
        from neural_transport.models.flowmatching import _gaussian_smooth_2d

        h_ak = self._get_h_ak_for_x(x_hat)

        # Forward model: H(x_hat)
        xco2_hat = self.forward(x_hat)  # [B, 1, Nlat, Nlon]

        # Column error: y - H(x_hat), zeroed at unobserved locations
        obs_safe = torch.where(obs_mask, obs_values.detach(), torch.zeros_like(xco2_hat))
        column_error = torch.where(obs_mask, obs_safe - xco2_hat, torch.zeros_like(xco2_hat))

        # Optional spatial smoothing of column error
        if spatial_smoothing_sigma > 0:
            column_error = _gaussian_smooth_2d(column_error, spatial_smoothing_sigma)

        # Denominator: sum_j (h_j * a_j)^2 per spatial location
        h_ak_sq_sum = (h_ak**2).sum(dim=1, keepdim=True)
        denom = h_ak_sq_sum + sigma**2

        correction = h_ak * column_error / denom
        return x_hat + correction

    @classmethod
    def from_masking_config(cls, masking_config: dict) -> "XCO2ForwardModel":
        """Bridge constructor extracting params from legacy masking_config dict.

        Parameters
        ----------
        masking_config : dict
            Dict with keys: pressure_weights, ak, xco2_prior, co2_profile_prior,
            obs_mean, obs_std, target_mean, target_std, targshift_mean.

        Returns
        -------
        XCO2ForwardModel
        """
        return cls(
            pressure_weights=masking_config.get("pressure_weights", None),
            ak=masking_config.get("ak", None),
            xco2_prior=masking_config.get("xco2_prior", None),
            co2_profile_prior=masking_config.get("co2_profile_prior", None),
            obs_mean=masking_config.get("obs_mean", None),
            obs_std=masking_config.get("obs_std", None),
            target_mean=masking_config.get("target_mean", None),
            target_std=masking_config.get("target_std", None),
            targshift_mean=masking_config.get("targshift_mean", None),
        )
