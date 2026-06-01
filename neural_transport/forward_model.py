"""XCO2 forward model: single source of truth for H(x) = column-averaged CO2.

Consolidates the duplicated forward model from:
- MaskedVelocityWrapper.compute_xco2() (flowmatching.py)
- FlowDPSSampler.compute_xco2() (posterior_samplers.py)
- compute_xco2_column() (metrics.py) — kept as-is (NumPy, physical space)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def build_interp_matrix(
    p_src: Tensor,
    p_dst: Tensor,
    *,
    log_pressure: bool = True,
) -> Tensor:
    """Linear-in-(log-)pressure interpolation matrix ``W`` with ``x_dst = W @ x_src``.

    Builds the weights that interpolate a profile defined on source pressure
    levels ``p_src`` to destination pressure levels ``p_dst``. Used to map the
    model's CO2 profile (on the coarse model grid) onto the retrieval's native
    pressure levels before applying the averaging-kernel formula (the
    MIP-prescribed "interpolate-then-apply" operator).

    The weights depend only on the pressures (not on the profile values), so the
    resulting operator ``x -> W @ x`` is linear and differentiable in ``x``.
    Out-of-range destination levels are handled by clamping to the nearest
    source level (constant extrapolation), which keeps every row of ``W`` a
    convex combination (rows sum to 1).

    Parameters
    ----------
    p_src : Tensor
        Source (model) pressure levels, shape ``[..., S]``. Need not be sorted.
    p_dst : Tensor
        Destination (retrieval) pressure levels, shape ``[..., D]``. Leading
        dims must broadcast with ``p_src``'s leading dims.
    log_pressure : bool
        Interpolate linearly in ``log(p)`` (default, physically natural for
        mixing ratios) rather than in ``p``.

    Returns
    -------
    Tensor
        Interpolation matrix ``W`` of shape ``[..., D, S]`` such that
        ``(W @ x_src)`` gives the profile on ``p_dst``.
    """
    ps = torch.log(p_src) if log_pressure else p_src
    pd = torch.log(p_dst) if log_pressure else p_dst

    # Sort source levels ascending along the level axis (keep a mapping back).
    ps_sorted, sort_idx = torch.sort(ps, dim=-1)
    S = ps_sorted.shape[-1]

    # Broadcast source levels against each destination level: [..., D, S].
    ps_b = ps_sorted.unsqueeze(-2)  # [..., 1, S]
    pd_b = pd.unsqueeze(-1)  # [..., D, 1]

    # Bracket index: largest src level <= dst level, clamped to [0, S-2].
    # (count of src levels strictly below dst) - 1
    left = (ps_b < pd_b).sum(dim=-1) - 1  # [..., D]
    left = left.clamp(0, S - 2)
    right = left + 1

    p_left = torch.gather(ps_sorted, -1, left)  # [..., D]
    p_right = torch.gather(ps_sorted, -1, right)
    denom = (p_right - p_left).clamp_min(torch.finfo(ps.dtype).eps)
    p_dst_flat = pd  # [..., D]
    t = ((p_dst_flat - p_left) / denom).clamp(0.0, 1.0)  # clamp -> constant extrapolation

    # Scatter (1-t) to `left` and t to `right`, in SORTED source coordinates.
    W_sorted = torch.zeros(*pd.shape, S, device=p_src.device, dtype=ps.dtype)
    W_sorted.scatter_(-1, left.unsqueeze(-1), (1.0 - t).unsqueeze(-1))
    W_sorted.scatter_add_(-1, right.unsqueeze(-1), t.unsqueeze(-1))

    # Undo the source sort: column j of W_sorted corresponds to sort_idx[..., j].
    inv = torch.argsort(sort_idx, dim=-1)  # [..., S]
    inv_b = inv.unsqueeze(-2).expand(*W_sorted.shape)  # [..., D, S]
    W = torch.gather(W_sorted, -1, inv_b)
    return W


def effective_column_kernel(
    p_model: Tensor,
    p_ret: Tensor,
    pressure_weights: Tensor,
    ak: Tensor,
    co2_profile_prior: Tensor,
    xco2_prior: Tensor,
    *,
    log_pressure: bool = True,
) -> tuple[Tensor, Tensor]:
    """Reduce the MIP "interpolate-then-apply" operator to a model-grid kernel.

    Returns ``(g, column_offset)`` such that, for a model CO2 profile ``x_model``
    (physical ppm, on ``p_model``):

        ``H(x_model) = column_offset + sum_l g_l * x_model_l``
                     ``= xco2_prior + sum_k h_k a_k (W x_model - x_apriori)_k``

    where ``W`` interpolates ``p_model -> p_ret`` and ``g = W^T (h*a)``.  Both the
    standalone :class:`XCO2ForwardModel` and the inline EnKF observation operator
    can consume this, so the corrected operator is a single source of truth.

    Shapes: ``p_model`` ``[C_model]`` or ``[B,C_model,Nlat,Nlon]``; everything
    else ``[B,C_ret,Nlat,Nlon]`` (``xco2_prior`` ``[B,1,Nlat,Nlon]``).
    Returns ``g`` ``[B,C_model,Nlat,Nlon]`` and ``column_offset`` ``[B,1,Nlat,Nlon]``.
    """
    h_a = pressure_weights * ak  # [B, C_ret, Nlat, Nlon]
    B, C_ret, Nlat, Nlon = h_a.shape
    C_model = p_model.shape[-3] if p_model.dim() == 4 else p_model.shape[-1]

    p_ret_l = p_ret.permute(0, 2, 3, 1)  # [B, Nlat, Nlon, C_ret]
    if p_model.dim() == 4:
        p_model_l = p_model.permute(0, 2, 3, 1)
    else:
        p_model_l = p_model.view(1, 1, 1, C_model).expand(B, Nlat, Nlon, C_model)

    W = build_interp_matrix(p_model_l, p_ret_l, log_pressure=log_pressure)  # [B,Nlat,Nlon,C_ret,C_model]
    h_a_l = h_a.permute(0, 2, 3, 1)  # [B, Nlat, Nlon, C_ret]
    g = (W * h_a_l.unsqueeze(-1)).sum(dim=-2).permute(0, 3, 1, 2)  # [B, C_model, Nlat, Nlon]
    column_offset = xco2_prior - (h_a * co2_profile_prior).sum(dim=1, keepdim=True)  # [B,1,Nlat,Nlon]
    return g, column_offset


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
        pressure_weights: Tensor | None,
        ak: Tensor | None,
        xco2_prior: Tensor | None = None,
        co2_profile_prior: Tensor | None = None,
        obs_mean: Tensor | None = None,
        obs_std: Tensor | None = None,
        target_mean: Tensor | None = None,
        target_std: Tensor | None = None,
        targshift_mean: Tensor | None = None,
        nlev: int | None = None,
        column_offset: Tensor | None = None,
    ) -> None:
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
        # Affine "interpolate-then-apply" operator (MIP-correct, P1): when set,
        # H(x_phys) = column_offset + sum_l (h_ak_eff_l * x_phys_l), where
        # h_ak_eff already encodes W^T(h*a) on the model grid. Shape [B,1,Nlat,Nlon].
        self.column_offset = column_offset

        # Eagerly compute h_ak if possible
        self._h_ak = self._compute_h_ak()

    def _compute_h_ak(self) -> Tensor | float | None:  # type: ignore[return-value]
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

    def _get_h_ak_for_x(self, x: Tensor) -> Tensor:
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

        if self.column_offset is not None:
            # Affine "interpolate-then-apply" operator (P1). h_ak already holds
            # the model-grid effective kernel W^T(h*a); column_offset holds
            # xco2_prior - sum(h*a*x_apriori). Linear & differentiable in x.
            x_corrected = x + self.targshift_mean if self.targshift_mean is not None else x
            ts = self.target_std if self.target_std is not None else 1.0
            tm = self.target_mean if self.target_mean is not None else 0.0
            x_phys = x_corrected * ts + tm
            h_ak = self._get_h_ak_for_x(x)
            xco2 = self.column_offset + (h_ak * x_phys).sum(dim=1, keepdim=True)
            om = self.obs_mean if self.obs_mean is not None else 0.0
            os_ = self.obs_std if self.obs_std is not None else 1.0
            return (xco2 - om) / os_
        elif self.has_priors:
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

    def project(
        self,
        x_hat: Tensor,
        obs_values: Tensor,
        obs_mask: Tensor,
        sigma: float,
        spatial_smoothing_sigma: float = 0.0,
        obs_weight: Tensor | None = None,
    ) -> Tensor:
        """Pseudoinverse projection onto column measurement manifold.

        x_hat_k += (h_k * a_k) * (y - H(x_hat)) / (||h*a||^2 + sigma^2)

        Only modifies at observed locations (via obs_mask or obs_weight).

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
        from neural_transport.tools.spatial import gaussian_smooth_2d as _gaussian_smooth_2d

        h_ak = self._get_h_ak_for_x(x_hat)

        # Forward model: H(x_hat)
        xco2_hat = self.forward(x_hat)  # [B, 1, Nlat, Nlon]

        # Column error: y - H(x_hat), weighted by obs_weight or masked by obs_mask
        if obs_weight is not None:
            column_error = obs_weight * (obs_values.detach() - xco2_hat)
        else:
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
    def from_retrieval_levels(
        cls,
        *,
        p_model: Tensor,
        p_ret: Tensor,
        pressure_weights: Tensor,
        ak: Tensor,
        co2_profile_prior: Tensor,
        xco2_prior: Tensor,
        obs_mean: Tensor | None = None,
        obs_std: Tensor | None = None,
        target_mean: Tensor | None = None,
        target_std: Tensor | None = None,
        targshift_mean: Tensor | None = None,
        log_pressure: bool = True,
    ) -> XCO2ForwardModel:
        """MIP-correct "interpolate-then-apply" operator (P1).

        Interpolates the model CO2 profile (on ``p_model``) to the retrieval's
        native pressure levels ``p_ret`` and applies the averaging formula
        there:  ``XCO2 = xco2_prior + sum_k h_k a_k (x_interp_k - x_apriori_k)``.

        Because interpolation ``x_interp = W x_model`` is linear, the whole
        operator is affine in the model state with an *effective* model-grid
        kernel ``g = W^T (h*a)`` and a constant ``column_offset``.  We precompute
        both so ``forward`` / ``jacobian_transpose`` / ``project`` reuse the
        existing machinery (the transpose of ``W`` is baked into ``g``).

        Parameters
        ----------
        p_model : Tensor
            Model pressure levels [hPa], shape ``[C_model]`` or
            ``[B, C_model, Nlat, Nlon]``.
        p_ret : Tensor
            Retrieval native pressure levels [hPa], ``[B, C_ret, Nlat, Nlon]``.
        pressure_weights, ak, co2_profile_prior : Tensor
            Retrieval-level ``h_k``, ``a_k``, ``x_apriori_k``,
            ``[B, C_ret, Nlat, Nlon]``.
        xco2_prior : Tensor
            Retrieval prior column XCO2 [ppm], ``[B, 1, Nlat, Nlon]``.
        """
        g, column_offset = effective_column_kernel(
            p_model, p_ret, pressure_weights, ak, co2_profile_prior, xco2_prior, log_pressure=log_pressure
        )
        return cls(
            pressure_weights=g,
            ak=None,  # h_ak == g directly
            xco2_prior=None,
            co2_profile_prior=None,
            obs_mean=obs_mean,
            obs_std=obs_std,
            target_mean=target_mean,
            target_std=target_std,
            targshift_mean=targshift_mean,
            column_offset=column_offset,
        )

    @classmethod
    def from_masking_config(cls, masking_config: dict) -> XCO2ForwardModel:
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
        # P1: dispatch to the MIP-correct interpolate-then-apply operator when
        # requested and the native-level retrieval fields + pressures are present.
        if (
            masking_config.get("forward_operator", "aggregate") == "interp"
            and masking_config.get("p_model") is not None
            and masking_config.get("p_ret") is not None
            and masking_config.get("ak") is not None
            and masking_config.get("pressure_weights") is not None
            and masking_config.get("co2_profile_prior") is not None
            and masking_config.get("xco2_prior") is not None
        ):
            return cls.from_retrieval_levels(
                p_model=masking_config["p_model"],
                p_ret=masking_config["p_ret"],
                pressure_weights=masking_config["pressure_weights"],
                ak=masking_config["ak"],
                co2_profile_prior=masking_config["co2_profile_prior"],
                xco2_prior=masking_config["xco2_prior"],
                obs_mean=masking_config.get("obs_mean", None),
                obs_std=masking_config.get("obs_std", None),
                target_mean=masking_config.get("target_mean", None),
                target_std=masking_config.get("target_std", None),
                targshift_mean=masking_config.get("targshift_mean", None),
                log_pressure=masking_config.get("log_pressure", True),
            )
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
