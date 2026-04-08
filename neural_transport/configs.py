"""Typed configuration dataclasses for neural_transport.

Single source of truth for all configuration defaults, magic numbers, and
hyperparameters. Bridge functions provide backward compatibility with existing
dict-based APIs.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field

import yaml

# ── Module-level constants (previously hardcoded magic numbers) ──────────

DT_FALLBACK = 0.1  # flowmatching.py:235
DEFAULT_T = 5  # generative.py:578
SATELLITE_TILT_RAD = -5 * math.pi / 180  # generative.py:389,471
SWATH_SPACING_FACTOR = 8  # generative.py:388,469
DEFAULT_TARGET_PRESSURES = (1013, 843, 441, 73)
MAX_N_DISTRIBUTIONAL = 200  # distributional_metrics.py:50,79,309,353

# Known sampler names (None = ODE baseline)
KNOWN_SAMPLERS = {"flowdps", "sde", "fig", "ictm", "mcg"}


# ── Dataclasses ──────────────────────────────────────────────────────────


@dataclass
class SamplerParams:
    """All sampler hyperparameters. Some fields are unused depending on the sampler."""

    # Shared
    sigma_obs: float = 0.1
    spatial_smoothing_sigma: float = 0.0
    soft_boundary_sigma: float = 0.0  # Gaussian blur of obs mask → soft [0,1] weights (0 = binary mask)
    fresh_noise: bool = True

    # SDE
    sigma_max: float = 0.5
    noise_schedule: str = "annealed"
    n_corrector_steps: int = 0
    corrector_step_size: float = 0.01
    corrector_snr: float = 0.16
    use_projection: bool = True

    # FIG
    k_steps: int = 1
    step_size_c: float = 10.0
    noise_scale_w: float = 0.0
    skip_first_last: bool = True

    # ICTM
    r_max: float = 1.0
    r_schedule: str = "decreasing"
    n_inner_steps: int = 1
    inner_lr: float = 0.1

    # MCG (Manifold Constrained Gradient)
    n_forward_steps: int = 1  # 1 = single Tweedie, >1 = multi-step Euler forward shooting


@dataclass
class ConditioningConfig:
    """Masking/conditioning behavior defaults from flowmatching.py:65-71."""

    masking: bool = False
    mask_source: str = "3d"
    mask_pattern: str = "vertical"
    obs_fraction: float = 0.2
    masking_method: str = "interpolate"
    masking_time: str | None = None
    t_threshold: float = 0.9
    conditioning_mode: str = "correction"
    guidance_scale: float = 1.0
    condition_one_timestep: bool = True


@dataclass
class GenerateConfig:
    """Top-level generation configuration."""

    n_samples: int = 10
    steps: int = 11
    method: str = "midpoint"
    time_grid_spacing: str = "uniform"
    refine_start: float = 1.0
    sampler: str | None = None
    sampler_params: SamplerParams = field(default_factory=SamplerParams)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    noise_pattern: str | None = None
    analyze_noise: bool = False
    analyze_masking: bool = False
    obs_var: str = "co2massmix"
    avg_over_levels: bool = True

    def __post_init__(self):
        if self.sampler is not None and self.sampler not in KNOWN_SAMPLERS:
            raise ValueError(f"Unknown sampler {self.sampler!r}. Must be one of {KNOWN_SAMPLERS} or None.")
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        # Coerce nested dicts to dataclass instances
        if isinstance(self.sampler_params, dict):
            self.sampler_params = SamplerParams(**self.sampler_params)
        if isinstance(self.conditioning, dict):
            self.conditioning = ConditioningConfig(**self.conditioning)

    def to_dict(self) -> dict:
        """Convert to nested dict (serializable)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> GenerateConfig:
        """Reconstruct from nested dict."""
        d = copy.deepcopy(d)
        if "sampler_params" in d and isinstance(d["sampler_params"], dict):
            d["sampler_params"] = SamplerParams(**d["sampler_params"])
        if "conditioning" in d and isinstance(d["conditioning"], dict):
            d["conditioning"] = ConditioningConfig(**d["conditioning"])
        return cls(**d)

    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> GenerateConfig:
        """Deserialize from YAML string."""
        d = yaml.safe_load(yaml_str)
        return cls.from_dict(d)

    def merge(self, **overrides) -> GenerateConfig:
        """Return a new GenerateConfig with overrides applied.

        Supports:
        - Flat keys: ``cfg.merge(steps=21)``
        - Dotted keys: ``cfg.merge(**{"sampler_params.sigma_obs": 0.5})``
        - Nested dicts: ``cfg.merge(sampler_params={"sigma_obs": 0.5})``
        """
        d = self.to_dict()

        for key, value in overrides.items():
            if "." in key:
                # Dotted key: "sampler_params.sigma_obs" -> d["sampler_params"]["sigma_obs"]
                parts = key.split(".")
                target = d
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
            elif key in d and isinstance(d[key], dict) and isinstance(value, dict):
                # Nested dict merge
                d[key].update(value)
            else:
                d[key] = value

        return GenerateConfig.from_dict(d)


@dataclass
class DataConfig:
    """Dataset specification."""

    dataset: str = "carbontracker"
    grid: str = "latlon5.625"
    vertical_levels: str = "l10"
    freq: str = "6h"
    target_vars: list[str] = field(default_factory=lambda: ["co2massmix"])
    forcing_vars: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DataConfig:
        return cls(**copy.deepcopy(d))


@dataclass
class EvalConfig:
    """Evaluation settings."""

    max_n_distributional: int = MAX_N_DISTRIBUTIONAL
    n_gt_samples: int = 5
    n_gen_samples: int = 5

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> EvalConfig:
        return cls(**copy.deepcopy(d))


@dataclass
class PlotConfig:
    """Plotting settings."""

    imgformats: list[str] = field(default_factory=lambda: ["pdf"])
    dpi: int = 150
    target_pressures: tuple[int, ...] = DEFAULT_TARGET_PRESSURES
    save_dir: str = "plots"
    figsize_scale: float = 1.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PlotConfig:
        d = copy.deepcopy(d)
        if "target_pressures" in d and isinstance(d["target_pressures"], list):
            d["target_pressures"] = tuple(d["target_pressures"])
        return cls(**d)


# ── Bridge functions ─────────────────────────────────────────────────────

# Fields that belong to SamplerParams
_SAMPLER_PARAMS_FIELDS = {f.name for f in dataclasses.fields(SamplerParams)}
# Fields that belong to ConditioningConfig
_CONDITIONING_FIELDS = {f.name for f in dataclasses.fields(ConditioningConfig)}
# Fields that belong to GenerateConfig (top-level, excluding nested dataclass fields)
_GENERATE_FIELDS = {f.name for f in dataclasses.fields(GenerateConfig)} - {
    "sampler_params",
    "conditioning",
}


def compat_to_generate_kwargs(config: GenerateConfig) -> dict:
    """Flatten a GenerateConfig into a dict matching the legacy generate_kwargs API.

    Sampler params and conditioning fields are promoted to top-level keys,
    which is what inference_forward(), MaskedVelocityWrapper(**generate_kwargs),
    and sampler constructors expect.
    """
    d = {}

    # Top-level GenerateConfig fields
    for f in dataclasses.fields(GenerateConfig):
        if f.name in ("sampler_params", "conditioning"):
            continue
        d[f.name] = getattr(config, f.name)

    # Flatten sampler_params
    for f in dataclasses.fields(SamplerParams):
        d[f.name] = getattr(config.sampler_params, f.name)

    # Flatten conditioning
    for f in dataclasses.fields(ConditioningConfig):
        d[f.name] = getattr(config.conditioning, f.name)

    return d


def compat_from_generate_kwargs(d: dict) -> GenerateConfig:
    """Reconstruct a GenerateConfig from a flat legacy generate_kwargs dict.

    Routes keys to the correct nested dataclass based on field names.
    Handles ``_sampler`` sentinel -> ``sampler`` field.
    Ignores unknown keys (e.g., ``_unconditional``).
    """
    d = dict(d)  # shallow copy

    # Handle _sampler sentinel
    if "_sampler" in d:
        d["sampler"] = d.pop("_sampler")

    sampler_kwargs = {}
    conditioning_kwargs = {}
    generate_kwargs = {}

    for key, value in d.items():
        if key in _SAMPLER_PARAMS_FIELDS:
            sampler_kwargs[key] = value
        elif key in _CONDITIONING_FIELDS:
            conditioning_kwargs[key] = value
        elif key in _GENERATE_FIELDS:
            generate_kwargs[key] = value
        # else: ignore unknown keys (e.g., _unconditional)

    if sampler_kwargs:
        generate_kwargs["sampler_params"] = SamplerParams(**sampler_kwargs)
    if conditioning_kwargs:
        generate_kwargs["conditioning"] = ConditioningConfig(**conditioning_kwargs)

    return GenerateConfig(**generate_kwargs)
