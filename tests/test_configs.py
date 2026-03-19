"""Tests for the typed configuration system (Phase 2)."""

import math

import pytest
import yaml

from neural_transport.configs import (
    DEFAULT_T,
    DEFAULT_TARGET_PRESSURES,
    DT_FALLBACK,
    MAX_N_DISTRIBUTIONAL,
    SATELLITE_TILT_RAD,
    SWATH_SPACING_FACTOR,
    ConditioningConfig,
    DataConfig,
    EvalConfig,
    GenerateConfig,
    PlotConfig,
    SamplerParams,
    compat_from_generate_kwargs,
    compat_to_generate_kwargs,
)

# ── TestSamplerParams ────────────────────────────────────────────────────


class TestSamplerParams:
    def test_defaults_match_flowmatching(self):
        """Defaults must match flowmatching.py:682-745 .get() defaults."""
        sp = SamplerParams()
        # Shared
        assert sp.sigma_obs == 0.1
        assert sp.spatial_smoothing_sigma == 0.0
        assert sp.fresh_noise is True
        # SDE
        assert sp.sigma_max == 0.5
        assert sp.noise_schedule == "annealed"
        assert sp.n_corrector_steps == 0
        assert sp.corrector_step_size == 0.01
        assert sp.corrector_snr == 0.16
        assert sp.use_projection is True
        # FIG
        assert sp.k_steps == 1
        assert sp.step_size_c == 10.0
        assert sp.noise_scale_w == 0.0
        assert sp.skip_first_last is True
        # ICTM
        assert sp.r_max == 1.0
        assert sp.r_schedule == "decreasing"
        assert sp.n_inner_steps == 1
        assert sp.inner_lr == 0.1


# ── TestConditioningConfig ───────────────────────────────────────────────


class TestConditioningConfig:
    def test_defaults_match_flowmatching(self):
        """Defaults must match flowmatching.py:65-71 .get() defaults."""
        cc = ConditioningConfig()
        assert cc.masking is False
        assert cc.mask_source == "3d"
        assert cc.mask_pattern == "vertical"
        assert cc.obs_fraction == 0.2
        assert cc.masking_method == "interpolate"
        assert cc.masking_time is None
        assert cc.t_threshold == 0.9
        assert cc.conditioning_mode == "correction"
        assert cc.guidance_scale == 1.0
        assert cc.condition_one_timestep is True


# ── TestGenerateConfig ───────────────────────────────────────────────────


class TestGenerateConfig:
    def test_defaults(self):
        cfg = GenerateConfig()
        assert cfg.n_samples == 10
        assert cfg.steps == 11
        assert cfg.method == "midpoint"
        assert cfg.time_grid_spacing == "uniform"
        assert cfg.refine_start == 1.0
        assert cfg.sampler is None
        assert isinstance(cfg.sampler_params, SamplerParams)
        assert isinstance(cfg.conditioning, ConditioningConfig)
        assert cfg.noise_pattern is None
        assert cfg.analyze_noise is False
        assert cfg.analyze_masking is False
        assert cfg.obs_var == "co2massmix"
        assert cfg.avg_over_levels is True

    def test_sampler_validation(self):
        with pytest.raises(ValueError, match="Unknown sampler"):
            GenerateConfig(sampler="bogus")

    def test_known_samplers_accepted(self):
        for s in ("flowdps", "sde", "fig", "ictm"):
            cfg = GenerateConfig(sampler=s)
            assert cfg.sampler == s

    def test_negative_steps_validation(self):
        with pytest.raises(ValueError, match="steps must be >= 1"):
            GenerateConfig(steps=0)

    def test_merge_flat(self):
        cfg = GenerateConfig()
        cfg2 = cfg.merge(steps=21)
        assert cfg2.steps == 21
        assert cfg.steps == 11  # original unchanged

    def test_merge_dotted(self):
        cfg = GenerateConfig()
        cfg2 = cfg.merge(**{"sampler_params.sigma_obs": 0.5})
        assert cfg2.sampler_params.sigma_obs == 0.5
        assert cfg.sampler_params.sigma_obs == 0.1

    def test_merge_nested_dict(self):
        cfg = GenerateConfig()
        cfg2 = cfg.merge(sampler_params={"sigma_obs": 0.5})
        assert cfg2.sampler_params.sigma_obs == 0.5
        # Other sampler_params fields should remain at defaults
        assert cfg2.sampler_params.fresh_noise is True

    def test_merge_does_not_mutate_original(self):
        cfg = GenerateConfig()
        original_sigma = cfg.sampler_params.sigma_obs
        cfg.merge(**{"sampler_params.sigma_obs": 999.0})
        assert cfg.sampler_params.sigma_obs == original_sigma

    def test_to_dict_from_dict_roundtrip(self):
        cfg = GenerateConfig(
            sampler="sde",
            steps=21,
            sampler_params=SamplerParams(sigma_obs=0.5, sigma_max=0.3),
            conditioning=ConditioningConfig(masking=True, guidance_scale=2.0),
        )
        d = cfg.to_dict()
        cfg2 = GenerateConfig.from_dict(d)
        assert cfg2.to_dict() == d

    def test_yaml_roundtrip(self):
        cfg = GenerateConfig(
            sampler="fig",
            sampler_params=SamplerParams(k_steps=3, step_size_c=20.0),
        )
        yaml_str = cfg.to_yaml()
        cfg2 = GenerateConfig.from_yaml(yaml_str)
        assert cfg2.to_dict() == cfg.to_dict()
        # Verify it's valid YAML
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict)

    def test_compat_to_generate_kwargs(self):
        cfg = GenerateConfig(
            sampler="flowdps",
            steps=21,
            sampler_params=SamplerParams(sigma_obs=0.5),
            conditioning=ConditioningConfig(masking=True, guidance_scale=2.0),
        )
        d = compat_to_generate_kwargs(cfg)
        # Sampler params flattened to top level
        assert d["sigma_obs"] == 0.5
        assert d["fresh_noise"] is True
        # Conditioning flattened to top level
        assert d["masking"] is True
        assert d["guidance_scale"] == 2.0
        # Top-level fields present
        assert d["sampler"] == "flowdps"
        assert d["steps"] == 21
        # No nested dicts
        assert "sampler_params" not in d
        assert "conditioning" not in d

    def test_compat_from_generate_kwargs(self):
        flat = {
            "_sampler": "sde",
            "sigma_obs": 0.3,
            "sigma_max": 0.7,
            "masking": True,
            "guidance_scale": 0.5,
            "steps": 15,
            "_unconditional": True,  # should be ignored
        }
        cfg = compat_from_generate_kwargs(flat)
        assert cfg.sampler == "sde"
        assert cfg.sampler_params.sigma_obs == 0.3
        assert cfg.sampler_params.sigma_max == 0.7
        assert cfg.conditioning.masking is True
        assert cfg.conditioning.guidance_scale == 0.5
        assert cfg.steps == 15

    def test_compat_roundtrip(self):
        cfg = GenerateConfig(
            sampler="flowdps",
            sampler_params=SamplerParams(sigma_obs=0.5),
            conditioning=ConditioningConfig(masking=True),
        )
        flat = compat_to_generate_kwargs(cfg)
        cfg2 = compat_from_generate_kwargs(flat)
        assert cfg2.sampler == cfg.sampler
        assert cfg2.sampler_params.sigma_obs == cfg.sampler_params.sigma_obs
        assert cfg2.conditioning.masking == cfg.conditioning.masking

    def test_dict_coercion_in_constructor(self):
        """Passing dicts for nested fields should auto-coerce to dataclasses."""
        cfg = GenerateConfig(
            sampler_params={"sigma_obs": 0.3},
            conditioning={"masking": True},
        )
        assert isinstance(cfg.sampler_params, SamplerParams)
        assert cfg.sampler_params.sigma_obs == 0.3
        assert isinstance(cfg.conditioning, ConditioningConfig)
        assert cfg.conditioning.masking is True


# ── TestDataConfig ───────────────────────────────────────────────────────


class TestDataConfig:
    def test_defaults(self):
        dc = DataConfig()
        assert dc.dataset == "carbontracker"
        assert dc.grid == "latlon5.625"
        assert dc.vertical_levels == "l10"
        assert dc.freq == "6h"
        assert dc.target_vars == ["co2massmix"]
        assert dc.forcing_vars == []

    def test_roundtrip(self):
        dc = DataConfig(dataset="test", target_vars=["a", "b"])
        dc2 = DataConfig.from_dict(dc.to_dict())
        assert dc2.to_dict() == dc.to_dict()


# ── TestMaskingConstants ─────────────────────────────────────────────────


class TestMaskingConstants:
    def test_dt_fallback(self):
        assert DT_FALLBACK == 0.1

    def test_default_t(self):
        assert DEFAULT_T == 5

    def test_satellite_tilt_rad(self):
        assert SATELLITE_TILT_RAD == pytest.approx(-5 * math.pi / 180)

    def test_swath_spacing_factor(self):
        assert SWATH_SPACING_FACTOR == 8

    def test_default_target_pressures(self):
        assert DEFAULT_TARGET_PRESSURES == (1013, 843, 441, 73)

    def test_max_n_distributional(self):
        assert MAX_N_DISTRIBUTIONAL == 200


# ── TestEvalConfig ───────────────────────────────────────────────────────


class TestEvalConfig:
    def test_defaults(self):
        ec = EvalConfig()
        assert ec.max_n_distributional == 200
        assert ec.n_gt_samples == 5
        assert ec.n_gen_samples == 5

    def test_roundtrip(self):
        ec = EvalConfig(max_n_distributional=100)
        ec2 = EvalConfig.from_dict(ec.to_dict())
        assert ec2.to_dict() == ec.to_dict()


# ── TestPlotConfig ───────────────────────────────────────────────────────


class TestPlotConfig:
    def test_defaults(self):
        pc = PlotConfig()
        assert pc.imgformats == ["pdf"]
        assert pc.dpi == 150
        assert pc.target_pressures == (1013, 843, 441, 73)
        assert pc.save_dir == "plots"
        assert pc.figsize_scale == 1.0

    def test_roundtrip(self):
        pc = PlotConfig(dpi=300, target_pressures=(500, 200))
        d = pc.to_dict()
        pc2 = PlotConfig.from_dict(d)
        assert pc2.dpi == 300
        assert pc2.target_pressures == (500, 200)
