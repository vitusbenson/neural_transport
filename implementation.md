# Flow Matching for CO2 Data Assimilation — Refactor & Development Plan

## Context

We have a working Flow Matching system for CO2 transport across two repos: `neural_transport` (core library) and `carbonbench` (experiment runner). Phases 1–9 of the original implementation plan are complete: evaluation infrastructure, code cleanup, toy OSSE gate, vanilla FM training, and 5 posterior sampling methods (DPS, FlowDPS, SDE, FIG, ICTM).

**Problem**: The codebase has grown organically and now suffers from:
- ~1,800 lines duplicated across 5 `run_ablation.py` scripts in carbonbench (only ~50 lines differ per file)
- ~1,350 lines duplicated across 5 `plot_ablation.py` scripts
- Monolithic `generative.py` (1031 lines) mixing generation, masking, OCO-2 handling, noise patterns
- Two nearly-identical generation functions (`iterative_generate` + `iterative_generate_oco2`) with ~65% code overlap
- `MaskedVelocityWrapper` (260 lines) doing too much: conditioning, forward model, masking, temporal weighting
- XCO2 forward model (`compute_xco2`) duplicated in 3 places: `flowmatching.py`, `posterior_samplers.py`, `metrics.py`
- Raw dicts everywhere (`generate_kwargs`, `masking_config`, `DATA_KWARGS`) with no validation
- Metrics scattered across `metrics.py`, `distributional_metrics.py`, `analyse.py` — plus duplicate metric defs in `plot_results.py` (lines 66-107)
- Plotting functions not composable, no auto figure saving, no publication defaults, no shared pipeline
- No clean data loading API for inference (tied to Lightning `CarbonDataModule`); OCO-2 data loading entangled with generation logic
- Hardcoded magic numbers throughout: `dt=0.1` fallback, `T=5` default, satellite tilt angle `-5*π/180`, swath width `obs_fraction*nlon/8`, target pressures `[1013, 843, 441, 73]`, `max_n=200` energy distance cutoff
- Regular transport model training/eval not integrated with generative eval pipeline
- No structured logging (print statements everywhere)

**Goal**: Refactor into a generalizable, robust, modular API supporting:
1. **Train FM models** on CO2 fields + comprehensive distributional evaluation
2. **OSSE experiments** with synthetic observations (including real OCO-2 masks) + proper scoring & visualization
3. **Real inversion** with OCO-2 XCO2 + evaluation against baselines

...while keeping the regular (deterministic) transport model training fully supported, sharing plots and metrics infrastructure across both model types.

**Development approach**: Test-driven development throughout — each phase writes tests first, then implements to make them pass.

---

## Completed Work (Phases 1-9 of original plan)

<details>
<summary>Click to expand completed phases</summary>

- **Phase 1**: Evaluation infrastructure — `metrics.py`, `conditioning_diagnostics.py`, `osse_runner.py`
- **Phase 2**: Code cleanup — removed 8 duplicate masking methods, fixed `compute_xco2` fallback, fixed guidance gradient
- **Phase 3**: Toy OSSE test gate — `toy_column_osse.py` as pytest, baseline metrics
- **Phase 4**: Vanilla FM training — OT-CFM, Sinkhorn coupling, adaptive ODE, time grid spacing, experiments 09-11
- **Phase 4.5**: Unconditional FM evaluation — distributional metrics, marginal plots, power spectra, tuning experiments
- **Phase 5**: DPS guidance — proper likelihood gradient, Gaussian spatial smoothing, experiment 12
- **Phase 6**: FlowDPS — `FlowDPSSampler`, Tweedie + projection + re-noise, experiment 13
- **Phase 7**: SDE posterior sampling — `StochasticPosteriorSampler`, Langevin corrector, experiment 14
- **Phase 8**: FIG — `FIGSampler`, measurement interpolants, experiment 15
- **Phase 9**: ICTM — `ICTMSampler`, r(t) schedules, closed-form MAP, experiment 16

</details>

---

## Phase 0: Update implementation.md

**Goal**: Replace the current `implementation.md` with this refactored plan.

**Modify**: `/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/neural_transport/implementation.md`

### Checklist
- [x] Rewrite `implementation.md`: completed work summary, refactor phases 1-19, development phases 20-27
- [ ] Commit the updated file

---

## Phase 1: Test Infrastructure & Shared Fixtures

**Goal**: Establish TDD foundation before any refactoring begins. Every subsequent phase writes tests first.

**Create**: `tests/conftest.py`

```python
@pytest.fixture
def mock_velocity_model():
    """Zero-velocity MockSubmodel for testing samplers/wrappers."""

@pytest.fixture(params=[(4, 8, 10), (32, 64, 10)])  # tiny + realistic
def synthetic_co2_field(request):
    """Random CO2 field [B, nlev, nlat, nlon] with physically plausible range."""

@pytest.fixture
def synthetic_pressure_weights():
    """Pressure layer weights for XCO2 computation."""

@pytest.fixture
def synthetic_averaging_kernel():
    """Averaging kernel (ak) for column observation forward model."""

@pytest.fixture
def sample_obs_mask():
    """Binary observation mask [B, 1, nlat, nlon] with ~30% coverage."""

@pytest.fixture
def sample_masking_config():
    """Full masking_config dict matching current API (bridge to new configs)."""
```

**Consolidate**: Move `MockSubmodel` from `test_forward_model.py` into `conftest.py`. Deduplicate any shared setup across the 4 existing test files.

**Create**: `tests/test_smoke.py` — minimal end-to-end smoke test that exercises the current API:
- Create toy model → generate 2 samples → compute metrics → assert no NaN

### Checklist
- [x] Create `conftest.py` with all shared fixtures
- [x] Move `MockSubmodel` and other shared helpers from existing tests into `conftest.py`
- [x] Update existing tests to use shared fixtures (remove duplication)
- [x] Create `test_smoke.py` with E2E smoke test on current API
- [x] All 4 existing test files still pass
- [x] Mark smoke test as `@pytest.mark.quick`

---

## Phase 2: Config System (Typed Dataclasses)

**Goal**: Replace all raw `dict` passing with typed, validated dataclasses. Foundation for everything else.

**Create**: `neural_transport/neural_transport/configs.py`

Key dataclasses:
- `SamplerConfig` — sampler type + all sampler-specific hyperparams (sigma_obs, r_max, k_steps, noise_schedule, etc.)
- `GenerateConfig` — n_samples, steps, masking, conditioning_mode, obs_fraction, noise_pattern, time_grid_spacing + nested `SamplerConfig`
- `DataConfig` — dataset, grid, vertical_levels, freq, target_vars, data_root, obs_source (None / "synthetic" / "oco2")
- `EvalConfig` — which metrics to compute, n_gt/gen_samples for distributional eval, works for both transport and generative
- `PlotConfig` — imgformats, dpi, figsize_scale, save_dir, target_pressures, experiment_type
- `TrainingConfig` — lr, weight_decay, warmup_steps, batch_size, max_steps, loss weighting
- `ExperimentConfig` — name + nested Data/Generate/Eval/Plot/Training configs + device + seed
- `MaskingConstants` — satellite tilt angle, swath width params, default T, temporal weight params (all currently hardcoded)

Each class: `from_dict()`, `to_dict()`, `merge(overrides: dict)`, YAML serialization support.

### Checklist
- [x] **Tests first**: Write `tests/test_configs.py` — serialization roundtrip, defaults, merge with nested overrides, validation (e.g. sampler name must be in known set), YAML round-trip
- [x] Implement `configs.py` with all dataclasses
- [x] Extract all magic numbers from `generative.py` (satellite tilt `-5*π/180`, swath width `obs_fraction*nlon/8`, `T=5` default) and `flowmatching.py` (`dt=0.1` fallback) into `MaskingConstants` / `GenerateConfig`
- [x] Add `compat_to_generate_kwargs()` bridge: `GenerateConfig → dict` for backward compat
- [x] Add `compat_from_generate_kwargs()` bridge: `dict → GenerateConfig`
- [x] Update `toy_column_osse.py` to use configs as proof-of-concept
- [x] All existing tests still pass

---

## Phase 3: Forward Model Extraction

**Goal**: Extract XCO2 forward model from 3 duplicated locations into one standalone, well-tested module.

Currently duplicated in:
- `flowmatching.py` `MaskedVelocityWrapper.compute_xco2` (lines 73-97) — PyTorch
- `posterior_samplers.py` (each sampler has inline column projection) — PyTorch
- `metrics.py` `compute_xco2_column` (lines 23-38) — NumPy

**Create**: `neural_transport/neural_transport/forward_model.py`

```python
class XCO2ForwardModel:
    """H(x) = xco2_prior + sum(h * a * (x - x_prior))"""
    def __init__(self, pressure_weights, ak, xco2_prior=None, co2_profile_prior=None,
                 obs_mean=None, obs_std=None, target_mean=None, target_std=None, targshift_mean=None): ...
    def forward(self, x: Tensor) -> Tensor           # [B,C,Nlat,Nlon] -> [B,1,Nlat,Nlon]
    def forward_numpy(self, x: ndarray) -> ndarray    # for metrics
    def jacobian_transpose(self, col_error) -> Tensor # h_k * a_k * error (used by DPS guidance)
    def project(self, x_hat, obs_values, obs_mask, sigma) -> Tensor  # pseudoinverse projection
    @classmethod
    def from_masking_config(cls, masking_config: dict) -> "XCO2ForwardModel":
        """Bridge constructor from legacy masking_config dict."""
```

### Checklist
- [x] **Tests first**: Extend `tests/test_forward_model.py`:
  - Roundtrip: `H(project(x, y)) ≈ y` at observed locations
  - Linearity: `H(ax + by) = aH(x) + bH(y)`
  - Projection idempotence: `project(project(x)) ≈ project(x)`
  - NumPy/PyTorch parity: `forward(x).numpy() ≈ forward_numpy(x.numpy())`
  - `from_masking_config` produces identical results to current `compute_xco2`
- [x] Implement `XCO2ForwardModel` in `forward_model.py`
- [x] Refactor `MaskedVelocityWrapper.compute_xco2` to delegate to `XCO2ForwardModel`
- [x] Refactor all 4 samplers to use shared `forward_model.project()`
- [x] Refactor `metrics.py` `compute_xco2_column` — kept as-is (simple NumPy one-liner), added cross-reference docstring
- [x] All existing tests still pass (40 in test_forward_model.py, 100 total quick suite)

---

## Phase 4: Sampler Abstract Base Class

**Goal**: Define the sampler interface and shared logic before migrating individual samplers.

**Create**:
- `neural_transport/inference/samplers/__init__.py`
- `neural_transport/inference/samplers/base.py`

```python
class PosteriorSampler(ABC):
    """Abstract base for all posterior samplers."""
    def __init__(self, velocity_model, forward_model: XCO2ForwardModel, config: SamplerConfig): ...

    @abstractmethod
    def sample(self, x_init: Tensor, time_grid: Tensor,
               masking_config: dict, return_intermediates: bool = False) -> Tensor: ...

    # Shared utilities (currently duplicated across FlowDPS/SDE/ICTM):
    def _tweedie_estimate(self, x_t, t, v_theta) -> Tensor:
        """x_hat_1 = x_t + (1-t) * v_theta"""
    def _renoise(self, x_hat, z, t_next) -> Tensor:
        """x_{t+1} = (1-t_{n+1})*z + t_{n+1}*x_hat"""
    def _compute_velocity(self, x_t, t, masking_config) -> Tensor:
        """Wrapper around velocity_model forward pass."""
```

- `neural_transport/inference/samplers/ode.py` — `ODESampler`: wraps standard ODE solver (no conditioning)

### Checklist
- [x] **Tests first**: Write `tests/test_samplers.py` with parametrized interface tests:
  - Each sampler returns correct shape `[B, C, Nlat, Nlon]`
  - No NaN in output
  - `return_intermediates=True` returns `[T, B, C, Nlat, Nlon]`
  - ABC enforcement (BaseSampler/PosteriorSampler cannot be instantiated)
  - Shared method formulas verified numerically (Tweedie, renoise, project)
- [x] Add `mock_velocity_wrapper` fixture to `tests/conftest.py`
- [x] Implement `BaseSampler` ABC and `PosteriorSampler` ABC with shared `_tweedie_estimate`, `_renoise`, `_project_column`
- [x] Implement `ODESampler` wrapping `flow_matching.solver.ODESolver`
- [x] All 18 tests pass, all 118 existing tests pass, ruff clean

---

## Phase 5: Sampler Registry & Migration

**Goal**: Migrate all 4 posterior samplers to the new interface and add registry-based dispatch.

**Create**:
- `neural_transport/inference/samplers/flowdps.py` — from `posterior_samplers.py` FlowDPSSampler
- `neural_transport/inference/samplers/sde.py` — from StochasticPosteriorSampler
- `neural_transport/inference/samplers/fig.py` — from FIGSampler
- `neural_transport/inference/samplers/ictm.py` — from ICTMSampler

**Add to `__init__.py`**:
```python
SAMPLER_REGISTRY = {"ode": ODESampler, "flowdps": FlowDPSSampler, "sde": ..., "fig": ..., "ictm": ...}
def create_sampler(name, velocity_model, forward_model, config) -> PosteriorSampler: ...
```

**Modify**: `flowmatching.py` `inference_forward()` — replace 100-line if/elif with:
```python
sampler = create_sampler(config.sampler, self.velocity_model, self.forward_model, config)
return sampler.sample(x_init, time_grid, masking_config)
```

**Delete**: `posterior_samplers.py`

### Checklist
- [x] **Tests first**: Extend `tests/test_samplers.py` — parametrized over all 5 samplers (ODE + 4 posterior). Test that `create_sampler()` returns correct type.
- [x] Migrate `FlowDPSSampler` to `samplers/flowdps.py` inheriting `PosteriorSampler`
- [x] Migrate `StochasticPosteriorSampler` to `samplers/sde.py`
- [x] Migrate `FIGSampler` to `samplers/fig.py`
- [x] Migrate `ICTMSampler` to `samplers/ictm.py`
- [x] Verify shared `_tweedie_estimate` and `_renoise` replace duplicated code in each sampler
- [x] Implement registry + `create_sampler()` factory
- [x] Refactor `FlowMatching.inference_forward()` to use `create_sampler()`
- [x] Delete `posterior_samplers.py`
- [x] Update imports in `test_toy_column_osse.py` and any other consumers
- [x] All existing tests still pass

---

## Phase 6: Masking Module Extraction

**Goal**: Extract all masking logic from `generative.py` and `flowmatching.py` into a dedicated module.

Currently spread across:
- `generative.py`: `create_mask()`, `create_column_mask()`, `create_oco2_mask()`, satellite patterns (~300 lines)
- `flowmatching.py`: `MaskedVelocityWrapper.masking_*` methods (4 methods, ~80 lines), `_get_temporal_weight()` (~30 lines)

**Create**: `neural_transport/inference/masking.py`

Contents:
- Mask creation: `create_mask()`, `create_column_mask()`, `create_oco2_mask()`, satellite pattern logic
- Masking methods: `masking_simple()`, `masking_interpolate()`, `masking_total_column_average_simple()`, `masking_total_column_average_mult()` (as standalone functions, not methods)
- Temporal weighting: `compute_temporal_weight()` (was `_get_temporal_weight`)
- All use `MaskingConstants` from configs (no more hardcoded values)

### Checklist
- [x] **Tests first**: Write `tests/test_masking.py`:
  - `create_mask()` produces correct shapes and coverage fractions for each pattern
  - Satellite mask respects tilt angle and swath width from config constants
  - `masking_total_column_average_simple` correctly constrains column mean
  - `get_temporal_weight` returns correct schedule shapes (smooth_late, smooth_early, step)
  - Edge cases: obs_fraction=0, obs_fraction=1, single-pixel masks
- [x] Extract mask creation functions from `generative.py` to `masking.py`
- [x] Extract `masking_*` methods from `MaskedVelocityWrapper` to standalone functions in `masking.py`
- [x] Extract `_get_temporal_weight` to `masking.py` (as `get_temporal_weight`)
- [ ] Replace all hardcoded masking constants with `MaskingConstants` (deferred — constants already in `configs.py`)
- [x] Update imports in `generative.py`, `flowmatching.py`
- [x] `MaskedVelocityWrapper` delegates via thin methods to standalone functions
- [x] All existing tests still pass (174 passed)

---

## Phase 7: Noise & Spatial Utilities

**Goal**: Extract noise generation and shared spatial utilities into dedicated modules.

**Create**: `neural_transport/inference/noise.py`
- `generate_noise()` — main entry point
- Noise patterns: `spiral_noise()`, `geodesic_noise()`, `linear_noise()`, `antipodal_noise()`
- Currently embedded in `generative.py` (~80 lines)

**Create**: `neural_transport/tools/spatial.py`
- `gaussian_smooth_2d()` — currently `_gaussian_smooth_2d` in `flowmatching.py` (lines 325-378)
- Used by both `flowmatching.py` (guidance smoothing) and posterior samplers (gradient smoothing)
- Periodic longitude padding, separable 2D convolution

### Checklist
- [x] **Tests first**: Write `tests/test_noise.py`:
  - Each noise pattern produces correct shape
  - No NaN/Inf in outputs
  - Noise patterns are deterministic given seed
- [x] **Tests first**: Write `tests/test_spatial.py`:
  - `gaussian_smooth_2d` with sigma=0 returns input unchanged
  - Periodic boundary: smoothing at lon=0 uses lon=360 data
  - Output shape matches input shape
  - Kernel size = `6*sigma+1` (from current code)
- [x] Extract noise functions to `noise.py`
- [x] Extract `_gaussian_smooth_2d` to `tools/spatial.py`
- [x] Update all imports in `flowmatching.py`, `samplers/*.py`, `generative.py`
- [x] All existing tests still pass

---

## Phase 8: Unified Generation Pipeline

**Goal**: Merge `iterative_generate()` and `iterative_generate_oco2()` into a single `GenerationPipeline`. These share ~65% of code; the key differences are:
- `iterative_generate`: per-sample conditioning, sample-dimension output, synthetic obs
- `iterative_generate_oco2`: time-series with dual dataset alignment, window aggregation, real OCO-2

**Create**: `neural_transport/inference/generation.py`

```python
class GenerationPipeline:
    """Unified generation for all use cases."""
    def __init__(self, model, dataset, *, dataset_gen=None,
                 target_vars_3d=None, target_vars_2d=None,
                 device="cuda", verbose=True): ...

    @property
    def mode(self) -> str:
        """'timeseries' if dataset_gen provided, else 'sample'."""

    def run(self, out_dir, *, zarr_filename=None, freq=None,
            rollout=False, zero_surfflux=False, remap=False,
            save_obs=True, **generate_kwargs) -> xr.Dataset:
        """Single entry point — dispatches based on mode."""

    def run_distributional(self, out_dir, *, n_gt_samples=50, n_gen_samples=200,
                           seed=42, batch_size=20,
                           generate_kwargs=None) -> tuple[xr.Dataset, xr.Dataset]:
        """Replaces generate_for_distributional_eval()."""

    # Shared internal methods:
    def _setup(self, ...): ...                    # output paths, prototype zarr, model config
    def _run_inference(self, model, batch): ...    # model forward + fix list preds
    def _finalize(self, dss, obss, ...): ...       # concat, save, diagnostics
    def _apply_sample_masking(self, ...): ...      # masking dispatch for sample mode
    def _get_grid_coords(self): ...                # lat/lon from grid prototypes

    # Mode-specific methods:
    def _run_sample(self, ...): ...                # per-sample loop (OSSE)
    def _run_timeseries(self, ...): ...            # per-timestep loop (OCO-2)
```

Utility functions: `get_zarrpath_obspath`, `remap_with_cdo`, `get_batches`, `is_bad_sample`, `parse_freq`, `align_time` — all moved from `generative.py`.

Backward-compatible wrappers: `iterative_generate()`, `iterative_generate_oco2()`, `generate_for_distributional_eval()` — thin functions that construct `GenerationPipeline` and delegate.

**Replace**: `generative.py` → thin re-export shim (imports from `generation.py`) for backward compat.

### Checklist
- [x] **Tests first**: Write `tests/test_generation_pipeline.py` (44 tests):
  - `GenerationPipeline.run()` with synthetic data produces correct output shape
  - Sample mode: output has `sample` dimension, correct count, zarr saved
  - Time-series mode: output has `time` dimension, `sample` dimension present
  - `run_distributional()` returns GT and gen datasets with correct sample counts
  - Config with `masking=False` produces unconditional samples (no obs_mask/obs_values)
  - Utility functions: `parse_freq`, `is_bad_sample`, `align_time`, `get_zarrpath_obspath`, `get_batches`
  - Backward-compatible wrapper functions work correctly
- [x] Implement `GenerationPipeline` with shared + mode-specific methods
- [x] Integrate distributional eval as `run_distributional()`
- [x] Replace `generative.py` with thin re-export, update imports in `train.py` and `osse_runner.py`
- [x] Fix pre-existing test failures in `test_toy_column_osse.py` (GenerateConfig→dict conversion, FlowDPSSampler API)
- [ ] Verify: unconditional generation produces identical output to before (requires GPU model)
- [ ] Verify: OSSE conditioning produces identical output (requires GPU model)
- [x] All existing tests still pass (256 quick + 25 toy OSSE = 281 tests, ruff clean)

---

## Phase 9: MaskedVelocityWrapper Slim-Down

**Goal**: Reduce `MaskedVelocityWrapper` from ~260 to ~100 lines. After phases 3, 6, 7 most of its responsibilities have been extracted.

**Remove from `MaskedVelocityWrapper`**:
- `compute_xco2` → delegates to `XCO2ForwardModel` (Phase 3)
- `masking_*` methods → now in `inference/masking.py` (Phase 6)
- `_get_temporal_weight` → now in `inference/masking.py` (Phase 6)
- `_gaussian_smooth_2d` → now in `tools/spatial.py` (Phase 7)

**Keep**: Only the 5 conditioning modes (correction, velocity_projection, guidance, repaint, velocity_projection_repaint) as the forward pass logic.

**Refactor**: `__init__` receives `XCO2ForwardModel` instance instead of constructing internally.

### Checklist
- [x] **Tests first**: Verify all 40 `test_forward_model.py` tests still define expected behavior
- [x] Remove extracted methods from `MaskedVelocityWrapper` (`compute_xco2`, `_get_temporal_weight`, `apply_temporal_weighting`, `compute_dt`)
- [x] Refactor `__init__` to receive `XCO2ForwardModel` via dependency injection
- [x] Store `masking_config` as dict, keep only frequently-accessed shortcuts
- [x] Update `FlowMatching.return_velocity_wrapper()` to construct and pass `XCO2ForwardModel`
- [x] Move `compute_dt()` to `inference/masking.py` as module-level function
- [x] Inline `get_temporal_weight()` calls at 3 call sites
- [x] Simplify `apply_masking()` to pass kwargs from `self.masking_config`
- [x] Update `tests/test_forward_model.py`: return `masking_config` from `_make_wrapper()`, fix attribute accesses
- [x] All 40 `test_forward_model.py` tests pass
- [x] All 256 quick tests pass
- [x] ruff check clean
- [x] `MaskedVelocityWrapper` is now ~123 lines (down from ~174)

---

## Phase 10: Evaluation — Metric Consolidation

**Goal**: Create a single source of truth for all metric functions. Currently metrics are defined in 4 places:
- `inference/metrics.py`: rmse_3d, rmse_xco2, crps_ensemble, spread_skill_ratio, etc.
- `inference/distributional_metrics.py`: energy_distance, mmd_rbf, wasserstein_distance
- `inference/analyse.py`: compute_score_df, compute_score_df_generate (contain inline metric logic)
- `plots/plot_results.py` lines 66-107: rmse(), bias(), r2(), nse() (duplicates!)

**Create** evaluation subpackage:
- `neural_transport/evaluation/__init__.py`
- `neural_transport/evaluation/pointwise.py` — **THE** single source for: RMSE, R2, bias, NSE, RelRMSE, AbsBias, mass balance. Both transport and generative models use these.
- `neural_transport/evaluation/ensemble.py` — CRPS, spread-skill, calibration, rank histogram
- `neural_transport/evaluation/distributional.py` — energy distance, MMD, Wasserstein, KL

### Checklist
- [x] **Tests first**: Write `tests/test_evaluation_metrics.py`:
  - Each metric function: known input → known output (e.g. RMSE of zeros = 0)
  - Metric symmetry/invariance properties where applicable
  - Latitude-weighted RMSE differs from unweighted
  - CRPS with perfect ensemble = 0
  - Energy distance of identical distributions = 0
- [x] Implement `pointwise.py` consolidating all metric definitions
- [x] Implement `ensemble.py` migrating from `metrics.py`
- [x] Implement `distributional.py` migrating from `distributional_metrics.py`
- [x] Delete duplicate metric definitions from `plot_results.py` (lines 66-107), import from `evaluation.pointwise`
- [x] Extract `max_n=200` energy distance cutoff into `EvalConfig` *(done in Phase 2: MAX_N_DISTRIBUTIONAL in configs.py)*
- [x] Update `analyse.py` to import from `evaluation.*` instead of defining inline
- [x] All existing metric tests still pass (`test_distributional_metrics.py`)

---

## Phase 11: Evaluation — EvaluationSuite Orchestrator

**Goal**: Unified orchestrator that runs the right metrics for any experiment type.

**Create**: `neural_transport/evaluation/suite.py`

```python
@dataclass
class EvalResult:
    pointwise: dict[str, float]      # RMSE, R2, bias per variable/level
    ensemble: dict[str, float] | None # CRPS, spread-skill (None for transport)
    distributional: dict[str, float] | None
    maps: dict[str, np.ndarray]      # spatial error maps (RMSE, bias, CRPS)
    diagnostics: dict                # rank histogram data, calibration curves
    metadata: dict                   # experiment name, config, timestamps

class EvaluationSuite:
    """Unified evaluation for transport and generative models."""
    def __init__(self, config: EvalConfig, forward_model: XCO2ForwardModel | None = None): ...
    def evaluate_deterministic(self, preds, gt, lat_weights=None) -> EvalResult:
        """For transport models: RMSE, R2, bias, NSE, mass balance."""
    def evaluate_ensemble(self, samples, gt, mask_2d=None) -> EvalResult:
        """For FM models: pointwise + ensemble metrics (CRPS, spread-skill, rank hist)."""
    def evaluate_distributional(self, gt_pool, gen_pool, lat, lon) -> EvalResult:
        """For unconditional FM: energy distance, MMD, Wasserstein."""
    def to_dataframe(self, result: EvalResult) -> pd.DataFrame:
    def to_json(self, result: EvalResult, path: Path): ...
```

**Modify**:
- `compute_score_df` in `analyse.py` → thin wrapper around `evaluate_deterministic()`
- `compute_score_df_generate` in `analyse.py` → thin wrapper around `evaluate_ensemble()`
- `gen_eval_callback.py` → use `EvaluationSuite` instead of ad-hoc metrics

### Checklist
- [x] **Tests first**: Write `tests/test_evaluation_suite.py` (24 tests):
  - `evaluate_deterministic` returns `EvalResult` with populated `pointwise`, `None` ensemble
  - `evaluate_ensemble` returns `EvalResult` with both `pointwise` and `ensemble`
  - `evaluate_distributional` returns `EvalResult` with `distributional` populated
  - `to_dataframe` produces correct columns
  - `to_json` roundtrips with `from_json`
- [x] Implement `EvaluationSuite` — `evaluation/suite.py`
- [x] Implement `EvalResult` with serialization — `evaluation/suite.py`
- [ ] ~~Migrate `analyse.py` functions to be thin wrappers~~ — **skipped**: `analyse.py` is deeply xarray-specific; `EvaluationSuite` is a parallel numpy-level API
- [x] Refactor `gen_eval_callback.py` to use `EvaluationSuite` (deterministic + direct energy_distance)
- [x] Verify transport model validation still produces correct metrics (317 tests pass)
- [x] All existing tests still pass

---

## Phase 12: Plotting — Base Infrastructure & Always-On Plots

**Goal**: Establish plotting framework and implement plots that are always produced regardless of experiment type.

**Create**: `neural_transport/plots/base.py`

```python
class PlotContext:
    """Manages figure saving, style, and output paths."""
    def __init__(self, config: PlotConfig): ...
    def savefig(self, fig, name: str): ...         # auto-saves to all configured formats
    def subplot_grid(self, nrows, ncols): ...      # consistent sizing
    @contextmanager
    def figure(self, name: str, **kwargs): ...     # auto-save on exit

PLOT_REGISTRY: dict[str, dict] = {}  # name -> {func, categories, description}

def register_plot(name: str, categories: list[str]):
    """Decorator to register a plot function."""

PLOT_CATEGORIES = {
    "always":         ["field_maps", "xco2_maps", "lat_height"],
    "ensemble":       ["spread_maps", "rank_histogram", "calibration"],
    "distributional": ["marginals", "power_spectrum", "qq_plot", "sample_grid"],
    "conditioning":   ["conditioning_comparison", "error_maps", "zonal_mean"],
    "transport":      ["metric_curves", "obspack_stations"],
    "ablation":       ["sweep_plots", "summary_bars", "pareto_front"],
    "animation":      ["field_animation"],
}

def run_plots(result: EvalResult, ctx: PlotContext, categories: list[str] | None = None): ...
```

**Create**: `neural_transport/plots/field_plots.py` — always-on plots shared across all experiment types:
- `plot_field_maps()` — CO2 at key pressure levels (configurable via `PlotConfig.target_pressures`)
- `plot_xco2_maps()` — column-averaged XCO2
- `plot_lat_height()` — zonal-mean latitude-height cross-section
- All accept `EvalResult` and use `PlotContext`

### Checklist
- [x] **Tests first**: Write `tests/test_plots.py`:
  - `PlotContext.savefig` creates files in expected formats
  - `register_plot` adds to `PLOT_REGISTRY`
  - `run_plots(categories=["always"])` calls exactly the "always" plots
  - `run_plots(categories=None)` infers categories from `EvalResult` content
  - Plot functions don't crash with minimal synthetic data
- [x] Implement `PlotContext` with publication rcParams (font sizes, IPCC colormaps, consistent figure sizing)
- [x] Implement `register_plot` decorator and `run_plots` dispatcher
- [x] Implement `field_plots.py` with always-on plots
- [x] Extract hardcoded target pressures `[1013, 843, 441, 73]` into `PlotConfig.target_pressures`
- [x] Verify always-on plots render correctly with sample data
- [x] Added `save_dir` and `figsize_scale` to `PlotConfig`

---

## Phase 13: Plotting — Specialized Plots & Smart Categories

**Goal**: Migrate all existing plot functions to use `PlotContext` + `EvalResult`, organize by category.

**Refactor**: `conditioning_diagnostics.py` → registered in "conditioning" category:
- `plot_conditioning_comparison()` accepts `EvalResult`
- `plot_ensemble_diagnostics()` → split into registered "ensemble" plots
- `plot_zonal_mean()` → registered in "conditioning"

**Refactor**: `distributional_plots.py` → registered in "distributional" category:
- `plot_marginal_distributions()`, `plot_spatial_pattern_comparison()`, `plot_power_spectrum_comparison()`, `plot_qq()`, `plot_lat_height_comparison()`, `plot_sample_grid()` — all accept `EvalResult`, use `PlotContext`

**Create**: `neural_transport/plots/metrics_plots.py` — "ablation" category:
- `plot_ablation_sweep()` — generic sweep plot (metric vs hyperparameter)
- `plot_summary_bars()` — bar chart comparing methods
- `plot_pareto_front()` — wall-time vs RMSE
- Extracted from the 5 carbonbench `plot_ablation.py` scripts

**Create**: `neural_transport/plots/ensemble_plots.py` — "ensemble" category:
- `plot_rank_histogram()`, `plot_calibration_curve()`, `plot_spread_map()`

**Refactor**: `plot_results.py` transport-specific plots → registered in "transport" category:
- `plot_metric_curves()` (RMSE/R2/NSE over lead time)
- `plot_obspack_stations()` (station time series)

**Extract**: `animate_predictions()` → `neural_transport/plots/animation.py`, registered in "animation" category

**Smart inference**: `run_plots()` with `categories=None` auto-selects based on `EvalResult`:
- Has `ensemble` → add "ensemble" category
- Has `distributional` → add "distributional"
- Has `metadata.experiment_type == "transport"` → add "transport"
- Always includes "always"

### Checklist
- [x] **Tests first**: Extend `tests/test_plots.py`:
  - Each refactored plot function accepts `EvalResult` without error
  - Category auto-inference selects correct categories for different `EvalResult` contents
  - Ablation plots are utility functions (not registered), tested separately
- [x] Refactor `conditioning_diagnostics.py` — added 3 registered wrappers (`conditioning_comparison`, `error_maps`, `zonal_mean`)
- [x] Refactor `distributional_plots.py` — added 7 registered wrappers; cartopy guarded with try/except
- [x] Create `metrics_plots.py` — standalone utility functions (`plot_ablation_sweep`, `plot_summary_bars`, `plot_pareto_front`)
- [x] Create `ensemble_plots.py` — 3 registered plots (`rank_histogram`, `calibration`, `spread_maps`)
- [x] Create `transport_plots.py` — 2 registered plots (`metric_curves`, `obspack_stations`)
- [x] Create `animation.py` — registered `field_animation` with lazy import
- [x] Implement smart category inference in `run_plots()` — conditioning, transport, distributional metadata
- [x] Store distributional fields in `EvalResult.metadata` via `evaluate_distributional()`
- [x] Guard `to_json()` against large arrays in metadata
- [ ] Verify all plots visually on one ablation result + one transport result

**Deviations**: Ablation plots (`metrics_plots.py`) are NOT registered — they take different signatures (multiple configs) and don't fit the `(result, ctx)` pattern. Multi-result comparison functions in `conditioning_diagnostics.py` also stay standalone. Legacy plot functions are preserved; registered wrappers delegate to them.

---

## Phase 14: Inference Data Loading

**Goal**: Clean data loading API for generation, decoupled from Lightning.

**Create**: `neural_transport/neural_transport/data/__init__.py`
**Create**: `neural_transport/neural_transport/data/inference_loader.py`

```python
class InferenceDataLoader:
    """Lightweight data loader for generation (no Lightning dependency)."""
    def __init__(self, data_config: DataConfig): ...
    def load_dataset(self) -> CarbonDataset: ...
    def get_batch(self, idx: int, device: str = "cuda") -> dict[str, Tensor]: ...
    def get_gt_field(self, idx: int) -> np.ndarray: ...
    def get_normalization_stats(self) -> dict: ...
    @property
    def grid_info(self) -> GridInfo: ...  # nlat, nlon, lat, lon, levels, cos_lat_weights

@dataclass
class GridInfo:
    nlat: int
    nlon: int
    nlev: int
    lat: np.ndarray
    lon: np.ndarray
    levels: np.ndarray
    cos_lat_weights: np.ndarray  # currently computed ad-hoc in every run_ablation.py
```

Note: `cos_lat_weights` is currently recomputed in every experiment script — this centralizes it.

### Checklist
- [x] **Tests first**: Write `tests/test_data_loading.py`:
  - `InferenceDataLoader` constructs from `DataConfig`
  - `get_batch` returns tensors with correct shapes
  - `grid_info` matches expected dimensions
  - `cos_lat_weights` matches manually computed values
- [x] Implement `InferenceDataLoader` and `GridInfo`
- [x] Update `GenerationPipeline` (Phase 8) to use `InferenceDataLoader`
- [x] Verify generation produces identical output

**Notes**: `GridInfo` provides three cos-lat-weight properties (`cos_lat_weights_2d`, `cos_lat_weights_flat`, `cos_lat_weights` alias). `GenerationPipeline.__init__` accepts both `CarbonDataset` and `InferenceDataLoader` — fully backward compatible. 15 new tests, all passing. `data_path` is a separate arg from `DataConfig` per design.

---

## Phase 15: OCO-2 Data Loading

**Goal**: Separate OCO-2 observation handling from generation logic. The time alignment, window aggregation, and per-sounding AK logic currently live inside `iterative_generate_oco2()` — they belong in the data layer.

**Create**: `neural_transport/neural_transport/data/oco2_loader.py`

```python
@dataclass
class ObservationBatch:
    """Structured observation data (replaces raw dicts in masking_config)."""
    obs_values: Tensor           # XCO2 observations
    obs_mask: Tensor             # Binary mask
    pressure_weights: Tensor | None
    averaging_kernel: Tensor | None
    xco2_prior: Tensor | None
    co2_profile_prior: Tensor | None

class OCO2DataLoader(InferenceDataLoader):
    """Extends InferenceDataLoader with OCO-2 observation handling."""
    def __init__(self, data_config: DataConfig, oco2_dataset=None): ...
    def get_observations(self, time_idx: int, window_steps: int = 1) -> ObservationBatch: ...
    def align_time(self, ct_times, oco2_times) -> int: ...  # offset computation
    def aggregate_window(self, start_idx, window_steps) -> ObservationBatch: ...
```

**Modify**: `GenerationPipeline._collect_timeseries_batch()` now delegates to `OCO2DataLoader.get_observations()` instead of inline logic.

### Checklist
- [x] **Tests first**: Write `tests/test_oco2_loader.py`:
  - `ObservationBatch` has correct tensor shapes
  - `align_time` computes correct offset for known time arrays
  - `aggregate_window` unions sparse observations correctly
  - Window of size 1 returns single-timestep observations
- [x] Implement `ObservationBatch` dataclass
- [x] Implement `OCO2DataLoader` with time alignment and window aggregation
- [x] Update `GenerationPipeline` time-series mode to use `OCO2DataLoader`
- [x] Add `get_window_batch()` to `InferenceDataLoader` base class
- [x] Add `get_window_batch` tests to `tests/test_data_loading.py`
- [x] `align_time` in generation.py delegates to `OCO2DataLoader.align_time`
- [x] Backward compat: `get_batches`, `align_time`, `dataset_gen` raw path all preserved
- [ ] Verify OCO-2 generation produces identical output to before (requires real data)

---

## Phase 16: Experiment Runner (Carbonbench Deduplication)

**Goal**: Eliminate ~3,000 lines of duplication across carbonbench experiments.

**Create**: `neural_transport/neural_transport/experiments/ablation_runner.py`

```python
class AblationRunner:
    def __init__(self, experiment_dir: Path, data_config: DataConfig,
                 model_dirs: list[Path], device: str = "cuda"): ...
    def load_model(self) -> NeuralTransport: ...
    def run_single_eval(self, config: GenerateConfig, name: str) -> EvalResult: ...
    def run_ablation(self, base_config: GenerateConfig,
                     ablation_configs: dict[str, dict],
                     filter_str: str | None = None) -> dict[str, EvalResult]: ...
    def run_distributional_eval(self, config: GenerateConfig) -> EvalResult: ...
    def save_results(self, results: dict[str, EvalResult]): ...
    def plot_results(self, results: dict[str, EvalResult]): ...
    def main_cli(self, base_config: GenerateConfig, ablation_configs: dict): ...
```

Internally uses: `InferenceDataLoader`, `GenerationPipeline`, `EvaluationSuite`, `run_plots()`.

**Rewrite each carbonbench experiment** to ~50 lines:
```python
# Example: 13_flowdps_ablation/run_ablation.py
from neural_transport.configs import GenerateConfig, SamplerConfig
from neural_transport.experiments.ablation_runner import AblationRunner

BASE_CONFIG = GenerateConfig(
    n_samples=100, masking=True, conditioning_mode="velocity_projection",
    sampler=SamplerConfig(sampler="flowdps", sigma_obs=1.0),
)
ABLATIONS = {"unconditional": {"masking": False}, "sigma_0.5": {"sampler.sigma_obs": 0.5}, ...}

if __name__ == "__main__":
    AblationRunner(EXP_DIR, DATA_CONFIG, MODEL_DIRS).main_cli(BASE_CONFIG, ABLATIONS)
```

### Checklist
- [x] **Tests first**: Write `tests/test_ablation_runner.py` (25 tests):
  - `AblationRunner` constructs without error
  - `run_single_eval` with mock model returns `EvalResult`
  - `run_ablation` iterates over configs correctly
  - `save_results` produces valid JSON
  - `main_cli` parses `--filter` correctly, `--dist-eval-only`, `--plot-only`
- [x] Implement `AblationRunner` (`neural_transport/experiments/ablation_runner.py`)
- [x] Export from `neural_transport/experiments/__init__.py`
- [x] Rewrite `12_dps_guidance_ablation/run_ablation.py` as template (~65 lines)
- [x] Rewrite remaining 4 experiment runners (13-16, each ~50-70 lines)
- [x] Rewrite all 5 `plot_ablation.py` files (use `plot_ablation_sweep()` + `plot_summary_bars()`)
- [ ] Verify one experiment produces identical metrics JSON to pre-refactor output (requires GPU)
- Note: Old code replaced in-place (carbonbench is separate repo, originals in git history)

---

## Phase 17: Logging & Type Annotations

**Goal**: Replace print statements with structured logging; add type annotations to public APIs.

**Logging**:
- Replace `print()` calls with `logging.getLogger(__name__)` throughout
- Structured log format: timestamp, module, level, message
- Configurable verbosity via `ExperimentConfig` or CLI flag
- Key locations: `GenerationPipeline`, `AblationRunner`, `EvaluationSuite`, samplers

**Type annotations**:
- Add return types and parameter types to all public functions in refactored modules
- Add `py.typed` marker file
- Run `mypy` on refactored modules (non-strict, just catch obvious errors)

### Checklist
- [x] Replace `print()` with `logging` in: `generation.py`, `ablation_runner.py`, `train.py`, `gen_eval_callback.py` (24 statements across 4 files; `suite.py` and samplers had no prints)
- [x] Configure logging format via `logging.basicConfig()` in entry points: `ablation_runner.main_cli()`, `train_and_eval_singlestep()`, `train_and_eval_rollout()`
- [x] Add type annotations to all public APIs in: `forward_model.py`, `samplers/base.py`, `plots/base.py`, `data/inference_loader.py`, `experiments/ablation_runner.py` (`configs.py` and `evaluation/suite.py` already had excellent coverage)
- [x] Add `py.typed` marker
- [x] Run `mypy` on refactored modules, fix obvious errors (remaining errors are pre-existing `Tensor | None` patterns in `forward_model.py` guarded by runtime checks)
- [x] Run `ruff` on all modified files

---

## Phase 18: E2E Validation — FM Training & Optuna Tuning

**Goal**: Validate the refactored codebase end-to-end by training a flow matching model and tuning hyperparameters.

This is both a validation of the refactor AND a re-establishment of the best unconditional model.

**Training validation**:
- Train FM model for 100 steps using refactored pipeline
- Verify: `gen_eval_callback` uses `EvaluationSuite` + `run_plots()` correctly
- Verify: checkpointing, EMA, LR scheduling all still work
- Compare validation metrics to pre-refactor baseline

**Optuna integration**:
- Create `neural_transport/training/tuning.py` with Optuna objective function
- Search space: lr, weight_decay, batch_size, OT coupling, time sampling distribution, loss weighting
- Objective: distributional metrics (energy distance + MMD) from `EvaluationSuite`
- Pruner: median pruning after 1000 steps
- Storage: SQLite for reproducibility

**Full training run**:
- Train best config to convergence via SLURM
- Evaluate with full `EvaluationSuite` (pointwise + distributional)
- Produce publication plots via `run_plots()`

### Checklist
- [x] **Tests first**: Write `tests/test_training_e2e.py` (`@pytest.mark.slow`):
  - Train 1 epoch → validate → `Loss/Train` and `Loss/Val_singlestep` are finite
  - `GenerationQualityCallback` produces `GenEval/RMSE` and `GenEval/energy_distance`
  - Optuna objective completes a single trial with finite value
- [x] Create `training/tuning.py` with Optuna objective:
  - `MODEL_SIZES`: XXS (~100k), XS (~550k), S (~1.2M), M (~4.8M), L (~10.8M)
  - `suggest_hyperparams()`: lr, weight_decay, warmup_steps, halfcosine_steps, max_lr, model_size, use_ot_coupling, time_sampling (+conditional kwargs), time_loss_weight, gradient_clip_val
  - `FMOptunaObjective`: mirrors `train_singlestep()` with direct trainer access, EMA, GenEval callback, pruning
  - `run_optuna_study()`: TPE sampler, MedianPruner, SQLite storage
  - `get_best_config()`, `export_study_results()`
- [x] Create `training/study_analysis.py` — publication-quality visualization:
  - `analyze_study()`: optimization history, parameter importance (fANOVA), parallel coordinates, contour plots, slice plots, summary table, training curves
- [x] Add optuna optional dependency to `pyproject.toml`
- [x] Update `training/__init__.py` with conditional exports
- [x] Guard `litmodule.plots()` against None logger
- [x] Create experiment scripts — reorganized into `10_fm_unet_tuning/` (baseline) and `11_fm_unet_final/` (Optuna + best model). Old `18_fm_tuning/` deleted.
- [x] All 152 quick tests pass, all 6 slow E2E tests pass, ruff clean
- [x] `suggest_hyperparams()` accepts `model_sizes` param to restrict search space per GPU
- [x] Fixed SQLite race condition in `run_optuna_study()` with retry + random backoff
- [x] Fixed OOM: cached `PreBatchedDataset` dataloaders in `CarbonDataModule.get_dataloader()` and guarded `setup("fit")` against re-creation
- [x] Run Optuna study (352 trials: 208 complete, 77 pruned, 67 failed). Best: `energy_distance=250.1` (trial 310)
- [x] Best config: size=S, norm=group, OT=off, time=uniform, lr=0.00308, wd=0.425, clip=16
- [x] Train best config to convergence (10k steps): `energy_distance=66.3`, `coverage=0.98`, `vendi_score_gen=8.0`
- [x] Baseline (exp 10, 3k steps): `energy_distance=60.4`, `coverage=1.0`, `vendi_score_gen=8.6`
- [x] Comparison: best model and baseline perform similarly; all top-5 Optuna trials chose GroupNorm + no OT, confirming the Phase 20 diagnosis

---

## Phase 19: GenEval Callback — Generation Quality During Training

**Goal**: The Phase 18 Optuna sweep optimized velocity-field loss, but the best model produces mostly NaN/divergent samples at inference time (1/200 valid). The training loss does not predict generation quality. Fix this by making the `GenerationQualityCallback` a reliable training signal, without adding excessive overhead.

**Problem**: Current `GenEval/energy_distance` is computed from 5 tiny samples every N epochs — too noisy and small to catch instability. ODE integration at inference (11+ steps) can diverge even when single-step velocity prediction is accurate.

**Approach**:
- Increase callback sample count to be meaningful (e.g. 20 gen samples) but keep it fast by using fewer ODE steps (3-5 euler steps instead of 11 midpoint)
- Add a **generation stability metric**: fraction of non-NaN/non-exploding samples (`GenEval/valid_fraction`)
- Add `GenEval/gen_std` as a mode-collapse detector (std across generated samples should match GT std)
- Make Optuna objective a weighted combination: penalize trials where `valid_fraction < 1.0` heavily
- Log per-epoch so pruning can catch unstable configs early

### Checklist
- [x] Update `GenerationQualityCallback` to log `GenEval/valid_fraction` and `GenEval/gen_std`
- [x] Increase default `n_gen_samples` in callback (e.g. 20), use fast ODE settings (euler, 5 steps)
- [x] Update `FMOptunaObjective` to use composite objective: `energy_distance + penalty * (1 - valid_fraction)`
- [x] Add tests for new callback metrics (4 new tests + 1 updated, all 6 slow tests pass)
- [x] Re-run Optuna smoke test to verify stability metrics are logged (GPU verified: `valid_fraction=1.0`, `energy_distance`, `RMSE`, `gen_std` all logged)
- [x] `suggest_hyperparams` unchanged — ODE steps are controlled via `generate_kwargs` in callback, not search space

---

## Phase 20: Debug OT Coupling & BatchNorm (combined with Phase 21)

**Goal**: Investigate generation quality issues — the Phase 18 best model (trained 10k steps, 625 epochs) produces mode-collapsed samples: `vendi_score_gen=1.0` (vs GT 5.99), `coverage=0.0`, `density=0.0`, despite `valid_fraction=1.0`.

**Root cause found**: `litmodule.py:119` called `self.model.train()` in `validation_step()`, which recursively set all submodules (including BatchNorm) to training mode. Over 625 epochs, this contaminated BatchNorm running statistics with validation data. At inference (`model.eval()`), the corrupted running stats caused the network to collapse all outputs to a single mode.

**Fix**: Added a `mode` keyword argument to `FlowMatching.forward()` (and threaded through `NeuralTransport.forward()` → `common_step()`). The `validation_step` now passes `mode="train"` instead of calling `model.train()`, keeping BatchNorm layers in eval mode while still dispatching to `training_forward()`.

**Additional**: Added `norm` (batch/group) to Optuna search space in `suggest_hyperparams()`, since GroupNorm has no train/eval mismatch by design.

### Checklist
- [x] Fix `validation_step` BatchNorm corruption — `mode` kwarg in `FlowMatching.forward()` (`flowmatching.py:273`)
- [x] Thread `mode` through `NeuralTransport.forward()` and `common_step()` (`litmodule.py:60-126`)
- [x] Create diagnostic script: `diagnose_mode_collapse.py` — train vs eval mode, velocity field analysis, ODE solver comparison, BatchNorm stats inspection
- [x] Create diagnostic training script: `train_diagnostic.py` — trains with `--norm {batch,group}` and `--no-ot` flags
- [x] Create SLURM array job: `train_diagnostic.slurm` — 3 experiments (batch+OT, group+OT, batch+noOT)
- [x] Add `norm` to Optuna search space: `suggest_hyperparams()` and `_apply_hyperparams()` in `tuning.py`
- [x] Update `train_best.py` to use `best.get("norm", "batch")` instead of hardcoded `norm="batch"`
- [x] Write 7 new tests (mode dispatch, BatchNorm preservation, GroupNorm compat, validation_step) — all pass
- [x] All 152 quick tests pass, ruff clean
- [x] Run `diagnose_mode_collapse.py` on GPU with existing checkpoint — **confirmed BatchNorm divergence**: `model.eval()` produces exploded values (field_mean=61873, std=160792) while `model.train()` produces correct values (mean=0.86, std=4.37). `num_batches_tracked=124` (far too low). Velocity magnitudes 2× higher in eval mode. euler/20 and midpoint/11 produce 0/10 valid samples in eval mode.
- [x] Smoke-tested `train_diagnostic.py` on GPU (100 steps, batch_size=256): BatchNorm+fix → `vendi_score_gen=14.76` (was 1.0!), `energy_distance=460` (was 1051). GroupNorm → `vendi_score_gen=15.48`, `energy_distance=515`. **Fix eliminates mode collapse.** Both variants produce diverse samples.
- [x] Full Optuna sweep (208 completed trials) confirmed: all top-5 trials chose GroupNorm + no OT coupling, validating the diagnosis
- [x] Document findings: root cause = `model.train()` in `validation_step` corrupting BatchNorm running stats; fix = `mode` kwarg API; effect = divergence/explosion in eval mode, not collapse

**Deviations from original plan**: Phases 20 and 21 merged — the BatchNorm bug was identified as the primary cause during investigation. OT coupling confirmed as detrimental by Optuna (all top trials disable it). Diagnostic training scripts replaced by full Optuna sweep which covers the same ground more rigorously. Experiments reorganized: exp 10 = baseline, exp 11 = tuning + best model, exp 18 deleted.

---

## Phase 22: SwinTransformer Model + Optuna Tuning

**Goal**: The UNet architecture may have limitations for flow matching on atmospheric data. SwinTransformer (already implemented in `models/swintransformer.py`) uses attention mechanisms that may better capture long-range spatial dependencies in CO2 fields. Train and tune a SwinTransformer FM model, leveraging lessons from Phases 19-21.

**Approach**:
- Define `MODEL_SIZES` for SwinTransformer (analogous to UNet sizes)
- Use the improved GenEval callback (Phase 19) and fixes from Phases 20-21
- Run Optuna sweep with SwinTransformer-specific search space
- Compare best SwinTransformer vs best UNet on distributional metrics

**Search space additions**:
- `submodel`: categorical `["unet", "swintransformer"]`
- SwinTransformer-specific: `window_size`, `num_heads`, `depths`, `embed_dim`
- Shared: same optimizer/scheduler/FM params as Phase 18

### Checklist
- [x] Define SwinTransformer `SWIN_MODEL_SIZES` (XXS through L) in `tuning.py` — embed_dim (64-384), depths, num_heads
- [x] Update `suggest_hyperparams` to support `submodel` choice — conditional norm (UNet) vs drop_path_rate (Swin)
- [x] Update `_apply_hyperparams` to handle SwinTransformer model_kwargs — sets embed_dim/depths/num_heads/window_size, removes UNet keys
- [x] Update `run_optuna_study()` and `FMOptunaObjective` with `submodel` parameter
- [x] Add tests: `test_suggest_hyperparams_swintransformer`, `test_suggest_hyperparams_unet_has_norm`, `test_apply_hyperparams_swintransformer`, `test_swin_fm_train_10_steps` (slow)
- [x] Create experiment scripts in `12_fm_swin_tuning/`: `run_optuna.py`, `train_best.py`, `analyze_results.py`, SLURM scripts
- [x] All 437 quick tests pass, ruff clean
- [x] Run Optuna sweep (186 trials: 116 complete, 42 pruned, 28 failed). Best: `energy_distance=258.9` (trial 155)
- [x] Best config: size=M, OT=on, time=logit_normal(mean=-0.89,std=1.67), lr=0.00463, wd=0.095, drop_path_rate=0.21, clip=16
- [x] Train best config to convergence (10k steps, 208 epochs): `energy_distance=106.5`, `coverage=0.82`, `vendi_score_gen=9.67`
- [x] Comparison: UNet outperforms Swin on fidelity (`energy_distance` 66.3 vs 106.5, `coverage` 0.98 vs 0.82, `wasserstein_mean` 1.07 vs 1.43); Swin produces slightly more diverse samples (`vendi_score` 9.67 vs 8.01)
- [ ] Publication-quality comparison plots

**Deviations from original plan**: No `submodel` categorical in the Optuna search space — kept as separate studies for cleaner analysis. Swin search converged heavily on size M (104/116 trials), unlike UNet which preferred size S. OT coupling found beneficial for Swin (opposite of UNet). Experiment placed at `12_fm_swin_tuning` (existing 12-16 ablation dirs left in place for now).

---

## Phase 23: Comprehensive Posterior Conditioning Ablation (Exp 13)

**Goal**: Comprehensively tune and compare all posterior sampling methods using Optuna, with satellite orbit mask patterns (OCO-2-like) on synthetic CarbonTracker data. Replaces deprecated experiments 12-16 with a single unified ablation. Produces publication-quality plots proving posterior conditioning with total column CO2 works.

**Note**: Experiments `12_dps_guidance_ablation`, `13_flowdps_ablation`, `14_sde_ablation`, `15_fig_ablation`, `16_ictm_ablation` are now deprecated. Their results informed initial hyperparameter ranges but used synthetic vertical masks. This experiment uses satellite orbit patterns and joint optimization.

**Approach**:
- One Optuna study per method (DPS, FlowDPS, SDE, FIG, ICTM), 50 trials each
- Satellite mask via `create_column_mask(mask_pattern="satellite")` with `obs_fraction=0.3`
- Masking method: `total_column_average_simple` (column-integrated CO2 conditioning)
- **Objective**: pressure-weighted RMSE of ensemble mean vs ground truth, averaged over 10 target samples, with NaN penalty (`objective = mean_pw_rmse + 100 * nan_fraction`)
- After tuning: compare best configs via `AblationRunner` (100 samples, full evaluation)

**New code**:
- `neural_transport/inference/tuning.py` — `PosteriorSamplerObjective`, per-method `suggest_*` functions, `run_posterior_study()`, `compute_pressure_weights()`, `pressure_weighted_rmse()`
- `neural_transport/inference/generation.py` — `generate_multi_target()`: batched GPU generation for multiple targets with pre-allocated zarr, per-target chunk flushing, mixed-target batches
- `neural_transport/experiments/ablation_runner.py` — added `wall_time_sec` timing, fixed lat_weights 3D broadcasting in `run_single_eval()`
- `neural_transport/plots/conditioning_diagnostics.py` — added `plot_obs_match_scatter()`, `plot_spread_at_unobs()`, `plot_per_target_panel()`
- `neural_transport/plots/ensemble_plots.py` — added `plot_conditioning_residual()` (registered, category "conditioning")

**Experiment**: `carbonbench/.../13_posterior_conditioning_ablation/`
- `run_optuna.py` — runs 5 Optuna studies (one per method)
- `compare_methods.py` — multi-target batched generation (20 targets × 10 samples), builds `OSSEResult`, produces diagnostic plots
- `plot_results.py` — loads multi-target zarr, builds `OSSEResult` objects, calls toolkit plotting functions (conditioning_comparison, ensemble_diagnostics, obs_match_scatter, spread_at_unobs, per_target_panel, xco2_maps, zonal_mean, metrics_summary, Optuna analysis)
- `run_optuna.slurm` / `compare_methods.slurm` — SLURM GPU jobs

### Checklist
- [x] Create `neural_transport/inference/tuning.py` with `PosteriorSamplerObjective`, per-method suggest functions, `run_posterior_study()`, `compute_pressure_weights()`, `pressure_weighted_rmse()`
- [x] Add `wall_time_sec` timing to `AblationRunner.run_single_eval()` (3 lines in `ablation_runner.py`)
- [x] Create `tests/test_posterior_tuning.py` — 22 tests (pressure weights, suggest functions, config building, reconstruction)
- [x] All 459 quick tests pass, ruff clean
- [x] Create `13_posterior_conditioning_ablation/run_optuna.py` — runs 5 Optuna studies
- [x] Create `13_posterior_conditioning_ablation/compare_methods.py` — multi-target batched generation using `generate_multi_target()`
- [x] Create `13_posterior_conditioning_ablation/plot_results.py` — diagnostic plots using OSSEResult + toolkit functions
- [x] Create SLURM scripts: `run_optuna.slurm` (48h, tuning), `compare_methods.slurm` (24h, evaluation)
- [x] Create `generate_multi_target()` in `generation.py` — batched GPU generation with pre-allocated zarr, per-target flushing
- [x] Add diagnostic plots to toolkit: `plot_obs_match_scatter`, `plot_spread_at_unobs`, `plot_per_target_panel`, `plot_conditioning_residual`
- [x] Run Optuna studies for all 5 methods: 250/250 trials completed, 0 failures
- [x] Run multi-target comparison: 20 targets × 10 samples × 6 methods on GPU
- [x] Best method DPS: RMSE=2.48 (38.2% improvement over unconditional 4.01)
- [x] All spread-skill ratios in [0.5, 2.0]: DPS=0.81, FlowDPS=0.82, SDE=0.58, FIG=0.85, ICTM=0.83
- [x] Publication-quality comparison plots (17 diagnostic + 25 Optuna)

### Results (20 targets × 10 samples, satellite mask, column XCO2 conditioning)

| Method | RMSE_3D | RMSE_obs | RMSE_away | R² | Spread-Skill |
|--------|---------|----------|-----------|-----|-------------|
| **DPS** | **2.480** | 2.297 | 2.505 | **0.896** | 0.814 |
| SDE | 2.588 | **2.185** | 2.640 | 0.886 | 0.579 |
| ICTM | 2.970 | 2.561 | 3.024 | 0.850 | 0.826 |
| FlowDPS | 3.332 | 2.944 | 3.383 | 0.811 | 0.822 |
| FIG | 3.540 | 3.141 | 3.594 | 0.787 | 0.852 |
| Unconditional | 4.015 | — | 4.015 | 0.726 | 0.920 |

**Key observations**:
- All conditioning methods significantly improve over unconditional (R² from 0.73 to 0.79-0.90)
- DPS has lowest overall RMSE but SDE best matches observations (lowest RMSE_obs)
- SDE spread-skill ratio (0.58) is borderline low → may be overconfident
- Conditioning artifacts visible in spatial patterns (stripes at orbit tracks) — addressed in Phase 23c

---

## Phase 23b: Cluster Storage Clean-Up

**Goal**: The `/Net/Groups/BGI/people/vbenson/` drive is filling up with generated artifacts (zarr predictions, checkpoints, Optuna DBs, plots). Move all generated data to `/Net/Groups/BGI/tscratch/vbenson/` while keeping code on the home drive, in a way that's backwards compatible and easy for future experiments.

**Problem**: Currently, experiment directories like `carbonbench/.../13_posterior_conditioning_ablation/results/` contain both code (`run_optuna.py`, `compare_methods.py`) and large artifacts (`multitarget_predictions.zarr`, `optuna_runs/*.db`, `results/plots/`). The code belongs on the home drive; the artifacts belong on tscratch.

**Approach**: Symlink-based strategy — keep experiment code in place, symlink artifact directories to tscratch.

1. **Artifact convention**: Each experiment dir gets a `DATA_ROOT` pointing to tscratch. Generated outputs (results/, checkpoints/, optuna_runs/) are symlinks to `tscratch/vbenson/carbonbench_artifacts/{experiment_name}/`.
2. **`setup_experiment_storage.sh`** utility script: Takes experiment name, creates tscratch dirs, creates symlinks. Idempotent (safe to re-run).
3. **Update existing experiments**: Run the script for experiments 12-13. Move existing artifacts to tscratch, replace with symlinks.
4. **Template for future experiments**: Add `setup_storage.sh` to experiment template in `carbonbench/`.
5. **`.gitignore`**: Ensure symlink targets and large artifacts are never committed.

**Key constraint**: No changes to Python code paths — scripts still write to `results/`, `checkpoints/`, etc. The symlinks make the storage transparent.

### Checklist
- [x] Create `carbonbench/scripts/setup_experiment_storage.sh` utility — idempotent, supports `--base` for different subdirs
- [x] Define artifact directory convention: `tscratch/vbenson/carbonbench_artifacts/{base}/{exp_name}/{results,singlestep,optuna_runs,...}`
- [x] Migrate existing artifacts for experiments 10-16 + xco2 experiments to tscratch
- [x] Create symlinks in experiment dirs pointing to tscratch
- [x] Update `.gitignore` — added `**/quick_check_results/`, `**/results/`
- [x] Turn off ds_val storing by default — added `save_val_ds=False` param to `plots_val_step()` in `tools/plot.py`
- [ ] Test: existing scripts still work unchanged after migration (requires GPU)
- [x] Document convention in `carbonbench/README.md` ("Experiment Storage Convention" section)

---

## Phase 23c: Investigate & Fix Posterior Conditioning Quality

**Goal**: Current posterior conditioning methods fail at least one of two key requirements:
1. **Perfect obs matching**: Generated samples should near-perfectly reproduce conditioning values at observed locations (zero residual at orbit tracks in spatial RMSE maps)
2. **Distributional plausibility**: Generated samples should be indistinguishable from the GT distribution — no sharp edges, no visible conditioning artifacts (orbit-track stripes in total column CO2)

Phase 23 results show conditioning improves RMSE overall, but spatial patterns reveal artifacts at orbit track boundaries. DPS best Optuna trial already had `spatial_smoothing_sigma=4.62`, confirming the problem goes deeper than smoothing defaults.

**Root cause analysis**: Binary `obs_mask` creates step-function corrections at orbit edges. Smoothing the column error helps but the mask itself is applied via `torch.where(obs_mask, ...)` as a hard boundary. Additionally, the pseudoinverse column projection pushes corrections off the learned data manifold.

**Approach**: Three complementary fixes:
1. **Soft mask** (`obs_weight`): Replace binary mask with smooth float [0,1] weight field via Gaussian blur of the mask boundary. Non-breaking additive API.
2. **NSGA-II multi-objective tuning**: 3 objectives (rmse_away, obs_residual_xco2, roughness) to avoid weight sensitivity. Pareto front selection.
3. **MCG sampler**: Manifold Constrained Gradient — after column projection, snap corrections back onto the data manifold via an additional velocity model pass.

**New code**:
- `neural_transport/configs.py` — added `soft_boundary_sigma` to `SamplerParams`, `mcg` to `KNOWN_SAMPLERS`
- `neural_transport/inference/masking.py` — `create_column_mask(soft_boundary_sigma=...)` computes `obs_weight` via Gaussian blur; `masking_total_column_average_simple` supports float `obs_weight`
- `neural_transport/models/flowmatching.py` — `MaskedVelocityWrapper` unpacks `obs_weight`; all 4 conditioning modes (`forward_guidance`, `forward_repaint`, `forward_velocity_projection`, `forward_correction` via masking) use `obs_weight` for smooth blending when available
- `neural_transport/forward_model.py` — `XCO2ForwardModel.project(obs_weight=...)` uses float weight instead of binary mask
- `neural_transport/inference/samplers/base.py` — `PosteriorSampler` unpacks `obs_weight`, passes to `_project_column`
- `neural_transport/inference/samplers/{fig,sde,ictm}.py` — column error uses `obs_weight` when available
- `neural_transport/inference/samplers/mcg.py` — **new** MCG sampler (Tweedie → project → manifold re-project → blend → re-noise)
- `neural_transport/inference/samplers/__init__.py` — registered `"mcg": MCGSampler`
- `neural_transport/inference/metrics.py` — added `gradient_at_boundary()`, `xco2_obs_residual()`; extended `MetricsResult` with both
- `neural_transport/evaluation/spectral.py` — **new** `power_spectrum_2d()`, `spectral_divergence()`, `spectral_slope()`
- `neural_transport/inference/tuning.py` — added `suggest_mcg_params()`, `soft_boundary_sigma`/`spatial_smoothing_sigma` to all suggest functions, `MultiObjectivePosteriorObjective` (NSGA-II), `select_from_pareto()`, `run_multi_objective_study()`
- `neural_transport/inference/generation.py` — threads `soft_boundary_sigma` through to `create_column_mask` and `generate_multi_target`

**Experiment scripts** in `13_posterior_conditioning_ablation/`:
- `diagnose_stripes.py` — 8-config diagnostic: baseline, no-smoothing, repaint, velocity-projection, late-masking, soft-mask, soft+smooth, MCG
- `run_optuna_v2.py` — NSGA-II multi-objective tuning for DPS, SDE, MCG (100 trials each)
- `compare_methods_v2.py` — final 6-method comparison (old v1 + new v2 + unconditional)
- SLURM scripts for all three

### Checklist
- [x] Bug audit: normalization (correct — `targshift=False` intentional, forward_model compensates), mask broadcasting (correct — [B,1,Nlat,Nlon] broadcasts with [B,C,Nlat,Nlon]), column XCO2 computation (correct — verified by tests)
- [x] Add diagnostic metrics: `gradient_at_boundary`, `xco2_obs_residual` in `metrics.py`; `power_spectrum_2d`, `spectral_divergence`, `spectral_slope` in `evaluation/spectral.py`
- [x] Implement soft mask via `obs_weight` field — non-breaking additive API, threaded through all conditioning modes and samplers
- [x] Implement MCG manifold-constrained sampler — registered in SAMPLER_REGISTRY
- [x] Implement NSGA-II multi-objective tuning — 3 objectives, Pareto selection, `run_multi_objective_study()`
- [x] Add `soft_boundary_sigma` + `spatial_smoothing_sigma` to ALL suggest functions (was missing from SDE, FIG, ICTM)
- [x] Create diagnostic experiment script (`diagnose_stripes.py`) — 8 configs, full metric comparison
- [x] Create NSGA-II experiment script (`run_optuna_v2.py`) — DPS, SDE, MCG
- [x] Create final comparison script (`compare_methods_v2.py`) — old vs new methods
- [x] All 489 quick tests pass, ruff clean
- [x] Run `diagnose_stripes.py` on GPU — 8 configs compared (Job 6004421, 1min)
- [x] Run `run_optuna_v2.py` on GPU — DPS 100 trials (4 Pareto), SDE 100 trials (7 Pareto), MCG 100 trials (all failed → fixed config bug, re-ran 100 trials, 78 Pareto)
- [x] Run `compare_methods_v2.py` on GPU — 6 methods × 20 targets × 10 samples
- [x] Fixed NaN bug: `0 * NaN` when `obs_weight` used with NaN obs_values → only set `obs_weight` when `soft_boundary_sigma > 0`, `nan_to_num` on obs_values
- [x] Fixed MCG config bug: `manifold_alpha`, `n_manifold_steps` missing from `SamplerParams` dataclass
- [ ] Achieve: `xco2_obs_residual < 0.5 ppm` — best is SDE 1.15 (not met, see Phase 23d/e for improvements)
- [ ] `gradient_ratio < 1.5` — DPS v1 achieves 1.00 ✓, MCG needs fix (see Phase 23d)
- [ ] Document findings in `CONDITIONING_NOTES.md`

### Phase 23c Results (20 targets × 10 samples)

| Method | RMSE_3D | RMSE_obs | RMSE_away | XCO2_res | Grad_ratio | Roughness |
|--------|---------|----------|-----------|----------|------------|-----------|
| unconditional | 4.49 | — | 4.49 | — | — | 0.51 |
| **dps_v1** | **3.07** | 2.63 | **3.13** | **1.32** | **1.00** | **0.48** |
| **sde_v1** | 3.13 | **2.45** | 3.21 | **1.15** | 1.10 | 0.69 |
| dps_v2 (NSGA-II) | 3.47 | 2.97 | 3.53 | 1.61 | 1.06 | 0.49 |
| mcg_v2 (NSGA-II) | 23.78 | 4.79 | 25.36 | 3.23 | 3.37 | 9.91 |
| soft_mask_best | 18.39 | 16.28 | 18.68 | 13.70 | 0.86 | 0.90 |

**Observations**: DPS v1 remains best overall. MCG v2 failed — re-noising step destroys conditioning signal (see Phase 23d analysis). Soft mask eliminates boundary artifacts (gradient_ratio=0.86) but needs tuning. NSGA-II DPS didn't improve over original Optuna. SDE has best obs matching.

### Diagnostic Experiment Results (5 targets × 10 samples, DPS baseline varied)

| Config | RMSE_3D | Grad_ratio | Key finding |
|--------|---------|------------|-------------|
| a_baseline (DPS+smooth) | 3.29 | 1.00 | Best overall |
| b_no_smoothing | exploded | 35M | Smoothing critical for DPS guidance |
| c_repaint | 4.84 | 2.01 | Higher boundary gradient |
| d_velocity_proj | 4.84 | 2.01 | Identical to repaint |
| e_late_masking | 4.05 | 1.02 | Near-ideal gradient ratio |
| f_soft_mask | 17.86 | 0.85 | Lowest gradient but high RMSE (untuned) |
| g_soft+smooth | 64.15 | 3.72 | Too aggressive |
| h_mcg | 4.35 | 1.02 | MCG works with simple config, near-ideal gradient |

---

## Phase 23d: Fix MCG Sampler + Proper Tuning

**Goal**: The MCG sampler (Phase 23c) achieved RMSE 23.8 under NSGA-II tuning — far worse than DPS (3.07). The root cause is identified: the re-noising step in the manifold projection destroys the conditioning signal injected by column projection. Fix the implementation, verify against the original MCG paper (Chung et al., NeurIPS 2022), and properly tune.

**Root cause analysis**: In `_manifold_project()`, after projecting `x_hat_proj` to satisfy column constraints, the state is re-noised to `t_next` and passed through the velocity model. This re-noising adds random noise that overwrites the careful column projection. With 10-20 ODE steps, this accumulates into total loss of conditioning.

**Fix approach**: Replace re-noising manifold projection with **PCFM-style OT interpolant** (Utkarsh et al., 2025, arXiv 2506.04171). Instead of re-noising + velocity re-evaluation:
1. Forward-shoot to predict clean sample: `x_hat_1 = x_t + (1-t) * v_theta(x_t, t)` (Tweedie)
2. Project: `x_proj = x_hat_1 - H^T (H H^T)^{-1} (H x_hat_1 - y)` (exact for linear H)
3. OT interpolant back: `x_t_next = (1 - t_next) * z + t_next * x_proj` (same as re-noise but with projected estimate)

The key insight: step 3 is identical to the FlowDPS re-noise step but uses the projected estimate. The difference from our current MCG is that we skip the expensive and destructive "manifold projection via re-velocity" step. Instead, the manifold consistency comes from the velocity model at the next step naturally producing on-manifold velocities.

**Additional improvement**: Add an optional "forward shooting" step where instead of Tweedie (one-step estimate), we integrate the ODE from current t to t=1 using a few Euler steps. This gives a better clean estimate for the projection. Cost: a few extra velocity evaluations.

### Checklist
- [x] **Bug audit**: Compare MCG implementation against original Chung et al. algorithm step-by-step — root cause confirmed: `_manifold_project()` re-noising destroys conditioning signal
- [x] **Fix**: Replace `_manifold_project()` with OT-interpolant approach (no re-noising + re-velocity) — rewrote `mcg.py` using forward-shoot → project → OT-interpolant-back pattern
- [x] **Add forward shooting option**: multi-step clean estimate via Euler integration to t=1 — `n_forward_steps` param (1=Tweedie, >1=multi-step Euler)
- [x] **Bug fix**: Added MCG to `_SAMPLER_KWARGS_MAP` in `flowmatching.py` (was missing — MCG could not be dispatched via `inference_forward()`)
- [x] All 490 tests pass (161 quick + 329 others), ruff clean
- [x] **Tune with single-objective Optuna first**: pw-RMSE, 50 trials (Job 6042986) — 50/50 complete, 0 failures. **Best trial #9: RMSE = 2.11** (params: sigma_obs=0.18, spatial_smoothing_sigma=4.61, soft_boundary_sigma=0.27, fresh_noise=True, n_forward_steps=2)
- [x] **Then NSGA-II**: 100 trials with fixed MCG (Job 6043032) — 100/100 complete, 6 Pareto trials. Best Pareto: rmse_away=3.54, obs_residual=0.19, roughness=0.26
- [x] **Compare**: fixed MCG vs DPS v1 vs SDE v1 on 20 targets × 10 samples (Job 6043638) — all 5 methods ran successfully
- [x] **Target**: MCG RMSE < 3.5 — **achieved: 2.11** (single-obj), **3.54** (NSGA-II Pareto). MCG now **outperforms DPS** (was 3.07) with single-objective tuning

---

## Phase 23e: Literature-Driven Posterior Sampling Methods

**Goal**: Investigate recent (2024-2025) training-free posterior sampling methods from the literature and implement the most promising ones for column XCO2 conditioning. The current DPS approach works well (RMSE 3.07) but has room for improvement in obs matching and artifact reduction.

**Key papers** (all training-free):

1. **PCFM** (Utkarsh et al., 2025, arXiv 2506.04171) — Physics-Constrained Flow Matching. Projects predicted clean sample onto constraint manifold via Gauss-Newton, maps back via OT interpolant. For linear H, projection is exact in one step. No backprop through model. Already partially incorporated in Phase 23d MCG fix.

2. **FMPS** (arXiv 2411.07625) — Flow Matching Posterior Sampling. Adds correction term to velocity field: `dx = [v_theta + r * Delta(x,c)] dt`. Two variants: gradient-aware (Tweedie + backprop) and gradient-free. Steers velocity directly rather than projecting state.

3. **DiffStateGrad** (arXiv 2410.03463, ICLR 2025) — Projects measurement gradient onto low-rank SVD subspace of current state. Specifically designed to prevent artifacts. Could replace our spatial smoothing with a more principled subspace projection.

4. **FGPS** (arXiv 2411.15295) — Frequency-Guided Posterior Sampling. Time-varying low-pass filtering in frequency domain. Addresses high-frequency artifacts from posterior sampling. Orthogonal to other methods — can be combined as a post-processing step.

5. **OC-Flow** (arXiv 2410.18070, ICLR 2025) — Optimal Control for Flow Matching. Frames guided generation as optimal control with KL regularization. Provides theoretical convergence guarantees. More expensive (requires ODE adjoints).

**Implementation plan** (prioritized by expected impact and simplicity):

### Step 1: PCFM Sampler
Implement full PCFM algorithm as `samplers/pcfm.py`:
- Forward shoot: Euler steps from t to 1 to predict clean sample
- Linear projection: `x_proj = x_1 - H^T (HH^T)^{-1} (Hx_1 - y)` (reuse `forward_model.project()`)
- OT interpolant back: `x_t' = (1-t') * u_0 + t' * x_proj`
- Optional lambda-penalty step for soft constraints (probably not needed for linear H)

### Step 2: FMPS Sampler
Implement FMPS as `samplers/fmps.py`:
- Gradient-aware variant: Tweedie estimate + likelihood gradient added to velocity
- Key difference from DPS: includes a scaling factor `beta_t` derived from the flow's noise schedule
- Key difference from our guidance mode: proper normalization of the gradient step

### Step 3: DiffStateGrad Enhancement
Add SVD-based gradient projection as an option in the sampler base class:
- At each step, compute SVD of the current batch of states
- Project the column correction gradient onto top-k singular vectors
- Replace `spatial_smoothing_sigma` with subspace projection (more principled artifact reduction)

### Step 4: Ablation experiment
Compare on 20 targets × 10 samples:
- DPS v1 (current best)
- Fixed MCG (Phase 23d)
- PCFM
- FMPS (gradient-aware)
- DPS + DiffStateGrad (SVD projection instead of Gaussian smoothing)
- Best method + FGPS frequency filtering

### Checklist
- [ ] Implement PCFM sampler (`samplers/pcfm.py`), register, add suggest function
- [ ] Implement FMPS sampler (`samplers/fmps.py`), register, add suggest function
- [ ] Implement DiffStateGrad as `_project_to_svd_subspace()` in sampler base class
- [ ] Implement FGPS post-processing in `evaluation/spectral.py` or `inference/filtering.py`
- [ ] Write tests for all new samplers (shape, no NaN, interface compliance)
- [ ] Tune each method (single-objective Optuna, 50 trials)
- [ ] Run ablation experiment (6+ methods × 20 targets × 10 samples)
- [ ] Identify best method or combination
- [ ] Document findings and update Phase 23 results

---

## Phase 24: E2E Validation — Real OCO-2 Inversion

**Goal**: Run actual inversion using real OCO-2 data via `OCO2DataLoader` + `GenerationPipeline`. Validate the complete real-data pipeline.

**Setup**:
- Use `OCO2DataLoader` to load real OCO-2 XCO2, averaging kernels, a priori profiles
- Run best 2-3 methods from Phase 23
- Time-series generation over test period

**Evaluation**:
- Qualitative: global posterior maps, uncertainty maps
- Quantitative (where possible): comparison to CarbonTracker analysis, TCCON stations
- `EvaluationSuite` for available validation targets

**Plots** via `run_plots()`:
- "always" + "conditioning" + "ensemble" categories
- Global XCO2 posterior maps
- Uncertainty (ensemble spread) maps
- Time series at TCCON stations (if available)

### Checklist
- [ ] Load real OCO-2 data via `OCO2DataLoader`
- [ ] Run 2-3 methods in time-series mode via `GenerationPipeline`
- [ ] Evaluate against CarbonTracker analysis
- [ ] Generate publication-quality plots
- [ ] Document results and observations

---

## Phase 25: Conjugate Integrators — Few-Step Conditioning

*Ref: arXiv 2405.17673*

- [ ] **Tests first**: Add to `test_samplers.py` parametrized tests for `ConjugateIntegratorSampler`
- [ ] `ConjugateIntegratorSampler` in `samplers/conjugate.py`
- [ ] Register in `SAMPLER_REGISTRY`
- [ ] Toy OSSE gate
- [ ] Efficiency benchmark (wall-time vs RMSE Pareto front)

**Deliverable**: 5-step conditional generation. Pareto front across all methods.

---

## Phase 26: Advanced FM Training (W-CFM, OAT-FM)

- [ ] Weighted CFM: Gibbs kernel weighting in `training_forward()`
- [ ] Time-dependent loss weighting: `w(t) = 1/sigma(t)^2` or SNR-based
- [ ] OAT-FM fine-tuning (optional): minimize acceleration
- [ ] Retrain + re-evaluate all posterior methods via `AblationRunner`

**Deliverable**: Training curves. Improved unconditional RMSE. Re-evaluation via `EvaluationSuite`.

---

## Phase 27: Conditional Flow Matching (Retraining)

*Ref: Lipman et al. 2023*

- [ ] +2 UNet input channels: obs_channel + mask_channel
- [ ] Training: synthetic XCO2 from target, random masks, noise on obs
- [ ] 50k conditional + 50k mixed steps, 80/20 conditional/unconditional
- [ ] Classifier-free guidance: `v_guided = (1+w) * v(x,t|y) - w * v(x,t|empty)`
- [ ] Evaluate via `EvaluationSuite`

**Success**: RMSE < 3.0 ppm on column OSSE.

---

## Phase 28: Grand Comparison

- [ ] Aggregate `EvalResult` from all methods
- [ ] Statistical significance tests (paired t-test on per-timestep metrics)
- [ ] Publication plots via `run_plots()` with all categories
- [ ] Experiment: `carbonbench/.../grand_comparison/`

---

## Phase 29: Real OCO-2 Full Application

- [ ] Apply all top methods to real satellite data via `GenerationPipeline` + `OCO2DataLoader`
- [ ] Validate vs CarbonTracker posterior, TCCON, ObsPack surface flasks
- [ ] Full publication-quality evaluation

---

## Phase 30: Multi-step Temporal Conditioning

- [ ] Sequential/autoregressive generation
- [ ] Temporal consistency metrics (autocorrelation, mass conservation)
- [ ] Sliding-window 4D-Var style (stretch goal)

---

## Phase 31: Advanced Ideas (Brainstorm)

- [ ] Physics-informed guidance via torchtransport
- [ ] Latent-space FM with encoder/decoder
- [ ] Score distillation / consistency models
- [ ] Ensemble Kalman Flow
- [ ] Amortized posterior network
- [ ] Multi-resolution conditioning
- [ ] SWAG UQ for velocity model

---

## Phase 32: Publication (ACP/GMD)

1. Introduction: CO2 inverse modeling, generative approaches
2. Background: flow matching, OCO-2, CarbonTracker
3. Method: FM for CO2, XCO2 forward model, posterior sampling methods
4. OSSE: benchmark, method comparison
5. Real Data: OCO-2 posterior, TCCON/ObsPack validation
6. Discussion: methods, physics, cost, vs 4D-Var
7. Conclusion

---

## Dependency Graph

```
Phase 0 (Update implementation.md)
  │
Phase 1 (Test Infrastructure — TDD foundation)
  │
Phase 2 (Configs)
  ├── Phase 3 (Forward Model)
  │     ├── Phase 4 (Sampler ABC)
  │     │     └── Phase 5 (Sampler Registry & Migration)
  │     │           └── Phase 9 (MaskedVelocityWrapper Slim-Down)
  │     └── Phase 10 (Metric Consolidation)
  │           └── Phase 11 (EvaluationSuite)
  │                 └── Phase 12 (Plot Base & Always-On)
  │                       └── Phase 13 (Specialized Plots)
  ├── Phase 6 (Masking Extraction)
  │     └── Phase 7 (Noise & Spatial)
  │           └── Phase 8 (Unified Generation Pipeline)
  ├── Phase 14 (Inference Data Loading)
  │     └── Phase 15 (OCO-2 Data Loading)
  └── Phase 16 (Experiment Runner) [depends on 8 + 11 + 13 + 14]

Phase 17 (Logging & Types) — after all refactor phases

Phase 18 (E2E: Training & Tuning)
  │
  ├── Phase 19 (GenEval Callback)                        ┐
  │     └── Phase 20 (Debug OT Coupling + BatchNorm fix) ├── Fix generation quality
  │           └── Phase 22 (SwinTransformer + Tune)      ┘
  │
Phase 23 (Posterior Conditioning Ablation)  ← Optuna tuning + multi-target comparison
  ├── Phase 23b (Cluster Storage Clean-Up) ← Move artifacts to tscratch, symlinks ✓
  ├── Phase 23c (Fix Conditioning Quality) ← Soft mask, MCG sampler, NSGA-II tuning ✓ (DPS v1 best, MCG failed)
  ├── Phase 23d (Fix MCG Sampler)          ← OT-interpolant fix, forward shooting, proper tuning
  ├── Phase 23e (Literature Methods)       ← PCFM, FMPS, DiffStateGrad, FGPS
  └── Phase 24 (Real OCO-2 Inversion)     ← Apply best method to real data

Phases 25-32: Development & Publication
```

## Lines of Code Impact (Estimated)

| Module | Before | After | Change |
|--------|--------|-------|--------|
| `flowmatching.py` (MaskedVelocityWrapper) | 770 | ~350 | -55% |
| `generative.py` | 1031 | 0 → `generation.py` + `masking.py` + `noise.py` (~500) | -52% |
| `posterior_samplers.py` | 702 | 0 → `samplers/*.py` (~800, DRYer) | +14% (better organized) |
| metrics + distributional + analyse | ~820 | ~700 (in `evaluation/`) | -15% |
| `plot_results.py` duplicate metrics | ~40 | 0 (import from evaluation) | -100% |
| 5× `run_ablation.py` (carbonbench) | ~1800 | ~250 (5×50) | **-86%** |
| 5× `plot_ablation.py` (carbonbench) | ~1350 | ~150 (5×30) | **-89%** |
| New: configs, forward_model, runner, data loaders, spatial | 0 | ~700 | new |
| Tests | ~360 | ~1200 | +233% |

## Verification Strategy

After **every** phase:
1. `pytest tests/ -m "not slow and not gpu"` — all quick tests pass
2. `pytest tests/test_toy_column_osse.py -m quick` — toy OSSE gate passes
3. `ruff check neural_transport/` — no linting errors

Phase-specific gates:
- After Phase 5: sampler registry creates all samplers correctly
- After Phase 8: `GenerationPipeline.run()` matches old `iterative_generate()` output
- After Phase 9: `MaskedVelocityWrapper` is ~100 lines
- After Phase 13: `run_plots()` produces all expected plot files
- After Phase 16: one carbonbench experiment produces identical metrics JSON
- After Phase 18: trained model matches pre-refactor quality
- After Phase 23: best method pw-RMSE < 75% of unconditional, spread-skill in [0.5, 2.0], 5 Optuna studies complete ✓ (DPS: 38% improvement)
- After Phase 23b: existing scripts work unchanged with symlinked storage, artifacts on tscratch
- After Phase 23c: `xco2_obs_residual < 0.5 ppm`, `gradient_ratio < 1.5`, no visible orbit-track stripes in column XCO2

## Key Files Reference

**Core (neural_transport)** — files to be refactored:
- `neural_transport/models/flowmatching.py` — FlowMatching + MaskedVelocityWrapper (770 lines)
- `neural_transport/inference/posterior_samplers.py` — 4 sampler classes (702 lines)
- `neural_transport/inference/generative.py` — 2 generation functions (1031 lines)
- `neural_transport/inference/metrics.py` — evaluation metrics
- `neural_transport/inference/distributional_metrics.py` — distributional metrics
- `neural_transport/inference/analyse.py` — scoring dataframes (489 lines)
- `neural_transport/plots/plot_results.py` — transport + gen plots (1721 lines)
- `neural_transport/plots/conditioning_diagnostics.py` — conditioning plots (373 lines)
- `neural_transport/plots/distributional_plots.py` — distribution plots (708 lines)
- `neural_transport/plots/one_to_rule_them_all.py` — existing unified interface (162 lines)
- `neural_transport/datamodule.py` — CarbonDataset, CarbonDataModule
- `neural_transport/litmodule.py` — NeuralTransport Lightning module
- `neural_transport/training/train.py` — training entry points (860 lines)
- `neural_transport/training/gen_eval_callback.py` — generative eval callback (133 lines)
- `neural_transport/experiments/toy_column_osse.py` — toy OSSE testbed
- `tests/` — 4 test files (~360 lines total)

**Experiments (carbonbench)** — files to be deduplicated:
- `carbonbench/.../12_dps_guidance_ablation/{run,plot}_ablation.py`
- `carbonbench/.../13_flowdps_ablation/{run,plot}_ablation.py`
- `carbonbench/.../14_sde_ablation/{run,plot}_ablation.py`
- `carbonbench/.../15_fig_ablation/{run,plot}_ablation.py`
- `carbonbench/.../16_ictm_ablation/{run,plot}_ablation.py`

**implementation.md**: `/Net/Groups/BGI/people/vbenson/CarbonBench/dryrun/neural_transport/implementation.md`
