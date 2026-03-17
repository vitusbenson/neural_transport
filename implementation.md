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
    def __init__(self, model, data_loader, config: GenerateConfig): ...

    def run(self, out_dir: Path) -> xr.Dataset:
        """Single entry point — dispatches based on config."""

    def run_distributional(self, out_dir, n_gt, n_gen, batch_size) -> tuple[xr.Dataset, xr.Dataset]:
        """Replaces generate_for_distributional_eval()."""

    # Internal methods (shared across modes):
    def _prepare_batch(self, idx, device): ...
    def _apply_masking(self, batch, masking_config): ...
    def _run_inference(self, batch, noise): ...
    def _postprocess_and_save(self, preds, out_dir): ...

    # Mode-specific batch collection:
    def _collect_sample_batch(self, ...): ...       # per-sample (OSSE)
    def _collect_timeseries_batch(self, ...): ...   # windowed (OCO-2)
```

**Delete**: `generative.py` (all content now in `generation.py`, `masking.py`, `noise.py`)

### Checklist
- [ ] **Tests first**: Write `tests/test_generation_pipeline.py`:
  - `GenerationPipeline.run()` with synthetic data produces correct output shape
  - Sample mode: output has `sample` dimension
  - Time-series mode: output has `time` dimension
  - `run_distributional()` returns GT and gen datasets with correct sample counts
  - Config with `masking=False` produces unconditional samples
- [ ] Implement `GenerationPipeline` with shared + mode-specific methods
- [ ] Integrate distributional eval as `run_distributional()`
- [ ] Delete `generative.py`, update all imports across both repos
- [ ] Verify: unconditional generation produces identical output to before
- [ ] Verify: OSSE conditioning produces identical output
- [ ] All existing tests still pass

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
- [ ] **Tests first**: Verify all 17 `test_forward_model.py` tests still define expected behavior
- [ ] Remove extracted methods from `MaskedVelocityWrapper`
- [ ] Refactor `__init__` to receive `XCO2ForwardModel`
- [ ] Update `FlowMatching` to construct and pass `XCO2ForwardModel`
- [ ] Remove `dt=0.1` hardcoded fallback (now in `GenerateConfig`)
- [ ] All 17 `test_forward_model.py` tests still pass
- [ ] Toy OSSE still passes end-to-end
- [ ] `MaskedVelocityWrapper` is now ~100 lines

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
- [ ] **Tests first**: Write `tests/test_evaluation_metrics.py`:
  - Each metric function: known input → known output (e.g. RMSE of zeros = 0)
  - Metric symmetry/invariance properties where applicable
  - Latitude-weighted RMSE differs from unweighted
  - CRPS with perfect ensemble = 0
  - Energy distance of identical distributions = 0
- [ ] Implement `pointwise.py` consolidating all metric definitions
- [ ] Implement `ensemble.py` migrating from `metrics.py`
- [ ] Implement `distributional.py` migrating from `distributional_metrics.py`
- [ ] Delete duplicate metric definitions from `plot_results.py` (lines 66-107), import from `evaluation.pointwise`
- [ ] Extract `max_n=200` energy distance cutoff into `EvalConfig`
- [ ] Update `analyse.py` to import from `evaluation.*` instead of defining inline
- [ ] All existing metric tests still pass (`test_distributional_metrics.py`)

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
- [ ] **Tests first**: Write `tests/test_evaluation_suite.py`:
  - `evaluate_deterministic` returns `EvalResult` with populated `pointwise`, `None` ensemble
  - `evaluate_ensemble` returns `EvalResult` with both `pointwise` and `ensemble`
  - `evaluate_distributional` returns `EvalResult` with `distributional` populated
  - `to_dataframe` produces correct columns
  - `to_json` roundtrips with `from_json`
- [ ] Implement `EvaluationSuite`
- [ ] Implement `EvalResult` with serialization
- [ ] Migrate `analyse.py` functions to be thin wrappers
- [ ] Refactor `gen_eval_callback.py` to use `EvaluationSuite`
- [ ] Verify transport model validation still produces correct metrics
- [ ] All existing tests still pass

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
- [ ] **Tests first**: Write `tests/test_plots.py`:
  - `PlotContext.savefig` creates files in expected formats
  - `register_plot` adds to `PLOT_REGISTRY`
  - `run_plots(categories=["always"])` calls exactly the "always" plots
  - `run_plots(categories=None)` infers categories from `EvalResult` content
  - Plot functions don't crash with minimal synthetic data
- [ ] Implement `PlotContext` with publication rcParams (font sizes, IPCC colormaps, consistent figure sizing)
- [ ] Implement `register_plot` decorator and `run_plots` dispatcher
- [ ] Implement `field_plots.py` with always-on plots
- [ ] Extract hardcoded target pressures `[1013, 843, 441, 73]` into `PlotConfig.target_pressures`
- [ ] Verify always-on plots render correctly with sample data

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
- [ ] **Tests first**: Extend `tests/test_plots.py`:
  - Each refactored plot function accepts `EvalResult` without error
  - Category auto-inference selects correct categories for different `EvalResult` contents
  - Ablation plots render with synthetic sweep data
- [ ] Refactor `conditioning_diagnostics.py` to use `PlotContext` + `EvalResult`
- [ ] Refactor `distributional_plots.py` similarly
- [ ] Create `metrics_plots.py` (extracted from carbonbench `plot_ablation.py`)
- [ ] Create `ensemble_plots.py`
- [ ] Refactor transport plots in `plot_results.py` to use framework
- [ ] Extract `animate_predictions()` to `animation.py`
- [ ] Implement smart category inference in `run_plots()`
- [ ] Verify all plots visually on one ablation result + one transport result

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
- [ ] **Tests first**: Write `tests/test_data_loading.py`:
  - `InferenceDataLoader` constructs from `DataConfig`
  - `get_batch` returns tensors with correct shapes
  - `grid_info` matches expected dimensions
  - `cos_lat_weights` matches manually computed values
- [ ] Implement `InferenceDataLoader` and `GridInfo`
- [ ] Update `GenerationPipeline` (Phase 8) to use `InferenceDataLoader`
- [ ] Verify generation produces identical output

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
- [ ] **Tests first**: Write `tests/test_oco2_loader.py`:
  - `ObservationBatch` has correct tensor shapes
  - `align_time` computes correct offset for known time arrays
  - `aggregate_window` unions sparse observations correctly
  - Window of size 1 returns single-timestep observations
- [ ] Implement `ObservationBatch` dataclass
- [ ] Implement `OCO2DataLoader` with time alignment and window aggregation
- [ ] Update `GenerationPipeline` time-series mode to use `OCO2DataLoader`
- [ ] Verify OCO-2 generation produces identical output to before

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
- [ ] **Tests first**: Write `tests/test_ablation_runner.py`:
  - `AblationRunner` constructs without error
  - `run_single_eval` with mock model returns `EvalResult`
  - `run_ablation` iterates over configs correctly
  - `save_results` produces valid JSON
  - `main_cli` parses `--ablation` filter correctly
- [ ] Implement `AblationRunner`
- [ ] Rewrite `12_dps_guidance_ablation/run_ablation.py` as template
- [ ] Rewrite remaining 4 experiment runners
- [ ] Rewrite all 5 `plot_ablation.py` files (use `run_plots()` with "ablation" category)
- [ ] Verify one experiment produces identical metrics JSON to pre-refactor output
- [ ] Delete old duplicated code from all 5 experiments

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
- [ ] Replace `print()` with `logging` in: `generation.py`, `ablation_runner.py`, `suite.py`, all samplers
- [ ] Configure logging format in `ExperimentConfig`
- [ ] Add type annotations to all public APIs in: `configs.py`, `forward_model.py`, `samplers/base.py`, `evaluation/suite.py`, `plots/base.py`, `data/inference_loader.py`, `experiments/ablation_runner.py`
- [ ] Add `py.typed` marker
- [ ] Run `mypy` on refactored modules, fix obvious errors
- [ ] Run `ruff` on all modified files

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
- [ ] **Tests first**: Write `tests/test_training_e2e.py` (`@pytest.mark.slow`):
  - Train 10 steps → validate → metrics are finite
  - `gen_eval_callback` produces plots in expected directory
- [ ] Run 100-step training, verify identical metrics to pre-refactor
- [ ] Create `training/tuning.py` with Optuna objective
- [ ] Run Optuna study (50 trials, SLURM)
- [ ] Train best config to convergence
- [ ] Evaluate: distributional metrics + all plots
- [ ] Compare to pre-refactor model quality

---

## Phase 19: E2E Validation — OSSE with Real OCO-2 Mask

**Goal**: Run OSSE experiments using the real OCO-2 observation mask pattern (sparse, irregular, orbit tracks) applied to synthetic CarbonTracker data. Compare all posterior sampling methods.

This validates the full pipeline: data loading → masking → generation → evaluation → plotting.

**OSSE setup**:
- Ground truth: CarbonTracker test data
- Observations: synthetic XCO2 computed from CarbonTracker, masked with real OCO-2 coverage pattern
- Methods to compare: unconditional, DPS guidance, FlowDPS, SDE, FIG, ICTM (best config from each ablation)

**Method comparison**:
- Use `AblationRunner` with one config per method
- `EvaluationSuite.evaluate_ensemble()` for each method
- Metrics: RMSE_3D, RMSE_XCO2, CRPS, spread-skill ratio, calibration
- Spatial metrics: RMSE at observed vs unobserved locations

**Plots** via `run_plots()`:
- "always" + "conditioning" + "ensemble" categories
- Side-by-side method comparison grid
- Ablation summary bars
- Pareto front: wall-time vs RMSE

### Checklist
- [ ] Create experiment config with real OCO-2 mask on synthetic CT data
- [ ] Run all 6 methods (unconditional + 5 posterior) via `AblationRunner`
- [ ] Compute ensemble metrics for each method
- [ ] Generate comparison plots
- [ ] Verify: best method achieves RMSE_xco2_obs < 75% of unconditional
- [ ] Verify: spread-skill ratios in [0.5, 2.0]
- [ ] Document results in experiment directory

---

## Phase 20: E2E Validation — Real OCO-2 Inversion

**Goal**: Run actual inversion using real OCO-2 data via `OCO2DataLoader` + `GenerationPipeline`. Validate the complete real-data pipeline.

**Setup**:
- Use `OCO2DataLoader` to load real OCO-2 XCO2, averaging kernels, a priori profiles
- Run best 2-3 methods from Phase 19
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

## Phase 21: Conjugate Integrators — Few-Step Conditioning

*Ref: arXiv 2405.17673*

- [ ] **Tests first**: Add to `test_samplers.py` parametrized tests for `ConjugateIntegratorSampler`
- [ ] `ConjugateIntegratorSampler` in `samplers/conjugate.py`
- [ ] Register in `SAMPLER_REGISTRY`
- [ ] Toy OSSE gate
- [ ] Efficiency benchmark (wall-time vs RMSE Pareto front)

**Deliverable**: 5-step conditional generation. Pareto front across all methods.

---

## Phase 22: Advanced FM Training (W-CFM, OAT-FM)

- [ ] Weighted CFM: Gibbs kernel weighting in `training_forward()`
- [ ] Time-dependent loss weighting: `w(t) = 1/sigma(t)^2` or SNR-based
- [ ] OAT-FM fine-tuning (optional): minimize acceleration
- [ ] Retrain + re-evaluate all posterior methods via `AblationRunner`

**Deliverable**: Training curves. Improved unconditional RMSE. Re-evaluation via `EvaluationSuite`.

---

## Phase 23: Conditional Flow Matching (Retraining)

*Ref: Lipman et al. 2023*

- [ ] +2 UNet input channels: obs_channel + mask_channel
- [ ] Training: synthetic XCO2 from target, random masks, noise on obs
- [ ] 50k conditional + 50k mixed steps, 80/20 conditional/unconditional
- [ ] Classifier-free guidance: `v_guided = (1+w) * v(x,t|y) - w * v(x,t|empty)`
- [ ] Evaluate via `EvaluationSuite`

**Success**: RMSE < 3.0 ppm on column OSSE.

---

## Phase 24: Grand Comparison

- [ ] Aggregate `EvalResult` from all methods
- [ ] Statistical significance tests (paired t-test on per-timestep metrics)
- [ ] Publication plots via `run_plots()` with all categories
- [ ] Experiment: `carbonbench/.../grand_comparison/`

---

## Phase 25: Real OCO-2 Full Application

- [ ] Apply all top methods to real satellite data via `GenerationPipeline` + `OCO2DataLoader`
- [ ] Validate vs CarbonTracker posterior, TCCON, ObsPack surface flasks
- [ ] Full publication-quality evaluation

---

## Phase 26: Multi-step Temporal Conditioning

- [ ] Sequential/autoregressive generation
- [ ] Temporal consistency metrics (autocorrelation, mass conservation)
- [ ] Sliding-window 4D-Var style (stretch goal)

---

## Phase 27: Advanced Ideas (Brainstorm)

- [ ] Physics-informed guidance via torchtransport
- [ ] Latent-space FM with encoder/decoder
- [ ] Score distillation / consistency models
- [ ] Ensemble Kalman Flow
- [ ] Amortized posterior network
- [ ] Multi-resolution conditioning
- [ ] SWAG UQ for velocity model

---

## Phase 28: Publication (ACP/GMD)

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

Phase 18 (E2E: Training & Tuning)     ┐
Phase 19 (E2E: OSSE + OCO-2 Mask)     ├── Validate refactored codebase
Phase 20 (E2E: Real OCO-2 Inversion)  ┘

Phases 21-28: Development & Publication
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
- After Phase 19: OSSE RMSE_xco2_obs < 75% of unconditional, spread-skill in [0.5, 2.0]

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
