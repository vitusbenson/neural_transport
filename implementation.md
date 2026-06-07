# Flow-Matching Data Assimilation of Satellite XCO₂ — Implementation Plan

> **Scope (decided 2026-06-01).** Track A: present flow-matching (FM) probabilistic data
> assimilation as an *independent ML method* for assimilating satellite column CO₂. Get the
> observation operator right, train a leak-free transport model, motivate the method in an
> OSSE that mimics the OCO-2 MIP setup, then run a real **OCO-2 MIP v11-style** assimilation that
> estimates the **full 3-D concentration field** (we do **not** estimate surface fluxes). Validate
> the assimilated CO₂ against **held-out OCO-2, OCO-3, TCCON, and in-situ** observations, and against
> the **co-sampled outputs of submitted MIP participant models**, and ask: *are our errors comparable
> in magnitude to the OCO-2 MIP participants?* If they are, we claim an independent, similarly-accurate
> method. If worse, we quantify by how much and diagnose why. (No manuscript phase here — code + results only.)
>
> Full development history (refactor Phases 1–17, method development 18–25p, all documented negative
> results) lives in **`implementation_archive.md`**. This file is the forward-looking plan only.

---

## Current state (entry point for new sessions)

- **Repos**: `neural_transport` (library) + `carbonbench` (experiments). Branch `feature/xco2-l1`,
  committed & pushed through Phase 25p as of 2026-06-01.
- **OSSE SOTA**: **2.513 ppm XCO₂ RMSE** (n=100), CRPS 1.030, spread/err 0.32 — Phase 2p
  **EnKF (loc=4)** on the residual-FM model (deterministic backbone `f_det` rollout-fine-tuned with
  EMA shadow weights + a flow-matching residual head). Free (no-DA) ≈ 3.24 ppm. Cross-seed confirmed.
  *Concentration-field reconstruction vs a held-out CarbonTracker run, synthetic obs masks.*
- **Model recipe (SOTA)**: `f_det` = single-step train → rollout-FT (K=4, uniform step weights,
  `lr_mult=0.1`, EMA decay 0.999) → EMA-frozen ckpt; residual FM head on top (`sigma_res`,
  noise_scale 1.05). See archive Phase 25p.
- **Library is modular**: `configs`, `forward_model`, `inference/samplers/*`, `inference/masking`,
  `inference/noise`, `inference/generation`, `evaluation/*`, `plots/*`, `data/{inference_loader,
  oco2_loader}`, `datasets/{carbontracker,mip_oco2}`.

### OCO-2 MIP v11 — raw inputs (https://gml.noaa.gov/ccgg/OCO2_v11mip/download.php)
- **OCO-2**: `OCO2_b11.2_10sec_GOOD_r2.nc4` (10-sec averages). *Already staged.*
- **OCO-3**: `OCO3_b11_10sec_GOOD_r2.nc4` (10-sec averages).
- **In-situ / ObsPack**: `obspack_co2_1_OCO2MIP_v5.0.1_2024-11-14`.
- **TCCON**: `tccon_timeaverages_R20250609.tgz` (Caltech-hosted). *Already staged.*
- **Co-sampled participant model outputs** (preliminary): available from the MIP "Preliminary
  Output" / release page — participant submissions co-located to the obs. **We use these as direct
  benchmarks** (compare our co-sampled CO₂ to theirs at the same obs), even though preliminary.
- Note: staged data is already **b11.2 / v11-aligned** (`OCO2_b11.2...`, `tccon...R20250609`).

### Real-data assets already staged on tscratch
`/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/`
- `OCO2MIP_OCO2/oco2_assimilate.zarr` (raw soundings: XCO₂, AK, a priori, sigma levels, psurf),
  `OCO2MIP_OCO2/OCO2_b11.2_10sec_GOOD_r2.nc4`, `.../val/mip_oco2_latlon5.625_l10_6h.zarr` (regridded).
- `OCO2MIP_OCO3/`, `OCO2MIP_TCCON/tccon_timeaverages_R20250609`.
- `Carbontracker/{CT2022_flux, CT2022_molefrac}` — reference fields.
- Prep code: `neural_transport/datasets/mip_oco2.py` (`regrid_mip_oco2`, `vertical_aggregation_oco2`).
- **Action in P5**: re-pull/refresh all raw v11 inputs (OCO-2, OCO-3, ObsPack v5.0.1, TCCON, co-sampled
  participant runs) from the v11 download page and rebuild the products from scratch.

### Validation targets (researched 2026-06-01)
OCO-2 v10 MIP aircraft evaluation ([Jacobson et al., ACP 2025](https://acp.copernicus.org/articles/25/1725/2025/)):
flux-attributable posterior-CO₂-vs-aircraft RMSE **0.88–1.91 ppm** per region (55–85 % of total →
total mismatch ≈ 1–3.5 ppm); MIP ensemble spread under-estimates true error by 1.3–1.9× (calibration
is an open community problem — a place FM's flexible posterior can contribute). For v11 the **direct
benchmark is the co-sampled participant outputs** at our held-out obs. **Our bar**: independent-obs
RMSE in the same low-single-digit-ppm ballpark, well-calibrated spread.

---

## Roadmap overview

| Phase | Title | Output |
|---|---|---|
| **P1** | Averaging-kernel forward operator — correctness fix | Verified `XCO2ForwardModel` |
| **P2** | Leak-free model training (≤2013 train, 2014 val) | Retrained residual-FM |
| **P3** | OCO2MIP-style OSSE (realistic synthetic obs) | MIP-like OSSE results |
| **P4** | OSSE skill & shortcoming analysis | Skill/calibration/failure-mode figures |
| **P5** | OCO-2/OCO-3/TCCON/in-situ data prep (MIP v11) | Clean obs datasets + splits + benchmarks |
| **P6** | Real OCO2MIP-style 3-D concentration assimilation | Assimilated CO₂ fields |
| **P7** | Independent validation harness (held-out obs) | MIP-style skill scores |
| **P8** | Comparison to OCO-2 MIP + calibration analysis | Final verdict + analysis |

**Working conventions** (from archive): TDD where practical; after each phase run
`pytest tests/ -m "not slow and not gpu"`, the toy OSSE gate, and `ruff check`. Long runs via SLURM;
artifacts on tscratch with symlinks (`scripts/setup_experiment_storage.sh`).

---

## P1 — Averaging-kernel forward operator (correctness fix)

**Goal**: Implement the MIP-prescribed XCO₂ forward operator — **interpolate the model CO₂ profile to
the retrieval's native pressure levels, then apply the averaging formula there** — and prove it correct.
This comes first because *everything downstream (OSSE realism, assimilation, validation) depends on a
correct observation operator.*

**The gap**: MIP docs prescribe interpolate-then-apply at the retrieval's 20 native levels:
`XCO2 = xco2_prior + Σ_k h_k·a_k·(x_k − x_apriori_k)`. We currently do the **opposite** —
`vertical_aggregation_oco2` (`datasets/mip_oco2.py`) collapses the 20-level AK/`h`/a-priori *down* to
the model's coarse `l10` grid and applies the formula there (`forward_model.py:128`). These agree only
if the profile is linear within each aggregation group. (Confirmed by reading the code 2026-06-01.)

### Tasks
- [x] **Tests first** (`tests/test_forward_model.py`): on a sample of real soundings, compared
      (a) current down-aggregated op vs (b) interpolate-then-apply op; quantified the discrepancy.
      **Reference check** passes: `H(x_apriori) == xco2_apriori` (and `Σ h_k·xa_k ≈ xco2_apriori`
      to 0.003 ppm — the 2-decimal rounding of stored `xco2_apriori`).
- [x] Added `XCO2ForwardModel.from_retrieval_levels(p_model, p_ret[20], h[20], a[20], x_apriori[20],
      xco2_prior)`: linear-in-log-pressure interpolation `x_model→p_ret` (`build_interp_matrix`,
      clamped/constant extrapolation, rows sum to 1), then the averaging formula at 20 levels.
- [x] **Differentiable + adjoint**: key insight — interp `x_interp = W·x_model` is linear, so the
      whole operator is affine: `H = column_offset + gᵀx_model` with effective model-grid kernel
      `g = Wᵀ(h·a)` and `column_offset = xco2_prior − (h·a)ᵀx_apriori`. So `jacobian_transpose`
      (`= g·err`) and `project` work **unchanged** (Wᵀ is baked into `g`). Verified `g·err` matches
      autograd of `forward`. Shared helper `forward_model.effective_column_kernel()` is the single
      source of truth (samplers + future EnKF both consume it).
- [x] Wiring **mechanism** delivered: `XCO2ForwardModel.from_masking_config` dispatches to the
      corrected operator when `forward_operator="interp"` + native fields (`p_model`, `p_ret`, 20-level
      `ak/h/x_apriori`, `xco2_prior`) are present; falls back to `"aggregate"` (legacy l10) otherwise —
      this **is** the ablation flag. `ObservationBatch` now carries `pressure_levels` (p_ret) and the
      loader extracts it. **Full DA activation deferred** to P3/P6 (see note) because it needs the
      native l20 obs product with `C_ret=20 ≠ C_model=10`, which doesn't exist yet — the EnKF inline
      operator (`generation.py:2148`) and `_build_obs` (`generation.py:1639`) currently assume
      `C_ret==C_model`. The shared helper makes that switch-over a localized change.
- [x] **Storage decided**: keep **native 20-level** product. Verified `regrid_mip_oco2(...,
      vertical_levels="l20")` retains `xco2_averaging_kernel`, `co2_profile_apriori`, `pressure_weight`,
      `xco2_apriori`, and per-gridcell `pressure_levels[20]` (the l20 vertical aggregation is an
      identity relabel — no code change to `mip_oco2.py` needed). P5 rebuilds the multi-year product
      at l20.

**Result (discrepancy note)**: on 3000 real soundings with realistic structured model profiles
(boundary-layer enhancement + free-trop curvature), down-aggregated-l10 vs interpolate-then-apply
disagree by **bias ≈ 1.4 ppm, RMSE ≈ 1.7 ppm, max ≈ 5.3 ppm** — comparable to the entire 2.5 ppm SOTA
signal. A large part is the pressure-grid mismatch the legacy path ignores (CarbonTracker-l10 model
values index-aligned to OCO2-l10 aggregated kernels at *different* pressures). Confirms P1 matters.

**Next step (P3/P6)**: activate the corrected operator in the DA loop once the l20 obs product is
built — decouple `C_ret` (20, obs) from `C_model` (10, state) in `_build_obs`/EnKF and swap the inline
`h·a` for `effective_column_kernel(...)` (g + column_offset; the offset cancels in EnKF anomalies and
only enters the innovation). For the OSSE (P3), generate synthetic obs *with* the corrected operator.

**Key files**: `forward_model.py` (operator + `build_interp_matrix` + `effective_column_kernel`),
`datasets/mip_oco2.py` (l20 path verified), `data/oco2_loader.py` (`pressure_levels`),
`tests/test_forward_model.py` (+13 tests). **Deliverable**: verified forward operator + discrepancy
note ✅. **Gate**: a-priori reproduction test passes ✅; interpolate-then-apply is the default operator
(`forward_operator="interp"`) wherever native-level data is available ✅.

---

## P2 — Leak-free model training (≤2013 train, 2014 val)

**Goal**: Retrain the residual-FM model with the SOTA recipe but on data **only through 2013, with
2014 as validation**, so the entire OCO-2 MIP period (2015–2020) is held out — no leakage for the
independent-method claim.

### Tasks
- [x] **Leakage confirmed**: the SOTA CarbonTracker split was train **2000–2016** / val **2017** /
      test 2018–2020 — i.e. it trained on 2015–2016 and validated on 2017, *inside* the OCO-2 MIP
      period (2015–2020). The "independent method" claim required re-splitting.
- [x] **Re-prepared CT datasets (leak-free)**: re-sliced the existing full regrid zarr (no raw
      regridding) into **train 2000–2013 (n=20455)**, **val 2014 (n=1460)**, **test 2015–2020
      (n=8768)**. Norm stats recomputed on the **≤2013 train only** (no normalization leakage).
      New root `…/data/Carbontracker_leakfree`. Script: `prepare_leakfree_data.py`.
- [x] **Retrained `f_det`** (SLURM chain, leak-free data): single-step backbone 20k
      (val 0.340 ≈ SOTA 0.337) → rollout-FT **uniform K=4, lr_mult=0.1, EMA(0.999, start=200), 16k**
      → `freeze_ema_ckpt.py` → `ema_frozen.ckpt`.
- [x] **Recomputed `sigma_res`** on the leak-free EMA backbone; retrained residual-FM head
      (noise_scale 1.05, 20k steps, bs=128).
- [x] **Sanity gate PASSED** (EnKF loc=4, σ=0.1, noise 1.05, n_inits=20×n_samples=10, 120 steps):

      | window | free (no DA) | EnKF loc=4 | spread/err |
      |---|---|---|---|
      | val 2014 | 1.96 | **1.82** | 0.29 |
      | test 2015–2020 (held-out MIP) | 2.02 | **1.87** | 0.24 |
      | *SOTA ref (test 2018–2020)* | *3.24 (n=100)* | *2.54 (n=20)* | *0.27* |

      Leak-free EnKF skill (1.82–1.87 ppm) is comfortably in the SOTA ~2.5 ppm regime — **no
      regression** from removing 2014–2020. Caveats (honest): absolute OSSE numbers are
      **window-dependent** and not strictly comparable across different test years/`n`; the lower
      values vs 2.54 are not claimed as a real improvement. The OSSE scores fidelity to CarbonTracker
      *dynamics* (targshift removes the absolute CO₂-growth offset), which is why a ≤2013 model
      generalises to 2015–2020 in the OSSE. DA gain is modest here (~8%) because the free baseline is
      already accurate on these windows; the EnKF still improves over free at every lead.

**Key files**: `carbonbench/.../25c_v4_residual_fm/{prepare_leakfree_data.py, freeze_ema_ckpt.py,
run_leakfree_pipeline.sh, phase1_det_backbone_leakfree, phase1p_det_stable_rollout_ft_leakfree,
phase2p_residual_fm_stable_leakfree}`, `25_transport_prior_osse/eval_trajectory_v2.py` (`--model-dir`).
**Deliverable**: leak-free checkpoints + OSSE-parity note ✅. **Gate**: no 2015–2020 in train/val ✅;
OSSE skill in the SOTA regime ✅.

> **Forward note for P3/P4**: the leak-free free-run error grows with distance from the training
> period (val-2014 free 1.96 → the harder 2018–2020 sub-window of test will be larger). A strict
> 2018–2020-only restriction (matching the SOTA window exactly) is a clean apples-to-apples check to
> fold into the P4 OSSE analysis. The model + checkpoints are ready for P3 (MIP-style OSSE).

---

## P3 — OCO2MIP-style OSSE (realistic synthetic obs)

**Goal**: Build an OSSE that *resembles the OCO-2 MIP setup* — synthetic XCO₂ observations sampled
from a known CarbonTracker truth using **real OCO-2 orbit tracks, the corrected AK operator (P1),
realistic sparsity, and retrieval noise** — and run FM-DA on it.

**Why**: Bridges the idealised OSSE and the real experiment. It lets us measure skill against a
*known truth* under realistic observing conditions, and characterise the synthetic→real
"performance drop" the README flags, before touching real obs.

**Data note (answers "do we need P5 data?")**: **No.** The OSSE needs only the OCO-2
*observation geometry* (where/when soundings land + their real AKs), which is already staged:
`OCO2MIP_OCO2/train/mip_oco2_latlon5.625_l20_6h.zarr` is the native-l20 product (the one P1 was
waiting on), on the model grid, covering **2014-09 → 2020-12** (the whole MIP period). XCO₂ *values*
are synthesised from the CarbonTracker truth, so real retrievals aren't needed here. P5's full v11
re-download + assimilate/validate freeze is only for the **real** assimilation (P6+).

### Tasks
- [x] **P3.1 — orbit-obs provider** (`neural_transport/inference/orbit_obs.py`,
      `OrbitObsProvider`): reads the staged MIP-l20 product and serves, keyed by timestamp, the real
      OCO-2 obs mask + 20-level AK/pressure_levels/pressure_weight on the model grid (NaN-free, exact
      time alignment to the CarbonTracker truth; realised coverage ≈ **1.5–2.4 %** of cells per 6 h).
      +12 tests (provider gated on staged data; synthesis math via a CI-safe mock provider).
- [x] **P3.2 — corrected operator in the EnKF** (activated the P1-deferred C_ret=20 ≠ C_model=10
      path): `_build_orbit_enkf_obs` synthesises obs from the known truth via
      `effective_column_kernel` (the real 20-level AK + log-p interpolation baked into the effective
      model-grid kernel `g`). In a perfect-model OSSE `column_offset` cancels in the EnKF innovation,
      so only `g` is needed. `generate_trajectory_enkf` gains `orbit_obs`/`obs_noise`/`ak_mode`/
      `thin_fraction` (pure-EnKF). Adjoint/synthesis verified by tests.
- [x] **P3.3 — runner + smoke** (`carbonbench/.../27_mip_style_osse/eval_mip_osse.py`): leak-free
      (P2) residual-FM EnKF on real-orbit synthetic obs. **Smoke (n_inits=4, n_samples=10, 80 steps,
      real ~1.6 % coverage)**: EnKF beats free with the gain **growing monotonically with lead**
      — +0.3 % @ lead 4 → +1.9 % @ lead 40 → **+3.0 % @ lead 79** (free 1.907 → EnKF 1.851 ppm;
      summary RMSE 1.728 → 1.701, CRPS 0.780 → 0.747). Sparse real-orbit DA progressively corrects
      free-run drift, exactly as expected.
- [x] **P3.4 — realism sweep** (full stats **n_inits=20 × n_samples=10 = 200 trajectories, 120
      steps** over the leak-free test 2015–2020; runs fanned out across local GPUs, ≤24 threads):

      | run | config | summary RMSE | CRPS | spread/err |
      |---|---|---|---|---|
      | `none_full` | free (no DA) | 2.025 | 0.921 | 0.27 |
      | `enkf_full` | real AK, full real coverage | 1.991 | 0.879 | 0.26 |
      | `sweep_akuniform` | flat AK (AK ablation) | 1.989 | 0.878 | 0.26 |
      | `sweep_noise03` | retrieval noise σ=0.3, sigma_obs=0.3 | **1.985** | 0.888 | 0.26 |
      | `sweep_thin50` | half the soundings | 2.018 | 0.902 | 0.26 |
      | `sweep_inflation` | prior-inflation 1.08 | 2.054 | **0.863** | **0.41** |

**Findings**:
1. **FM-DA beats free in every realistic config** (summary RMSE 1.985–2.018 vs 2.025; CRPS clearly
   better) — under **~1.6–2.4 % OCO-2 coverage per 6 h, ~15–20× sparser than the 30 % idealised
   OSSE**. This quantifies the synthetic→realistic "performance drop" the README flags: the gain
   shrinks from idealised ~21 % (3.24→2.54 SOTA) to a few-% lead-averaged signal here.
2. **Gain grows with lead then reverses** (`enkf_full` vs free): +1.1 % @ lead 4 → +3.5 % @ 20 →
   +3.7 % @ 80, but **−9 % @ lead 119**. Sparse DA corrects free drift at short/medium lead, but the
   **small (n=10) ensemble's spurious covariance injects error at the longest leads** (>~25 days,
   many chained cycles) — a real, documented EnKF failure mode.
3. **AK shape barely matters in-OSSE** (1.991 ≈ 1.989): obs and forward use the *same* operator, so
   the kernel shape largely cancels in relative skill. P1's operator correctness matters for matching
   the **real** retrieval (P6), not for OSSE skill ranking.
4. **Sparsity** (thin 50 %) shrinks the gain but still beats free (2.018 < 2.025).
5. **Obs-error inflation is the effective long-lead RMSE stabiliser**: `sweep_noise03` (sigma_obs 0.3)
   gives the best summary RMSE (1.985) and the best long lead (L100 2.169 vs free 2.201; L119 2.647 vs
   `enkf_full` 2.726) — higher assumed obs-error damps over-fitting to spurious covariance.
6. **Prior inflation 1.08 trades RMSE for calibration** — best CRPS (0.863) and best spread/err
   (0.26→0.41, much less under-dispersed), but **worse RMSE, especially long lead** (L119 2.726→3.265):
   widening the ensemble amplifies the spurious-covariance error. *Not* the long-lead fix.

**Headline**: under realistic OCO-2 sparsity, FM-DA still improves the 3-D field over free
(lead-averaged RMSE −2 %, CRPS −5 %), with **gains concentrated at short/medium lead** and a
**long-lead degradation** traced to small-ensemble spurious covariance. Obs-error inflation, not
ensemble inflation, stabilises long lead; larger ensembles + stronger localization are the P4 levers.

### P3.5 — Why is the global gain only a few %? (diagnostic, `analyze_da_gain.py`)

A few-% *global-field* gain looked suspicious, so we decomposed it. **The DA is in fact very
effective — the small headline is a metric-dilution artifact**, from three stacked effects:

| RMSE locus (EnKF, all leads) | free → EnKF | gain |
|---|---|---|
| 3-D field, **global** | 2.043 → 2.012 | +1.5 % ← headline |
| 3-D field, **at observed cells** | 1.918 → 1.462 | **+23.8 %** |
| XCO2 column, **global** | 0.837 → 0.746 | +10.8 % |
| XCO2 column, **at observed cells** | 0.890 → 0.449 | **+49.6 %** |
| 3-D field, at *un*observed cells | 2.009 → 1.971 | +1.9 % |

1. **Spatial dilution (dominant)**: OCO-2 sees **~2 %/step, 16.7 % ever** (30 d). DA **halves the
   column error where it observes** (−49.6 %); averaging that over the ~98 % unconstrained domain
   yields the +1.5 % global number. *Info barely reaches unobserved cells (+1.9 %)* — the real limit.
2. **Vertical dilution**: XCO2 is a column integral; the 3-D RMSE (~2 ppm) is dominated by
   vertical-structure error the column obs can't see (global *column* RMSE is only ~0.84 ppm).
3. **Identical-twin**: truth = CarbonTracker and the model is *trained on* CarbonTracker → free XCO2
   error is only 0.15 ppm @ lead 0 → 1.36 @ lead 119; the forecast is near-perfect, so there is
   structurally little *global* error to correct. (Real-data P6 will have a far larger model-reality
   gap where DA matters more.)

**Implication for the MIP comparison (P7/P8)**: the MIP validates *at obs locations*, so the relevant
skill is the **−24 % 3-D / −50 % column at observed cells**, not the diluted global field.

### P3.6 — FMPS vs EnKF on the realistic OSSE (the user's question)

Wired FMPS/D-Flow to the orbit-obs path (`generate_trajectory_ensemble_batched` `orbit_obs`) and ran
the flow-matching posterior sampler with the corrected operator. **FMPS underperforms both free and
EnKF here** (matched 8×80; full 20×120 agrees):

| method | RMSE | CRPS | spread/err |
|---|---|---|---|
| free | 1.802 | 0.805 | 0.24 |
| **EnKF (loc=4)** | **1.770** | **0.760** | 0.24 |
| FMPS (smooth=0.5) | 1.907 | 0.900 | 0.30 |
| FMPS (smooth=0.0) | 1.930 | 0.902 | 0.30 |

- Not a tuning artifact: with **no spatial smoothing** (correction applied *only* at obs cells), FMPS
  is **still 53.7 % worse than free *at the observed cells*** (XCO2 0.613 → 0.942). It degrades from
  lead 0 onward.
- **Mechanism**: FMPS *replaces* the near-perfect deterministic forecast with a **generative sample**
  (residual-FM head, noise_scale 1.05) nudged toward obs. At ~2 % coverage the generative variance
  dominates the weak obs constraint, pushing even observed cells away from truth. The EnKF instead
  *keeps* the deterministic ensemble and applies a **targeted linear Kalman update** → halves obs-cell
  error. FMPS's only edge is higher spread (0.30 vs 0.24, better dispersion) — at a large RMSE/CRPS cost.
- **Caveat**: FMPS was designed/tuned for the *dense* idealised regime (SOTA-competitive there). The
  realistic-sparse + identical-twin setting is adversarial for a generative sampler; FMPS may regain
  value with **denser obs** or a **larger forecast-model error (real data, P6)**. **For the OSSE,
  EnKF is the method of choice.**

### P3.7 — Making DA "translate to the global field": diagnosis + single-pass posterior samplers

**Motivating concern (correct)**: DA halves obs-cell error but the global gain is small. Does the
signal propagate to the rest of the field via transport/mass-balance?

**Propagation diagnostic** (`diag_propagation.py`, never-observed-cell gain vs lead): the signal
**does** propagate — never-observed XCO2 gain grows **+2.7 % @1d → +11.7 % @18d** — then **collapses
at long lead** (−16.5 % @30d) under the small (n=10) ensemble's spurious covariance.
- **Larger ensemble (n=40)** cures the collapse (clean, monotone, +12 % stable through 30d), better
  spread/err (0.32), RMSE 1.860 vs 1.882 (n=10). Propagation strengthens modestly but stays bounded
  (~12 % unobserved vs ~40 % observed) — the per-cell filter has no flow-dependent horizontal
  covariance, so it can only spread via isotropic localization + transport over cycles.

**Single-pass posterior-sampler study** (Phase-1 of the "make the generative model work" plan; web
research synthesised in this session — DPS/ΠGDM/TMPD, FlowDPS, Score-based DA Rozet & Louppe 2023):
- We already have a proper sampler suite (`FMPSSampler`, `FlowDPSSampler`=closed-form PGDM column
  projection `forward_model.project`, DPS-`sde`, `mcg`, `pcfm`); FMPS already does Tweedie-at-`x̂₀` +
  decaying guidance schedule (so it is *not* the naive "gradient-at-noisy-xₜ" bug).
- **Robust negative result**: BOTH generative per-step samplers underperform free *and* EnKF on the
  realistic orbit OSSE (8×120): free 1.954, EnKF 1.882, EnKF-n40 1.860; **FMPS ≈1.93, FlowDPS ≈2.12
  (σ_obs-insensitive, spread/err 0.50)** — worse even *at observed cells* (FlowDPS XCO2 0.88→1.22).
- **Obs application is CORRECT (not the bug)** — `diag_obs_fit.py` isolation test (one conditioned
  step, no rollout): the conditioned ensemble **mean fits the obs 35 % better than free** at obs
  cells (FlowDPS 0.67→0.43, FMPS 0.67→0.43). So obs+prior *do* constrain the posterior mean in a
  single step; the failure is **not** a normalization/operator misapplication.
- **The failure is autoregressive process-variance accumulation**: the conditioned samplers run a
  higher-variance sampling process (spread/err **0.50 vs free's 0.26**) and feed their own
  spread-0.50 output back in for 120 steps → the nonlinearly-evolved ensemble mean drifts. Each step's
  conditioning helps locally, but the accumulated process variance corrupts the long-run mean.
  - **Not fixable cheaply**: `fresh_noise=False` (deterministic re-noise) changes nothing (variance
    is from per-member `x_init`, not the renoise); **lowering `noise_scale` is catastrophic** (RMSE
    7.7–9.2, diverges) — `noise_scale≈1.05` is the residual-FM model's **stability** point, not a
    dispersion dial. The conditioned-sampler variance is intrinsic.
- **Mechanism summary**: near-perfect forecast + sparse linear-Gaussian obs ⇒ ~linear-Gaussian
  problem; the EnKF (minimal linear increment on a *stable* forecast) is near-optimal, while any
  per-step *generative* filter re-samples the field with intrinsic process variance that accumulates.
  Generative flexibility expected to pay only with **denser obs**, **larger forecast error (real
  data, P6)**, or **multimodal posteriors**.

**Strategic implication**: the per-step generative *filter* is intrinsically variance-limited here —
confirmed, not cheaply fixable. The route to a generative win is the **window/4-D approach** (joint
constraint over K steps controls effective variance + adds upwind propagation).

**Window/4-D PoC — window-D-Flow on orbit obs (`generate_trajectory_window_dflow` + `orbit_obs`)**:
the 4-D approach **rescues the generative method to EnKF parity, better-calibrated** (8×120):

| method | RMSE | CRPS | spread/err | obs-cell XCO2 | never-obs gain |
|---|---|---|---|---|---|
| free | 1.954 | 0.807 | 0.25 | 0.88 | — |
| EnKF | 1.882 | 0.794 | 0.26 | ~0.62 | +11 % @18d |
| FMPS / FlowDPS (per-step) | 1.93 / 2.12 | 0.90 / 1.03 | 0.50 | 0.94 / 1.22 (worse!) | — |
| **window-D-Flow w=4** | **1.877** | **0.762** | 0.29 | **0.62 (+30 %)** | +13 % @18d |
| window-D-Flow w=8 | 1.912 | **0.730** | 0.36 | — | **+20 % @12d** (collapses @30d) |

- The window smoother **improves observed cells** (+30 %, vs per-step generative which *degraded*
  them) and **propagates to unobserved cells ≥ EnKF** (w=8 reaches +20 % @12d vs EnKF ~8 %).
- Matches EnKF RMSE with **better CRPS** (0.762/0.730 vs 0.794). Confirms the generative model *can*
  do good DA — with the 4-D framing, not greedy per-step filtering.
- **Caveat**: w=8 collapses at the very last lead (last-window/boundary instability); w=4 is stable.
  And window-D-Flow uses an **expensive inner Adam loop** — motivating the single-pass SDA version.

### P3.8 — ROOT CAUSE of per-step generative failure: residual-vs-state forward operator

Investigating the residual-FM internals revealed *why* the per-step samplers (FMPS/FlowDPS) fail —
a concrete, fixable bug, **not** a fundamental limit:

- The residual-FM generates a **residual** `r` (unit-variance); the state is `x = f_det(x_prev) +
  σ_res·r`. But `prepare_masking_config` passes **no `f_det` prediction and no `σ_res`** to the
  sampler's `XCO2ForwardModel`. So the per-step samplers condition the **residual `r` as if it were
  the full state** — `H` is applied to `r`, comparing `g·r` to obs `g·x_state`.
- Consequence: the obs guidance forces the *small* residual to absorb the **entire** observed
  signal (most of which `f_det` already explains) → the residual is **over-inflated** → spread/err
  0.50 and the rollout degradation. (Single-step obs-fit still improved because inflating `r` does
  reduce the obs error — at the cost of huge variance that accumulates over the AR rollout.)
- **Window-D-Flow works precisely because it applies `H` to the full state** (`x_next_phys` from the
  AR chain), not the residual — which is why it reached EnKF parity while per-step lost.

**The fix (DONE — `XCO2ForwardModel` state-aware, ~60 lines + 4 unit tests)**: the operator gains
`det_pred_phys` + `residual_scale`; for the residual-FM the "physical x" used in
`forward`/`project`/`jacobian_transpose`/likelihood-gradient is `x_phys = det_pred + σ_res·r`, and the
projection/guidance kernel folds in the chain-rule gain `h_ak·σ_res/target_std` (`effective_kernel`;
obs and target share normalization so the std factors cancel in state mode). `ResidualFlowMatching`
injects `det_pred_phys`/`σ_res` into `masking_config`; all sampler callers
(`base`/`ictm`/`fig`/`MaskedVelocityWrapper`) use `effective_kernel`. Full suite 154 green.

**Result — the fix is confirmed, and the full method comparison (8 inits × 120 leads, orbit OSSE):**

| method | RMSE | spread | CRPS | SER | note |
|---|---|---|---|---|---|
| free (no DA) | 1.9537 | 0.489 | 0.8070 | 0.251 | baseline |
| EnKF | 1.8823 | 0.489 | 0.7937 | 0.260 | best linear filter |
| EnKS lag24 | 1.8794 | 0.486 | 0.7974 | 0.259 | linear smoother ≈ filter |
| **window-D-Flow w4** | **1.8766** | — | 0.7616 | 0.294 | **best RMSE** (generative 4-D smoother) |
| **window-D-Flow w8** | 1.9117 | — | **0.7297** | 0.355 | **best CRPS** (−8 % vs EnKF) |
| FMPS — fixed, σ=0.02 | 1.9158 | 0.459 | 0.7795 | 0.240 | best *single-pass*; CRPS < EnKF |
| FMPS — fixed, σ=0.05 | 1.9402 | 0.465 | 0.7949 | 0.240 | helpful (was harmful) |
| FMPS — **buggy** | 2.1179 | **1.088** | 1.0255 | **0.497** | residual over-inflated |
| FlowDPS — fixed | 2.0572 | 0.993 | 0.9947 | 0.473 | over-disperses via fresh-noise renoise |
| SDA-Langevin (stable) | 2.0349 | 0.452 | 0.8634 | 0.220 | stable but weak optimiser (< D-Flow) |

The state-aware operator eliminates the residual over-inflation exactly as diagnosed: FMPS spread
1.088→0.465, SER 0.497→0.240, RMSE 2.118→1.916 (from *worse than free* to *better than free*). FlowDPS
still over-disperses — not the operator but its **fresh-noise renoise** (`--no-fresh-noise` gave an
identical curve, confirming the renoise per se isn't the driver; FlowDPS's project→renoise dynamics
amplify member spread regardless). FMPS is the better per-step sampler; σ=0.02 is its sweet spot
(σ=0.01 starts over-fitting, SER 0.240→0.291).

**The honest final story.** (1) On **RMSE** all DA methods cluster tightly (1.877–1.954): the
"few-percent" gain is **real and fundamental** to this regime (identical-twin + ~2 % coverage +
column-vs-3-D dilution), not an algorithm bug — the one genuine bug (per-step generative being
*harmful*) is now removed. (2) On **CRPS (probabilistic skill) the generative methods clearly win**:
window-D-Flow w8 0.730, w4 0.762, FMPS-σ0.02 0.780 all beat EnKF 0.794 and free 0.807 — the FM
posterior is better-calibrated. (3) **Linear vs generative smoother**: the *linear* ensemble smoother
(EnKS) ≈ filter (the near-perfect forecast leaves little for a linear past-state correction), but the
*generative* 4-D smoother (window-D-Flow) edges out EnKF on RMSE (1.877) and clearly on CRPS — the
nonlinear 4-D fit captures structure the linear update cannot. **Recommended product**: window-D-Flow
(best RMSE + CRPS) when compute allows; **FMPS σ=0.02** as the cheap single-pass option (best-in-class
CRPS, ~filter cost).

**SDA build (DONE; negative-but-useful result)**: `generate_trajectory_window_sda` — single-pass
SDA-style smoother: an annealed **Langevin** sweep in the FM latent `z` (prior score `−z`, FM latent is
N(0,I)) with **DPS** window-obs guidance, as `inner_sampler="sda_langevin"` in the tested window-D-Flow
scaffolding (shared chain rollout + `_window_misfit`). Aggressive settings (`temp=1, eps=0.2`) diverge
over long rollouts (latent over-shrinks ~0.9⁸× + noise accumulates → RMSE 3.03); a conservative config
(`temp=0, eps=0.05, 6 steps`) is stable but only reaches **RMSE 2.035** — a *weaker optimiser* than
D-Flow's Adam at equal NFEs, so it does not beat the baselines here. Wired as `--method window_sda`
(+`--sda-*` knobs). The takeaway: for this linear-Gaussian-likelihood window problem, **Adam (D-Flow)
> crude Langevin**; SDA's advantage would show with a true trajectory score model (not a one-step FM)
or a strongly nonlinear likelihood — a P4/real-data lever, not a win here.

**Key files (P3.5–3.7)**: `inference/samplers/{base,fmps,flowdps,sde,mcg,pcfm}.py`,
`carbonbench/.../27_mip_style_osse/{diag_propagation.py, analyze_da_gain.py}`, eval runner `--sampler`.

**Key files**: `neural_transport/inference/orbit_obs.py` (`OrbitObsProvider`), `inference/
generation.py` (`_build_orbit_enkf_obs` + `orbit_obs` path), `forward_model.effective_column_kernel`
(P1), `inference/generation.py` (`generate_trajectory_ensemble_batched` orbit path for FMPS),
`carbonbench/.../27_mip_style_osse/{eval_mip_osse.py, analyze_da_gain.py, run_mip_osse.slurm,
summarize.py, README.md}`, `tests/test_orbit_obs.py` (+12). **Deliverable**: MIP-like OSSE runs with known-truth
scores ✅ (machinery, smoke, full stats + 5-config realism sweep). **Gate**: stable runs ✅; skill
under realistic sampling quantified ✅; synthetic→idealised gap measured ✅.

> **Forward note for P4**: the long-lead EnKF degradation (lead 119: −9 % vs free) is the headline
> shortcoming to dissect. Levers to test in P4: (a) **larger ensemble** `n_samples` (n=10 → 20/40) to
> cut spurious covariance — the most likely real fix; (b) **stronger/adaptive localization**
> (`loc_sigma`, currently 4); (c) **obs-error inflation** (already shown to help: `sweep_noise03`);
> (d) **EnKS** (smoother) vs EnKF. Also fold in the idealised SOTA OSSE (30 % coverage, 2.54 ppm) as
> the dense-obs reference point on the obs-density saturation curve. **FMPS/D-Flow now run on the
> realistic orbit obs** (P3.6) and FMPS loses to EnKF here — a key P4 result is *when* the generative
> sampler is worth its variance (obs-density crossover where FMPS overtakes EnKF; and the real-data
> P6 regime). Calibration angle: prior inflation buys spread/err 0.26→0.41 at an RMSE cost, and FMPS
> is natively better-dispersed (0.30) but less accurate — the dispersion-vs-accuracy trade is a
> calibration story for P4/P8.

### P3.9 — Ceiling diagnostics: *why* the skill is "unimpressive" (the binding constraint)

Before chasing better samplers, we measured the achievable ceiling by feeding the EnKF
**oracle** information (`generate_trajectory_enkf` gains `perfect_obs` = replace the 3-D state
with truth at obs cells, `dense_obs` = observe the column at *every* cell, `init_perturb` =
corrupt the IC). 8 inits × 120 leads:

| experiment | RMSE | vs its free |
|---|---|---|
| free (good IC) | 1.9537 | — |
| EnKF (good IC) | 1.8823 | −3.7 % |
| **perfect 3-D truth @ 2 % orbit cells** | **1.8539** | −5.1 % ← coverage ceiling |
| dense column @ 100 % cells | 1.7559 | −10.1 % |
| dense + perfect-3-D @ 100 % | 0.7062 | −63.9 % (1–3-step forecast floor) |
| free (climatology IC) | 9.4120 | — |
| **EnKF (climatology IC)** | **7.6560** | **−18.5 %** ← DA works when the forecast is wrong |

**Three decisive findings:**
1. **Coverage is the binding constraint at OCO-2 density.** Even *perfect 3-D truth* inserted at
   the ~2 % observed cells each step reaches only 1.854 — and EnKF (1.882) is already within 1.5 %
   of that oracle. **No sampler/amortization can meaningfully beat ~1.85 at 2 % coverage**; the
   field is information-starved and transport propagates the constraint only slowly. The
   "few-percent" gain is a *physics ceiling*, not an algorithm failure.
2. **Headroom lives at higher coverage + in the vertical.** 2 %→100 % column drops 1.88→1.756;
   column→perfect-3-D at 100 % drops 1.756→0.706. So (a) **more observations** (denser networks,
   multi-instrument, temporal super-obs) is the dominant lever, and (b) **vertical de-aliasing**
   (`p(profile|column)`) is a large lever *only once coverage is high* — exactly where a
   generative prior should win over the linear EnKF.
3. **DA value scales with forecast error.** Good IC → −3.7 %; climatology IC → −18.5 %. The
   identical-twin forecast is so good there is little to correct; the unimpressive headline is the
   twin ceiling. The impressive-DA demonstration lives in a **model-error / real-data** regime.

### P3.10 — Amortized conditional FM (learn p(state|obs), one-pass)

Built `AmortizedResidualFlowMatching` (registered): the velocity UNet gains 2 input channels
(obs_value, obs_mask; `in_chans` 51→53) and trains with a randomly-masked (0–30 % coverage)
nominal-column observation of the target, learning `p(x_next | x_t, f_det, y)` so observations are
baked into a **single generation pass** — no test-time guidance/optimisation. Fine-tuned from a
surgically channel-expanded phase-2p checkpoint (`make_init_ckpt.py`; the time channel moves 50→52,
obs channels zero-init). Eval via `generate_trajectory_amortized` / `eval_mip_osse --method
amortized`, at orbit (2 %) or `--dense-obs` (100 %) coverage with the training-consistent column
operator.

> **Bug fixed en route (affects all `pretrained_ckptpath` fine-tunes):** the litmodule used
> `k.replace("model.","")`, which mangles `submodel.`→`sub` and silently drops every FM-head weight
> on load (amortized step-0 loss 36 vs base 0.8). Fixed to strip only the leading prefix → step-0
> loss 0.57.

**Expectation set by P3.9:** the amortized model will *not* beat EnKF at 2 % coverage (nothing can —
oracle ceiling 1.854); its value is in the **dense regime** (vertical de-aliasing toward the 0.706
floor) and on **real data** (model error).

**First result (fine-tune, random masks 0–30 % coverage, val 0.793 < base 0.805).** 8×120:
- orbit (2 %): RMSE **1.971** ≈ free 1.954 (no help — as P3.9 predicts, no headroom at 2 %).
- dense (100 %): RMSE **2.327** — *worse* than free, and far above dense-EnKF (1.756). The model was
  trained only up to 30 % coverage, so 100 % dense obs is **out-of-distribution** → it misreads the
  obs-value/mask channels. Confirms: the dense de-aliasing test needs a model trained for dense
  coverage. **Retraining with 0–100 % coverage** to give the de-aliasing hypothesis a fair test.

---

## P4 — OSSE skill & shortcoming analysis

**Goal**: Turn P3 (and the idealised SOTA OSSE) into the controlled-setting story: *what FM-DA can and
cannot do, and why* — the analysis reviewers need before trusting real-data results.

### Tasks
- [ ] **Sampler comparison table**: free / EnKF(loc=4) / FMPS / D-Flow — RMSE, CRPS, spread/err, wall-time.
- [ ] **Skill-vs-lead** curve (RMSE@1 … @119); quantify the long-lead wall.
- [ ] **Obs-density / realism sweep** results from P3 → skill saturation curves.
- [ ] **Vertical & spatial error structure**: lat-height cross-sections, global RMSE/bias maps,
      orbit-track stripe artifacts.
- [ ] **Calibration**: rank histograms, spread-skill, CRPS decomposition; diagnose under-dispersion.
- [ ] **Failure modes**: spatial bias drift (archive 25o/p), masking artifacts (README limitation).
- [ ] Publication-quality figures via `plots/run_plots()` (always + ensemble + conditioning).

**Deliverable**: OSSE results section (figures + table). **Gate**: every claim backed by a saved
figure/number; sampler table reproduces the headline.

---

## P5 — OCO-2/OCO-3/TCCON/in-situ data prep (MIP v11)

**Goal**: Rebuild all observation datasets **from scratch from the v11 raw inputs**, aligned with
**OCO-2 MIP v11**, with a clear assimilate-vs-validate split, plus the **co-sampled participant model
outputs** as benchmarks.

### Tasks
- [ ] Re-download v11 raw inputs from the download page: `OCO2_b11.2_10sec_GOOD_r2.nc4`,
      `OCO3_b11_10sec_GOOD_r2.nc4`, `obspack_co2_1_OCO2MIP_v5.0.1_2024-11-14`, `tccon...R20250609`,
      and the **co-sampled participant submissions** (preliminary output / release page).
- [ ] OCO-2 + OCO-3: regrid via `regrid_mip_oco2` → model grid; **keep native 20-level** AK/`h`/
      a-priori/pressure_levels (for the P1 operator); ingest OCO-3 analogously (`datasets/mip_oco3.py`).
- [ ] TCCON: prepare as **independent** validation (station XCO₂ + AKs).
- [ ] In-situ / ObsPack v5.0.1: prepare surface + aircraft as independent validation (reuse carbonbench
      ObsPack pipeline; licensing — not on HuggingFace).
- [ ] **Co-sampled participant outputs**: ingest into a comparable structure (co-located CO₂ at each
      validation obs) for direct benchmarking in P8.
- [ ] Define & freeze the **assimilate set** (what FM-DA ingests) vs **validation set** (strictly held
      out) — mirror the MIP experiment definitions so the comparison is fair.
- [ ] Tests: shapes, units (ppm), time alignment, no assimilate/validate leakage.

**Key files**: `datasets/mip_oco2.py`, new `datasets/mip_oco3.py`, `datasets/tccon.py`,
`data/oco2_loader.py` (extend to OCO-3/TCCON). **Deliverable**: zarr datasets + data README
(versions, flags, splits, benchmark sources). **Gate**: coverage matches published MIP v11 numbers.

---

## P6 — Real OCO2MIP-style 3-D concentration assimilation

**Goal**: Assimilate **real** OCO-2 (and OCO-3) XCO₂ over the MIP period, producing the **full 3-D CO₂
concentration field** via FM posterior sampling — **no flux estimation**. Time-series / windowed DA.

### Tasks
- [ ] Drive `oco2_assimilate` (corrected AK op from P1) through `OCO2DataLoader` → `GenerationPipeline`
      time-series mode → `generate_trajectory_enkf` (and FMPS/D-Flow).
- [ ] Smoke on one short window first, then scale to the full MIP test period.
- [ ] DA cadence/window matched to obs availability; handle sparse swaths + window aggregation.
- [ ] Ensemble run for uncertainty (spread maps).
- [ ] Mitigate the real-obs drop characterised in P3: noise_scale, localization, obs-error inflation;
      log what helps.
- [ ] Save assimilated 3-D fields + XCO₂ + ensemble spread as zarr.

**Key files**: new `carbonbench/.../26_real_oco2_inversion/`, `data/oco2_loader.py`,
`inference/generation.py`. **Deliverable**: assimilated CO₂ over the MIP period + posterior/uncertainty
maps. **Gate**: stable multi-month run, no NaNs, physically plausible fields.

---

## P7 — Independent validation harness (held-out obs)

**Goal**: Score the assimilated CO₂ against **held-out OCO-2, OCO-3, TCCON, and in-situ** the MIP way —
posterior CO₂ vs independent observations, AK applied where appropriate.

### Tasks
- [ ] Collocate model fields to each validation obs (P1 operator for column obs; vertical interp for
      profiles/in-situ).
- [ ] Metrics via `EvaluationSuite`: RMSE, bias, correlation — global, by latitude band, by season,
      by TransCom region where feasible.
- [ ] Separate **assimilated-network** check (fit) from **independent-network** check (real skill).
- [ ] Ensemble validation: CRPS, rank histograms, spread vs independent-obs error (calibration).
- [ ] Baselines for context: free-run (no DA), a priori, CT2022 mole fractions.

**Key files**: `evaluation/suite.py`, new `evaluation/obs_collocation.py`,
`carbonbench/.../26_real_oco2_inversion/validate.py`. **Deliverable**: validation score tables +
figures. **Gate**: independent-obs RMSE computed for all four obs types with documented methodology.

---

## P8 — Comparison to OCO-2 MIP + calibration analysis

**Goal**: Place our independent-obs skill next to the OCO-2 MIP participants (via co-sampled outputs
and published numbers) and deliver the verdict: comparable, or worse-by-how-much, with diagnosis.

### Tasks
- [ ] Tabulate our held-out RMSE/bias vs the **co-sampled participant outputs** at the same obs, and
      vs published MIP aircraft/TCCON skill (~0.88–1.91 ppm flux-attributable; ~1–3.5 ppm total) —
      matched regions/period/obs.
- [ ] Honest framing: we estimate *concentration*, MIP estimates *fluxes* — compare the observable
      both produce (CO₂ vs independent obs).
- [ ] **Calibration story**: our FM ensemble spread vs the MIP ensemble's known under-dispersion
      (1.3–1.9×). A well-calibrated FM posterior is a genuine contribution.
- [ ] If materially worse: decompose error (AK op, real-obs drop, transport bias, sparsity) and quantify.
- [ ] Sensitivity: ±OCO-3, ±in-situ, DA cadence.

**Deliverable**: comparison tables + figures + a defensible, quantified statement of where FM-DA sits
relative to the MIP. **Gate**: verdict supported by co-sampled benchmark + independent obs.

---

## Explicitly out of scope (future work / paper 2)
- **Surface flux estimation** (state-augmented ensemble inversion with NN transport as forward
  operator) — the path to head-to-head MIP *flux* competitiveness. Deferred.
- Mid/high-res grids; multi-satellite joint flux inversion; physics-informed guidance.
- **Manuscript writing** — out of scope for this plan (code + results focus).

## Dependency graph
```
P1 (AK fix) ──► P3 (MIP-style OSSE) ──► P4 (OSSE analysis)
P2 (leak-free training) ──► P3
P1 ──► P5 (data prep, native-level AK) ──► P6 (real assimilation) ──► P7 (validation) ──► P8 (MIP comparison)
P2 ──► P6
```
P1 and P2 can run in parallel (different sessions). P3 needs P1+P2. P5 needs P1. P6 needs P2+P5. P8 needs P4+P7.
