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
- [ ] **Tests first** (`tests/test_forward_model.py`): on a sample of real soundings (from
      `oco2_assimilate.zarr` / the raw nc4), compare (a) current down-aggregated op vs
      (b) interpolate-then-apply op; quantify the discrepancy. **Reference check**: applying the
      operator to the retrieval's own a-priori profile must reproduce `xco2_apriori` to round-off.
- [ ] Add `XCO2ForwardModel` mode taking model `(p_model, x_model)` + retrieval
      `(p_ret[20], h[20], a[20], x_apriori[20], xco2_prior)`: pressure-aware interpolation
      `x_model→p_ret` (mixing-ratio-consistent), then the averaging formula at 20 levels.
- [ ] Keep it **differentiable** — samplers / DA need `jacobian_transpose` / `project`; add the
      interpolation to the adjoint path.
- [ ] Wire through `data/oco2_loader.py` → `masking_config` / `ObservationBatch` so DA uses the
      corrected operator; keep the old path behind a flag for the ablation.
- [ ] Decide storage: keep **native 20-level** AK / `h` / a-priori / pressure_levels in the obs
      products (don't pre-aggregate to l10) so the corrected op has what it needs.

**Key files**: `forward_model.py`, `datasets/mip_oco2.py`, `data/oco2_loader.py`,
`tests/test_forward_model.py`. **Deliverable**: verified forward operator + short note quantifying
old-vs-new discrepancy. **Gate**: a-priori reproduction test passes; interpolate-then-apply is default.

---

## P2 — Leak-free model training (≤2013 train, 2014 val)

**Goal**: Retrain the residual-FM model with the SOTA recipe but on data **only through 2013, with
2014 as validation**, so the entire OCO-2 MIP period (2015–2020) is held out — no leakage for the
independent-method claim.

### Tasks
- [ ] Verify current CarbonTracker train/val split and the SOTA model's actual training years.
- [ ] Re-prepare CT datasets: **train ≤2013, val 2014** (test = MIP period 2015–2020).
- [ ] Retrain `f_det`: single-step → rollout-FT (K=4 uniform, `lr_mult=0.1`, EMA 0.999) → EMA-freeze.
- [ ] Recompute `sigma_res`; retrain residual FM head (noise_scale 1.05, ~20k steps).
- [ ] **Sanity gate**: re-run the SOTA OSSE on a ≤2014 window; confirm skill ≈ SOTA (~2.5 ppm regime).
      A large regression means the earlier period is materially harder — document it.

**Key files**: `carbonbench/.../25c_v4_residual_fm/{phase1p_*, phase2p_*}` (clone with leak-free data
paths), `training/train.py`. **Deliverable**: leak-free checkpoints + OSSE-parity note.
**Gate**: no 2015–2020 data in train/val; OSSE skill within ~10 % of SOTA.

---

## P3 — OCO2MIP-style OSSE (realistic synthetic obs)

**Goal**: Build an OSSE that *resembles the OCO-2 MIP setup* — synthetic XCO₂ observations sampled
from a known CarbonTracker truth using **real OCO-2 orbit tracks, the corrected AK operator (P1),
realistic sparsity, and retrieval noise** — and run FM-DA on it.

**Why**: Bridges the idealised OSSE and the real experiment. It lets us measure skill against a
*known truth* under realistic observing conditions, and characterise the synthetic→real
"performance drop" the README flags, before touching real obs.

### Tasks
- [ ] Sample obs locations/times from real OCO-2 (and OCO-3) swaths over a test window; apply the
      P1 forward operator + retrieval-error noise to CarbonTracker truth → synthetic XCO₂.
- [ ] Run FM-DA (EnKF loc=4, FMPS, D-Flow) over the window; reconstruct the 3-D field.
- [ ] Compare reconstruction to the (known) truth: RMSE/bias of field + XCO₂, plus at synthetic
      "held-out" sounding locations (mimicking the validation split).
- [ ] Sweep realism knobs: swath sparsity, retrieval noise level, AK on/off, obs cadence.
- [ ] Use the leak-free model from P2.

**Key files**: new `carbonbench/.../27_mip_style_osse/`, `inference/masking.py` (orbit-track masks),
`inference/generation.py`. **Deliverable**: MIP-like OSSE runs with known-truth scores.
**Gate**: stable runs; quantified skill under realistic sampling; synthetic→idealised gap measured.

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
