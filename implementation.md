# Flow-Matching Data Assimilation of Satellite XCO₂ — Publication Plan

> **Scope (decided 2026-06-01).** Track A: present flow-matching (FM) probabilistic data
> assimilation as an *independent ML method* for assimilating satellite column CO₂. Motivate
> it in a controlled OSSE, then run a real **OCO-2 MIP v11-style** assimilation that estimates
> the **full 3-D concentration field** (we do **not** estimate surface fluxes). Validate the
> assimilated CO₂ against **held-out OCO-2, OCO-3, TCCON, and in-situ** observations and ask:
> *are our errors comparable in magnitude to the OCO-2 MIP participants?* If they are, we can
> claim an independent, similarly-accurate method. If they are worse, we quantify by how much
> and diagnose why.
>
> The full development history (refactor Phases 1–17, method development Phases 18–25p, and
> all documented negative results) lives in **`implementation_archive.md`**. This file is the
> forward-looking plan only.

---

## Current state (entry point for new sessions)

- **Repos**: `neural_transport` (library) + `carbonbench` (experiments). Branch `feature/xco2-l1`,
  both committed & pushed through Phase 25p as of 2026-06-01.
- **OSSE SOTA**: **2.513 ppm XCO₂ RMSE** (n=100), CRPS 1.030, spread/err 0.32 —
  Phase 2p **EnKF (loc=4)** on the residual-FM model (deterministic backbone `f_det`
  rollout-fine-tuned with EMA shadow weights + a flow-matching residual head).
  Free (no-DA) run ≈ 3.24 ppm. Cross-seed confirmed. *This is concentration-field
  reconstruction against a held-out CarbonTracker run, with synthetic obs masks.*
- **Model recipe (SOTA)**: `f_det` = single-step train → rollout-FT (K=4, uniform step
  weights, `lr_mult=0.1`, EMA decay 0.999) → EMA-frozen checkpoint; residual FM head trained
  on top (`sigma_res`, noise_scale 1.05). See archive Phase 25p.
- **Library is modular**: `configs`, `forward_model`, `inference/samplers/*`, `inference/masking`,
  `inference/noise`, `inference/generation`, `evaluation/*`, `plots/*`, `data/{inference_loader,
  oco2_loader}`, `datasets/{carbontracker,mip_oco2}`.

### Real-data assets already staged on tscratch
`/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/`
- `OCO2MIP_OCO2/oco2_assimilate.zarr` — raw OCO-2 soundings (XCO₂, AK, a priori, sigma levels, psurf).
- `OCO2MIP_OCO2/val/mip_oco2_latlon5.625_l10_6h.zarr` — **regridded to our lowres model grid**.
- `OCO2MIP_OCO3/`, `OCO2MIP_TCCON/` (tccon_timeaverages_R20250609), `OCO2MIP_OCO2/OCO2_b11.2_10sec_GOOD_r2.nc4`.
- `Carbontracker/CT2022_flux`, `CT2022_molefrac` — reference fields / fluxes.
- Data prep code: `neural_transport/datasets/mip_oco2.py` (`regrid_mip_oco2`, `vertical_aggregation_oco2`).

### Validation targets (researched 2026-06-01)
OCO-2 v10 MIP, aircraft evaluation ([Jacobson et al., ACP 2025](https://acp.copernicus.org/articles/25/1725/2025/)):
the **flux-attributable** component of posterior-CO₂-vs-aircraft RMSE is **0.88–1.91 ppm** per
region (55–85 % of total regional RMSE → total mismatch ≈ 1–3.5 ppm). The MIP ensemble *spread
under-estimates* true error by 1.3–1.9× — calibration is an open community problem, and a place
where FM's flexible posterior can contribute. ([OCO-2 v10 MIP portal](https://www.gml.noaa.gov/ccgg/OCO2_v10mip/).)
**Our bar**: independent-obs RMSE in the same ballpark (low single-digit ppm), well-calibrated spread.

### ⚠️ Known correctness gap — averaging kernel (verify in P3)
The OCO-2 MIP docs prescribe: **interpolate the model CO₂ profile to the retrieval's native
(20) pressure levels, then apply** `XCO2 = xco2_prior + Σ_k h_k·a_k·(x_k − x_apriori_k)` at those
levels. We currently do the **opposite**: `vertical_aggregation_oco2` (`datasets/mip_oco2.py`)
collapses the 20-level AK / `h` / a-priori *down* to the model's coarse `l10` grid and applies the
formula there (`forward_model.py:128`). These agree only if the profile is linear within each
aggregation group. **This must be fixed/validated before the real-data comparison is credible.**

---

## Roadmap overview

| Phase | Title | Output |
|---|---|---|
| **P1** | OSSE skill & shortcoming analysis (controlled setting) | Paper §OSSE + figures |
| **P2** | OCO-2 / OCO-3 / TCCON / in-situ data prep (MIP v11-aligned) | Clean obs datasets + splits |
| **P3** | Averaging-kernel forward operator — correctness fix | Verified `XCO2ForwardModel` |
| **P4** | Leak-free model training (≤2013 train, 2014 val) | Retrained residual-FM |
| **P5** | Real OCO2MIP-style 3-D concentration assimilation | Assimilated CO₂ fields |
| **P6** | Independent validation harness (held-out obs) | MIP-style skill scores |
| **P7** | Comparison to OCO-2 MIP + calibration analysis | Paper §Real-data + verdict |
| **P8** | Manuscript (GMD / JAMES / ACP) | Submission |

**Working conventions** (unchanged from archive): TDD where practical; after each phase run
`pytest tests/ -m "not slow and not gpu"`, the toy OSSE gate, and `ruff check`. Long runs go via
SLURM; artifacts on tscratch with symlinks (`scripts/setup_experiment_storage.sh`).

---

## P1 — OSSE skill & shortcoming analysis (controlled setting)

**Goal**: Turn the existing OSSE SOTA into the paper's controlled-setting story: *what FM-DA can
and cannot do, and why*. No new modelling — analysis and figures on the current SOTA model.

**Why**: Reviewers need a controlled experiment (known truth) to trust the real-data results.
This is where we characterise skill vs lead time, observation density, vertical structure, and
calibration — and honestly surface failure modes (bias drift, mask/orbit-track artifacts).

### Tasks
- [ ] Lock the SOTA OSSE config as a reproducible baseline (frozen `eval_trajectory_v2.py --phase 25p2`
      EnKF loc=4, n=100, fixed seeds). Record exact ckpt paths + data window.
- [ ] **Sampler comparison table**: free / EnKF(loc=4) / FMPS / D-Flow — RMSE, CRPS, spread/err,
      wall-time. (Pareto: skill vs cost.)
- [ ] **Skill-vs-lead** curve (RMSE@1 … @119) for each method; quantify the long-lead wall.
- [ ] **Obs-density sweep** (obs-every / obs-fraction) → skill saturation curve.
- [ ] **Vertical & spatial error structure**: lat-height error cross-section, global RMSE/bias maps,
      identify orbit-track stripe artifacts if present.
- [ ] **Calibration**: rank histograms, spread-skill, CRPS decomposition. Diagnose under-dispersion.
- [ ] **Failure-mode write-up**: spatial bias drift (archive Phase 25o/p), masking artifacts
      (README known limitation).
- [ ] Produce publication figures via `plots/run_plots()` (always + ensemble + conditioning).

**Deliverable**: OSSE results section (text + ~5 figures + 1 table). **Gate**: every claim backed
by a saved figure/number; sampler table reproduces 2.513 ppm headline.

---

## P2 — OCO-2 / OCO-3 / TCCON / in-situ data prep (MIP v11-aligned)

**Goal**: Produce clean, documented observation datasets for OCO-2, OCO-3, TCCON, and in-situ
(ObsPack), aligned with **OCO-2 MIP v11** conventions, with a clear assimilate-vs-validate split.

**Why**: Credible comparison requires using the same observations, quality flags, bias-correction,
and assimilate/validation partitioning as the MIP. Note staged data appears to be v10/b11.2 —
**confirm against MIP v11** (obs versions, period 2015–2020, experiment definitions).

### Tasks
- [ ] Confirm OCO-2 MIP **v11** spec: obs product versions, time range, quality flags, the
      assimilation experiments (e.g. LNLG / OG / IS combinations), and which obs are held out for
      independent validation. Document deltas vs the staged v10/b11.2 data.
- [ ] OCO-2: finalize `regrid_mip_oco2` → model grid (latlon5.625_l10_6h); keep **native 20-level**
      AK / `h` / a-priori / pressure_levels available for the corrected forward op (P3) — i.e. do
      **not** pre-aggregate to l10 in the stored product, or store both.
- [ ] OCO-3: ingest analogously (`OCO2MIP_OCO3`).
- [ ] TCCON: prepare `tccon_timeaverages_R20250609` as an **independent** validation set (station
      XCO₂ time series + AKs).
- [ ] In-situ / ObsPack: prepare surface + aircraft profiles as independent validation (reuse
      carbonbench ObsPack pipeline; note licensing — not on HuggingFace).
- [ ] Define and freeze the **assimilate set** (what FM-DA ingests) vs **validation set**
      (strictly held out) — mirror MIP so the comparison is fair.
- [ ] Tests: shapes, units (ppm), time alignment, no assimilate/validate leakage.

**Key files**: `datasets/mip_oco2.py`, new `datasets/mip_oco3.py`, `datasets/tccon.py`,
`data/oco2_loader.py` (extend to OCO-3/TCCON). **Deliverable**: zarr datasets + a one-page data
README documenting versions, flags, splits. **Gate**: counts/coverage match published MIP v11 numbers.

---

## P3 — Averaging-kernel forward operator (correctness fix)

**Goal**: Implement the MIP-prescribed XCO₂ forward operator — **interpolate model profile to the
retrieval's native pressure levels, then apply the averaging formula there** — and prove it correct.

**Why**: See the ⚠️ note above. Our current down-aggregation of the AK biases the obs operator and
would invalidate the MIP comparison. This is the single highest-leverage correctness item.

### Tasks
- [ ] **Tests first** (`tests/test_forward_model.py`): on real soundings, compare
      (a) current down-aggregated operator vs (b) interpolate-then-apply operator; quantify the
      discrepancy. Add a reference check: applying the operator to the retrieval's own a-priori
      profile must reproduce `xco2_apriori`.
- [ ] Add `XCO2ForwardModel` mode that takes model `(p_model, x_model)` + retrieval
      `(p_ret[20], h[20], a[20], x_apriori[20], xco2_prior)`, interpolates `x_model→p_ret`
      (pressure-aware, mass-/mixing-ratio-consistent), then `xco2 = xco2_prior + Σ h·a·(x_interp − x_apriori)`.
- [ ] Keep it **differentiable** (samplers/DA need `jacobian_transpose` / `project`); add the
      interpolation to the adjoint path.
- [ ] Wire through `oco2_loader.py` → `masking_config` / `ObservationBatch` so DA uses the corrected op.
- [ ] Re-run a small OSSE/real slice both ways; report the skill delta.

**Deliverable**: verified forward operator + a short note quantifying old-vs-new discrepancy.
**Gate**: a-priori reproduction test passes; interpolate-then-apply is the default for real obs.

---

## P4 — Leak-free model training (≤2013 train, 2014 val)

**Goal**: Retrain the residual-FM model with the SOTA recipe but on data **only through 2013, with
2014 as validation**, so the entire OCO-2 MIP period (2015–2020) is held out and there is no leakage.

**Why**: The current SOTA model's training window may overlap the MIP period. For a clean
independent-method claim, the FM transport model must never have seen 2015–2020.

### Tasks
- [ ] Verify current CarbonTracker train/val split and the SOTA model's actual training years.
- [ ] Re-prepare CT train/val datasets: **train ≤2013, val 2014** (test = MIP period).
- [ ] Retrain `f_det`: single-step → rollout-FT (K=4 uniform, `lr_mult=0.1`, EMA 0.999) → EMA-freeze.
- [ ] Recompute `sigma_res`; retrain residual FM head (noise_scale 1.05, ~20k steps).
- [ ] **Sanity gate**: re-run the P1 OSSE on a ≤2014 window; confirm skill ≈ SOTA (2.5 ppm regime).
      A large regression means the earlier period is materially harder — document it.

**Key files**: `carbonbench/.../25c_v4_residual_fm/{phase1p_*, phase2p_*}` (clone with leak-free
data paths), `training/train.py`. **Deliverable**: leak-free checkpoints + OSSE parity note.
**Gate**: no 2015–2020 data in train/val; OSSE skill within ~10 % of SOTA.

---

## P5 — Real OCO2MIP-style 3-D concentration assimilation

**Goal**: Assimilate **real** satellite XCO₂ over the MIP period, producing the **full 3-D CO₂
concentration field** via FM posterior sampling — **no flux estimation**. Time-series / windowed DA.

**Why**: This is the core real-data experiment. The README flags a known performance drop going
from synthetic 3-D obs to real 2-D satellite obs — characterising and mitigating that is a central
contribution.

### Tasks
- [ ] Drive `oco2_assimilate` (corrected AK op from P3) through `OCO2DataLoader` →
      `GenerationPipeline` time-series mode → `generate_trajectory_enkf` (and FMPS/D-Flow).
- [ ] One short window end-to-end first (smoke), then scale to the full MIP test period.
- [ ] Choose DA cadence/window to match obs availability; handle sparse swaths + window aggregation.
- [ ] Ensemble run for uncertainty (spread maps).
- [ ] Mitigate the real-obs drop: revisit noise_scale, localization, obs-error inflation; log what helps.
- [ ] Save assimilated 3-D fields + XCO₂ + ensemble spread as zarr.

**Key files**: new `carbonbench/.../26_real_oco2_inversion/`, `data/oco2_loader.py`,
`inference/generation.py`. **Deliverable**: assimilated CO₂ fields over the MIP period + global
posterior/uncertainty maps. **Gate**: stable multi-month run, no NaNs, physically plausible fields.

---

## P6 — Independent validation harness (held-out obs)

**Goal**: Score the assimilated CO₂ against **held-out OCO-2, OCO-3, TCCON, and in-situ** the way the
MIP does — posterior CO₂ vs independent observations, with AK applied where appropriate.

### Tasks
- [ ] Collocate model fields to each validation obs (apply AK for column obs; vertical interp for
      profiles/in-situ) using the P3 operator.
- [ ] Metrics via `EvaluationSuite`: RMSE, bias, correlation — globally, by latitude band, by
      season, and by TransCom region where feasible.
- [ ] Separate **assimilated-network** check (fit) from **independent-network** check (the real skill).
- [ ] Ensemble validation: CRPS, rank histograms, spread vs independent-obs error (calibration).
- [ ] Baselines for context: free-run (no DA), a priori, and CT2022 mole fractions.

**Key files**: `evaluation/suite.py`, new `evaluation/obs_collocation.py`,
`carbonbench/.../26_real_oco2_inversion/validate.py`. **Deliverable**: validation score tables +
figures. **Gate**: independent-obs RMSE computed for all four obs types with documented methodology.

---

## P7 — Comparison to OCO-2 MIP + calibration analysis

**Goal**: Place our independent-obs skill next to the OCO-2 MIP participants and deliver the paper's
verdict: comparable, or worse-by-how-much, with diagnosis.

### Tasks
- [ ] Tabulate our held-out RMSE/bias vs published MIP aircraft/TCCON skill (≈0.88–1.91 ppm
      flux-attributable; ~1–3.5 ppm total) — matched regions/period/obs where possible.
- [ ] Honest accounting: we estimate *concentration*, MIP estimates *fluxes* — frame the comparison
      precisely (we compare the observable both produce: CO₂ vs independent obs).
- [ ] **Calibration story**: compare our FM ensemble spread to the MIP ensemble's known
      under-dispersion (1.3–1.9×). A well-calibrated FM posterior is a genuine contribution.
- [ ] If materially worse: decompose error (AK op, real-obs drop, transport-model bias, sparsity)
      and quantify each.
- [ ] Sensitivity: with/without OCO-3, with/without in-situ, DA cadence.

**Deliverable**: real-data results section + comparison table + verdict. **Gate**: a defensible,
quantified statement of where FM-DA sits relative to the MIP.

---

## P8 — Manuscript

**Target**: GMD / JAMES / ACP. Outline:
1. Intro — satellite CO₂ DA, ML emulators, generative DA, the independent-method angle.
2. Method — neural transport + residual FM, XCO₂ forward op (corrected AK), posterior samplers (EnKF/FMPS/D-Flow).
3. OSSE — controlled skill & shortcomings (P1).
4. Real data — OCO2MIP-style assimilation (P5) + independent validation (P6).
5. Comparison to OCO-2 MIP + calibration (P7).
6. Discussion — strengths, the real-obs drop, no-flux scope, cost vs 4D-Var/EnKF.
7. Conclusion + outlook (flux estimation = future work).

- [ ] Assemble figures/tables from P1, P6, P7. [ ] Draft. [ ] Internal review. [ ] Submit.

---

## Explicitly out of scope (future work / paper 2)
- **Surface flux estimation** (state-augmented ensemble inversion with NN transport as forward
  operator) — the path to head-to-head MIP *flux* competitiveness. Deferred to a second paper.
- Mid/high-res grids; multi-satellite joint flux inversion; physics-informed guidance.

## Dependency graph
```
P1 (OSSE analysis) ──────────────────────────────► P8 (paper §OSSE)
P2 (data prep) ──► P3 (AK forward op) ──► P5 (real assimilation) ──► P6 (validation) ──► P7 (MIP comparison) ──► P8
P4 (leak-free training) ───────────────► P5
```
P1 and (P2→P3, P4) are parallelizable across sessions. P5 needs P3 + P4. P7 needs P6. P8 needs P1+P7.
