# Flow Matching Posterior Conditioning for CO2 Transport — Implementation Plan

## Context

This project uses Flow Matching generative models to sample 3D atmospheric CO2 fields, with the goal of conditioning on sparse satellite observations (OCO-2 XCO2 columns) for data assimilation. The unconditional model works well, but **posterior conditioning does not yet produce satisfactory results**: all four implemented methods (correction, velocity projection, guidance, repaint) either create spatial artifacts, have weak effect, or are worse than unconditional generation.

The core issue is a **distribution mismatch**: column-level constraints create uniform vertical shifts that are out-of-distribution for the UNet, and current guidance lacks proper likelihood gradient computation. The OSSE comparison (experiment 08) shows that only weak/late guidance provides marginal improvement (RMSE 3.3 vs 4.3 unconditional), while hard-constraint methods (correction, repaint) increase RMSE to ~5.0.

**Goal**: Fix conditioning, implement SOTA posterior sampling, benchmark on OSSE + real OCO-2, publish as **application paper** (ACP/GMD).

**Primary test case**: 2D total-column XCO2 observations (OCO-2 style). Forward model: `XCO2 = sum_k h_k a_k x_k`. 3D/other patterns are secondary "what if" experiments.

**Strategy**: (1) Evaluation infra + cleanup, (2) fix vanilla FM training, (3) training-free posterior methods, (4) advanced FM training + CFM retraining, (5) comparison + real data + paper. Every method passes **toy column OSSE** before full experiments.

---

## Current State Summary

### What Works
- [x] FM model trained on CarbonTracker (5.625deg, 10 levels, UNet)
- [x] Unconditional generation: RMSE ~4.3 ppm
- [x] Data pipeline: CT loading, OCO-2, vertical aggregation
- [x] OSSE framework with multiple mask patterns
- [x] Toy column OSSE for fast debugging
- [x] 4 conditioning modes: correction, velocity_projection, guidance, repaint
- [x] OCO-2 forward model (`compute_xco2`) with AK support

### What Doesn't Work
- [ ] Column conditioning: spatial artifacts, weak effect, or worse than unconditional
- [x] 9+ near-duplicate masking methods (removed 8, kept 4)
- [x] `compute_xco2` fallback missing targshift correction (fixed)
- [x] Guidance: uniform correction instead of Jacobian transpose (fixed)
- [ ] No spatial smoothing of guidance (TODO in code)

### Key Files
| File | Role |
|------|------|
| `neural_transport/models/flowmatching.py` | FlowMatching + MaskedVelocityWrapper |
| `neural_transport/inference/generative.py` | Inference, mask creation, OCO-2 |
| `neural_transport/datasets/carbontracker.py` | CT data, vertical aggregation |
| `neural_transport/experiments/toy_column_osse.py` | Debugging testbed |
| `neural_transport/plots/plot_results.py` | Visualization |
| `carbonbench/.../08_fm_unet_osse_conditioning_comparison/` | Latest OSSE experiment |

---

## Additional Development Guidelines

- **Fast iteration**: Always use minimal test cases during development
- **E2E after each phase**: Run full pipeline after each phase, even on toy data
- **Plots are deliverables**: Every phase must produce visual evidence

---

## Phase 1: Evaluation Infrastructure

Build reusable evaluation so every subsequent phase auto-produces full diagnostics.

- [x] **Create** `neural_transport/inference/metrics.py`:
  `rmse_3d`, `rmse_xco2`, `rmse_at_obs`/`rmse_away`, `crps_ensemble`, `spread_skill_ratio`, `calibration_score`, `rank_histogram`, `spatial_roughness`, `compute_all_metrics`
- [x] **Create** `neural_transport/plots/conditioning_diagnostics.py`:
  `plot_conditioning_comparison` (GT/obs/ens.mean/|error|/samples grid), `plot_metrics_summary`, `plot_ensemble_diagnostics`, `plot_xco2_maps`
- [x] **Create** `neural_transport/inference/osse_runner.py`:
  `run_single_osse(model, config, gt, obs) -> OSSEResult`, `run_osse_comparison`, `save_osse_results`

**Deliverable**: Every subsequent phase calls `osse_runner` and gets metrics JSON + plots automatically.

---

## Phase 2: Code Cleanup & Bug Fixes

### 2a: Remove dead masking methods
- [x] **Keep**: `masking_simple`, `masking_interpolate`, `masking_total_column_average_simple`, `masking_total_column_average_mult`
- [x] **Remove** 8 others (4 are identical to `_simple`, rest are experimental dead ends)
- [x] **File**: `flowmatching.py`

### 2b: Fix `compute_xco2` fallback
- [x] Add targshift correction to fallback path (`+ targshift_mean * h_ak_sum`)
- [x] Add `ak is not None` guard in fallback
- [x] **File**: `flowmatching.py`

### 2c: Fix guidance gradient
- [x] Current: `column_error / h_ak_sum` (uniform). Fix: `h_k * a_k * column_error` (Jacobian transpose)
- [x] **File**: `flowmatching.py`

**Deliverable**: Unit tests in `tests/test_forward_model.py` (17 tests, all passing). Roundtrip error < 1e-5.

---

## Phase 3: Toy OSSE Test Gate

Formalize `toy_column_osse.py` as a mandatory validation gate.

- [x] Make importable as test module with pytest markers (quick ~30s, full ~5min)
- [x] Every conditioning method must pass: column RMSE < threshold, no NaN, no divergence
- [x] Run on all existing methods to establish baseline
- [x] Run `osse_runner` on real CT data for baseline evaluation of current methods

**Deliverable**: Baseline metrics + plots for unconditional and all 4 existing conditioning modes.

---

## Phase 4: Vanilla FM Training Fixes

Ensure the unconditional model matches SOTA vanilla flow matching before adding conditioning.

### 4a: Audit training pipeline
- [x] Verify OT-CFM path (AffineProbPath + CondOTScheduler) is correctly implemented
  - `x_t = (1-t)*x_0 + t*x_1`, target `dx_t = x_1 - x_0` ✓
- [x] Check noise-data pairing: currently random → replaced with OT coupling in 4b
- [x] Verify loss function: plain MSE on velocity, no time-dependent weighting needed ✓
- [x] Check inference: fixed-step midpoint, `steps=11`, `step_size=0.2` in exp 01

### 4b: Minibatch OT coupling
- [x] Replace random noise-data pairing with OT-optimal pairing within minibatch
- [x] GPU-native Sinkhorn (pure PyTorch) instead of scipy — `compute_ot_coupling()` in `flowmatching.py`
- [x] **Modify**: `FlowMatching.training_forward()` — add OT pairing before `self.path.sample()`
- [x] Also added to `toy_column_osse.py` `train_flow_matching()` via `use_ot_coupling` param

### 4c: Inference improvements
- [x] Adaptive ODE solver support: `atol`/`rtol` params, `method` overridable via `generate_kwargs`
- [x] Time grid spacing: `_build_time_grid()` with uniform/cosine/front_loaded options
- [x] Ablation script in exp 09: steps (11/21/51), solver (midpoint/dopri5), timegrid variants

### 4d: Retrain + evaluate
- [x] New experiment: `09_fm_unet_ot_training/` with `use_ot_coupling=True`
- [x] `train.py` with `--max_steps` CLI arg for smoke testing
- [x] `run_eval.py` with ablation framework (ot/solver/steps/timegrid/all)
- [x] Tests: `tests/test_flowmatching_training.py` — OT coupling + time grid tests
- [x] Full training via Slurm: `sbatch 09_fm_unet_ot_training/train.slurm`
- [x] Evaluation: `sbatch 09_fm_unet_ot_training/run_eval.slurm`

**Deliverable**: Improved unconditional model. Training curves + sample quality comparison plots.

---

### Phase 4.5: Unconditional Flow Matching proper evaluation and tuning

The goal of this phase is to obtain a solid flow matching forward model. The problem is, the current evaluation is not robust enough to allow us to really assess which one of two generative models is better. What we want is an evaluation that properly checks the how well the generations are, i.e. evaluate their distribution.

- [x] Produce plots of the different marginals of the distributions in a reasonable way: we want multiple ground truth CO2 samples (so multiple time steps)... and then also multiple generated CO2 samples. For each of them we want to compute statistics (e.g. mean, std. dev., power spectrum etc.), and then compare the distributions of these statistics against each other.
- [x] We want to compute probabilistic scores comparing the two distributions with samples
- [x] We want plots that directly plot the spatial pattern (mean over samples) against each other, same for the lat-height pattern. Also compare the std. dev over samples for both.
- [x] During training of the flow matching model, make sure that the validation epoch spits out a meaningful metric that is associated with generation quality, and can be used to pick the best checkpoint afterwards
- [x] Check with literature for any other meaningful metrics & plots to assess the quality of the generative model
- [x] Where necessary, rewrite the current API / generalize it, such that this generative evaluation is more straight forward to support (but we still also want full support & ideally backwards compatibility for the forward transport models)
- [x] Tune a bunch of different training settings, especially optimization parameters like the learning rate (and others). Keep the batch size high to fill the GPU. Use SLURM for the tuning.
- [x] Also check with literature again for any tricks / bells & whistles to improve the flow matching training, and also tune these.
- [x] Also make a tuning experiment with inference-time parameters, i.e. those related to generation like the solver or the step size.
- [x] Create a comparison of the tuning experiments with bar plots & tables that show what changing each parameter / adding new features adds/changes in terms of generation quality.
- [x] Train a final model with the best tuning config for more epochs

**Deliverable**: A top notch unconditional model with plenty of visual evidence to support design choices & experiments.

---

## Phase 5: Proper DPS Guidance for Column XCO2

Fix the core guidance algorithm to use proper likelihood gradients.

### 5a: DPS likelihood gradient
For `p(y|x) ~ N(y; H(x), sigma_y^2 I)`, `H(x) = sum_k h_k a_k x_k`:
```
nabla_{x_k} log p(y|x) = (h_k * a_k / sigma_y^2) * (y - H(x))
```
- [x] Add `sigma_obs` parameter to `MaskedVelocityWrapper`
- [x] Key change: gradient proportional to `h_k * a_k` (surface-heavy), not uniform
- [x] **File**: `flowmatching.py`

### 5b: Gaussian spatial smoothing
- [x] `_gaussian_smooth_2d(field, sigma)` via `F.conv2d`, periodic longitude padding
- [x] `spatial_smoothing_sigma` parameter in `generate_kwargs`
- [x] **File**: `flowmatching.py`

### 5c: Toy OSSE gate + full OSSE via osse_runner

**Deliverable**: DPS ablation (guidance_scale x sigma_obs x smoothing x timing). Plots: RMSE vs scale, roughness, per-level guidance magnitude.

**Success**: Column RMSE < 75% of unconditional at obs locations.

---

## Phase 6: FlowDPS — Posterior Sampling via Projection

**Ref**: Kim et al., ICCV 2025

Tweedie estimate -> data projection -> re-noise. Projects clean estimate onto column constraint.

- [ ] **Create** `neural_transport/inference/posterior_samplers.py`
- [ ] `FlowDPSSampler`: `_tweedie_estimate`, `_project_column`, `_renoise`
- [ ] Column projection: `x_hat_k += (h_k a_k) * (y - H(x_hat)) / (sum(h_j a_j)^2 + sigma^2)`
- [ ] Integrate: `sampler="flowdps"` in `generate_kwargs`
- [ ] **File**: `flowmatching.py` `inference_forward` — dispatch to sampler

**Deliverable**: Toy OSSE gate + full OSSE. Trajectory visualization (Tweedie estimates at t=0.2,0.5,0.8). RMSE convergence vs steps.

**Success**: Lower RMSE_xco2_obs than best DPS configuration.

---

## Phase 7: Stochastic Posterior Sampling (SDE)

ODE = deterministic given noise. SDE = noise injection for better posterior exploration.

- [ ] `StochasticPosteriorSampler` in `posterior_samplers.py`
- [ ] SDE: `dx = v(x,t)dt + sigma(t)dW`, annealed schedule `sigma(t) = sigma_max(1-t)`
- [ ] Predictor-Corrector: flow step + Langevin MCMC corrector targeting `p(x_t|y)`
- [ ] Combine with FlowDPS projection

**Deliverable**: Toy OSSE gate + ablation (noise schedule, corrector steps). Ensemble spread comparison: ODE vs SDE.

**Success**: Larger spread while maintaining RMSE. Spread-skill ratio closer to 1.0.

---

## Phase 8: FIG — Flow with Interpolant Guidance

**Ref**: Ricci et al.

Measurement interpolants for theoretically-justified guidance. Modifies velocity directly (no re-noising).

- [ ] `FIGSampler` in `posterior_samplers.py`
- [ ] Unconditional step + conditional correction via measurement interpolant
- [ ] Toy OSSE gate + full OSSE

**Deliverable**: FIG vs FlowDPS vs DPS vs SDE comparison. Correction magnitude over time.

---

## Phase 9: ICTM — Iterative Corrupted Trajectory Matching

**Ref**: arXiv 2405.18816

Tweedie + local MAP. For linear column obs: closed-form (= FlowDPS projection). For nonlinear: inner gradient descent.

- [ ] `ICTMSampler` in `posterior_samplers.py`
- [ ] Toy OSSE gate + full OSSE

**Deliverable**: Quality vs cost comparison with FlowDPS.

---

## Phase 10: Conjugate Integrators — Few-Step Conditioning

**Ref**: arXiv 2405.17673

Plug-and-play wrapper. Target: 5-step conditional generation.

- [ ] `ConjugateIntegratorSampler` in `posterior_samplers.py`
- [ ] Toy OSSE gate + efficiency benchmark

**Deliverable**: Pareto front: wall-time vs RMSE across all methods.

---

## Phase 11: Advanced FM Training (W-CFM, OAT-FM)

*After training-free methods are benchmarked.*

### 11a: Weighted CFM (W-CFM)
- [ ] Gibbs kernel weighting of training pairs: `w_ij = exp(-||x_i - y_j||^2 / 2eps)`
- [ ] Approximates entropic OT, straighter paths
- [ ] **Modify**: `training_forward()` in `flowmatching.py`

### 11b: Time-dependent loss weighting
- [ ] `w(t) = 1/sigma(t)^2` or SNR-based weighting
- [ ] Velocity vs noise prediction parameterization

### 11c: OAT-FM fine-tuning (optional)
- [ ] Phase 2 fine-tune: minimize acceleration in sample-velocity space
- [ ] Straighter paths, fewer NFE needed

### 11d: Retrain + re-evaluate all posterior methods

**Deliverable**: Training curves. Improved unconditional RMSE. Re-evaluation of all Phases 5-10.

---

## Phase 12: Conditional Flow Matching (Retraining)

**Ref**: Lipman et al. 2023

Train `u(x,t|y)` directly. Most principled approach.

### 12a: Architecture
- [ ] +2 UNet input channels: obs_channel + mask_channel
- [ ] Training: synthetic XCO2 from target x_1, random mask patterns, noise on obs
- [ ] Mix conditional (80%) + unconditional (20%) for classifier-free guidance

### 12b: Training
- [ ] Fine-tune from Phase 4/11 checkpoint
- [ ] 50k conditional + 50k mixed steps

### 12c: Classifier-Free Guidance
- [ ] `v_guided = (1+w) * v(x,t|y) - w * v(x,t|empty)`

### 12d: Evaluate via osse_runner

**Success**: RMSE < 3.0 ppm on column OSSE.

---

## Phase 13: Grand Comparison

Aggregate all phase results into publication-quality comparison.

- [ ] Collect OSSEResults from Phases 5-12
- [ ] Statistical significance tests (paired t-test on per-timestep metrics)
- [ ] **Plots**: methods x patterns heatmap, Pareto front (time vs RMSE), visual comparison grid, calibration diagrams, rank histograms, zonal mean error, per-level RMSE
- [ ] **Experiment**: `carbonbench/.../11_grand_comparison/`

---

## Phase 14: Real OCO-2 Application

Apply best 2-3 methods to real satellite data.

- [ ] Real OCO-2 via `iterative_generate_oco2()`: per-sounding AK, retrieval uncertainty
- [ ] Validate vs CarbonTracker posterior, TCCON, ObsPack surface flasks
- [ ] **Experiment**: `carbonbench/.../12_real_oco2_posterior/`
- [ ] **Plots**: posterior global maps + spread, station time series, XCO2 vs TCCON scatter

---

## Phase 15: Multi-step Temporal Conditioning

### 15a: Sequential (autoregressive)
- [ ] Generate step-by-step, carry posterior mean forward

### 15b: Temporal consistency metrics
- [ ] Autocorrelation, mass conservation

### 15c: Sliding-window 4D-Var style (stretch goal)

---

## Phase 16: Advanced Ideas (Brainstorm)

- [ ] **Physics-informed guidance**: torchtransport as physics constraint
- [ ] **Latent-space FM**: encoder/decoder + compressed flow matching
- [ ] **Score distillation**: 1-step generator via consistency models
- [ ] **Ensemble Kalman Flow**: EnKF analysis + flow particles
- [ ] **Amortized posterior**: direct (obs, mask) -> posterior network
- [ ] **Multi-resolution**: coarse-to-fine conditioning
- [ ] **SWAG UQ**: Bayesian velocity model uncertainty (per turbulence paper)

---

## Phase 17: Publication (ACP/GMD)

1. Introduction: CO2 inverse modeling, generative approaches
2. Background: flow matching, OCO-2, CarbonTracker
3. Method: FM for CO2, XCO2 forward model, posterior sampling methods
4. OSSE: benchmark, method comparison
5. Real Data: OCO-2 posterior, TCCON/ObsPack validation
6. Discussion: methods, physics, cost, vs 4D-Var
7. Conclusion

**Key figures**: pipeline schematic, OSSE table, visual grid, Pareto front, OCO-2 maps, TCCON scatter, calibration diagrams.

---

## Dependency Graph

```
Phase 1 (eval infra) ──> Phase 2 (cleanup) ──> Phase 3 (toy OSSE baseline)
                                                    │
                                    ┌───────────────┤
                                    v               v
                              Phase 4 (vanilla FM)  Phase 5 (DPS guidance)
                                    │               │
                                    v               ├──> Phase 6 (FlowDPS)
                              Phase 11 (adv. FM)    ├──> Phase 7 (SDE)
                                    │               ├──> Phase 8 (FIG)
                                    v               ├──> Phase 9 (ICTM)
                              Phase 12 (CFM)        └──> Phase 10 (CCI)
                                    │                       │
                                    └───────┬───────────────┘
                                            v
                                      Phase 13 (comparison)
                                            │
                                    ┌───────┴───────┐
                                    v               v
                              Phase 14 (OCO-2) Phase 15 (temporal)
                                    │               │
                                    └───────┬───────┘
                                            v
                                      Phase 17 (paper)

Phase 4 and Phases 5-10 can proceed in parallel after Phase 3.
Phase 11-12 are deferred until training-free methods benchmarked.
Each phase self-evaluates via osse_runner.
```

---

## Verification Checklist (every phase)

- [ ] **Toy OSSE gate**: passes column XCO2 constraint, no NaN, no divergence
- [ ] **osse_runner**: metrics JSON + comparison plots auto-produced
- [ ] Column constraint: RMSE_xco2_obs < unconditional
- [ ] Spatial quality: roughness comparable to unconditional
- [ ] Ensemble diversity: spread-skill ratio in [0.5, 2.0]
