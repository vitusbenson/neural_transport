# Flow Matching Steps

## ✅ Completed
01. Load batch properly (only `co2massmix` first)
02. Get `training_forward` running (add `offset`/`scale`, `dt_alpha`/`dt_sigma`)
03. Get `inference_forward` runnning
04. Implement time embedding in UNet **(a)**
05. Integrate into `neural_transport`
06. Create SLURM script
07. Test a training run
08. Adapt evaluation (plotting, iterative_generate, score, loss) **(b)**
09. Create mask for observations

## 🔄 In Progress
10. Run conditional generation
11. Implement CRPS
12. Test different hyperparameters **(c)**
13. Load OCO-2 dataset into `neural_transport`
14. Integrate OCO-2 data into `carbonbench`
15. Add possibility of covariats and $(CO_2)_{t-1}$
16. Extend FlowMatching to use $X_0 = noise + weight \cdot \text{OCO-2}$
17. Handle $(\text{OCO-2})_t$
18. Attack with Ruff

**(a)**: It is just stacked as another channel:
`x_in = torch.cat(list(batch_normalized.values()), dim=-1)`

**(b)**:
- `iterative_generate()` to `iterative_forecast()` in `predict()` added
- `compute_metric_over_samples` added in `plot_results.py`
- `compute_score_df_generate` added in `analyse.py`

**(c)**:
- `MODEL_SIZE`: ["S", "M", "L"]<br>
        - `enc_filters`, `dec_filters`
- `lr`: [1e−4,3e−4,1e−3,3e−3] or `lr_find`<br>
        - `weight_decay`: [0, 0.01, 0.1] <br>
        - `warmup_steps`, `halfcosine_steps`, `min_lr`, `max_lr`
- `step_size`: [0.05, 0.1, 0.2]
- `method`: ["midpoint", "euler"]
- `BATCH_SIZE_TRAIN`
