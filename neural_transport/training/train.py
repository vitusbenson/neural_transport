import logging
import multiprocessing
import os
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import xarray as xr

logger = logging.getLogger(__name__)

from neural_transport.datamodule import CarbonDataModule, CarbonDataset, PreBatchedShuffleCallback
from neural_transport.inference.analyse import (
    compute_distributional_score_df,
    compute_local_scores,
    compute_score_df,
    compute_score_df_generate,
)
from neural_transport.inference.forecast import iterative_forecast
from neural_transport.inference.generation import (
    generate_for_distributional_eval,
    iterative_generate,
    iterative_generate_oco2,
)
from neural_transport.litmodule import NeuralTransport
from neural_transport.plots.plot_results import (
    animate_predictions,
    plot_metrics,
    plot_obspack_stations,
    plot_samples,
)
from neural_transport.tools.conversion import massmix_to_molemix


def load_model(exp_dir, ckpt="best", device="cuda", ema=False):
    """Load a trained NeuralTransport model from an experiment directory.

    Args:
        exp_dir: Path to experiment directory (e.g., 11_fm_unet_final/).
            Expects checkpoints at {exp_dir}/singlestep/checkpoints/.
        ckpt: "best" or "last".
        device: Device to load model onto.
        ema: If True and the checkpoint contains an ``ema_state_dict`` key
            (saved by EMACallback), swap those weights into the model. Standard
            for FM/diffusion inference.

    Returns:
        NeuralTransport LightningModule in eval mode.
    """
    exp_dir = Path(exp_dir)
    ckptdir = exp_dir / "singlestep" / "checkpoints"

    if ckpt == "best":
        # Find checkpoint with lowest val loss. Filter out smoke-test artifacts:
        # ckpts with Step=<smoke_threshold> typically have LossVal=0.0 from a
        # sanity check that ran before any training updates and would otherwise
        # be selected as "best" — silently breaking eval.
        SMOKE_STEP_MAX = 100

        def _is_smoke(p):
            try:
                step = int(p.name.split("Step=")[1].split("-")[0])
            except (IndexError, ValueError):
                return False
            try:
                lv = float(p.name.split("LossVal=")[1].split(".c")[0].split("-v")[0])
            except (IndexError, ValueError):
                return False
            return step <= SMOKE_STEP_MAX and lv == 0.0

        val_ckpts = [p for p in ckptdir.glob("*.ckpt") if "LossVal" in p.name and not _is_smoke(p)]
        if not val_ckpts:
            # Fallback to best.ckpt symlink
            ckptpath = ckptdir / "best.ckpt"
        else:
            ckptpath = sorted(
                val_ckpts,
                key=lambda p: float(p.name.split("=")[-1].split(".c")[0].split("-v")[0]),
            )[0]
    else:
        ckptpath = ckptdir / "last.ckpt"

    if not ckptpath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckptpath}")

    logger.info("Loading model from %s", ckptpath)
    model = NeuralTransport.load_from_checkpoint(str(ckptpath), map_location=device)
    if ema:
        ckpt_blob = torch.load(str(ckptpath), map_location=device, weights_only=False)
        ema_sd = ckpt_blob.get("ema_state_dict")
        if ema_sd is None:
            logger.warning("ema=True requested but no ema_state_dict in %s — using regular weights.", ckptpath)
        else:
            model.load_state_dict(ema_sd)
            logger.info("Loaded EMA weights from %s", ckptpath)
    model = model.to(device)
    model.eval()
    return model


def train_singlestep(
    run_dir,
    data_kwargs,
    lit_module_kwargs,
    trainer_kwargs,
    ckptpath=None,
    ckpt_kwargs=dict(
        save_top_k=1,  # -1,
        save_last=True,
        monitor="Loss/Val_rollout",
        filename="Epoch={epoch}-Step={step}-LossVal={Loss/Val_rollout:.6f}",
        auto_insert_metric_name=False,
        every_n_epochs=1,
    ),
    extra_callbacks=None,
):
    run_dir = Path(run_dir)

    tb_logger = pl.loggers.tensorboard.TensorBoardLogger(run_dir, name="", version="singlestep")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**ckpt_kwargs)
    checkpoint_callback.CHECKPOINT_NAME_LAST = "best"

    ckpt_kwargs["monitor"] = "step"
    ckpt_kwargs["mode"] = "max"
    ckpt_kwargs["filename"] = "latest-" + ckpt_kwargs["filename"]

    latest_checkpoint_callback = pl.callbacks.ModelCheckpoint(**ckpt_kwargs)

    lr_monitor = pl.callbacks.LearningRateMonitor()

    dset = CarbonDataModule(**data_kwargs)

    if ckptpath is not None:
        lit_module_kwargs["pretrained_ckptpath"] = ckptpath
    model = NeuralTransport(**lit_module_kwargs)

    callbacks = [checkpoint_callback, latest_checkpoint_callback, lr_monitor]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    if data_kwargs.get("compute", False):
        callbacks.append(PreBatchedShuffleCallback())

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=tb_logger,
        **trainer_kwargs,
    )

    logger.info("Starting singlestep training")
    trainer.fit(model, dset)

    return model.global_rank


def train_rollout(
    run_dir,
    data_kwargs,
    lit_module_kwargs,
    rollout_trainer_kwargs,
    rollout_constant_lr=1e-5,
    timesteps=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
    ckpt="last",
):
    run_dir = Path(run_dir)

    if rollout_constant_lr:
        lit_module_kwargs["lr"] = rollout_constant_lr
        lit_module_kwargs["lr_shedule_kwargs"] = dict(warmup_steps=1, halfcosine_steps=100000, min_lr=1, max_lr=1)

    log_path = run_dir / "singlestep"
    ckptdir = log_path / "checkpoints"
    bestckptpath = sorted(
        [p for p in ckptdir.glob("*.ckpt") if "LossVal" in p.name],
        key=lambda p: float(p.name.split("=")[-1].split(".c")[0].split("-v")[0]),
    )[0]

    lastckptpath = log_path / "checkpoints/last.ckpt"

    ckptpath = bestckptpath if ckpt == "best" else lastckptpath

    model = NeuralTransport.load_from_checkpoint(
        ckptpath,
        map_location="cpu",
        **lit_module_kwargs,
    )

    logger = pl.loggers.tensorboard.TensorBoardLogger(run_dir, name="", version="rollout")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        save_last=True,
        monitor="Loss/Val_rollout",
        filename="Epoch={epoch}-Step={step}-LossVal={Loss/Val_rollout:.6f}",
        auto_insert_metric_name=False,
        every_n_epochs=1,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor()

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        **rollout_trainer_kwargs,
    )
    BATCH_SIZE_TRAIN = data_kwargs["batch_size_train"]
    BATCH_SIZE_PRED = data_kwargs["batch_size_pred"]

    for n_timesteps in timesteps:
        data_kwargs["n_timesteps"] = n_timesteps
        data_kwargs["batch_size_train"] = max(BATCH_SIZE_TRAIN // (n_timesteps + 1), 1)
        data_kwargs["batch_size_pred"] = max(BATCH_SIZE_PRED // (n_timesteps + 1), 1)

        dset = CarbonDataModule(**data_kwargs)

        logger.info("Starting training %d timesteps", n_timesteps)
        trainer.fit(model, dset)
        trainer.fit_loop.max_epochs += rollout_trainer_kwargs["max_epochs"]
        trainer.fit_loop.epoch_loop.val_loop._results.clear()

    return model.global_rank


def load_dataset(data_path, data_kwargs, load_obspack=True):
    dataset = CarbonDataset(
        data_path=data_path,
        dataset=data_kwargs["dataset"],
        grid=data_kwargs["grid"],
        vertical_levels=data_kwargs["vertical_levels"],
        freq=data_kwargs["freq"],
        n_timesteps=1,
        target_vars=data_kwargs["target_vars"],
        forcing_vars=data_kwargs["forcing_vars"],
        load_obspack=load_obspack,
        new_zarr=True,
    )

    return dataset


def predict(
    log_path,
    data_path_forecast,
    data_kwargs,
    device="cuda",
    freq="QS",
    ckpt="last",
    lit_module_kwargs={},
    massfixer="default",
    zero_surfflux=False,
    save_obs=False,
    generate_kwargs={},
    distributional_eval=False,
    distributional_eval_kwargs=None,
):
    log_path = Path(log_path)

    ckptdir = log_path / "checkpoints"
    bestckptpath = sorted(
        [p for p in ckptdir.glob("*.ckpt") if "LossVal" in p.name],
        key=lambda p: float(p.name.split("=")[-1].split(".c")[0].split("-v")[0]),
    )[0]

    lastckptpath = log_path / "checkpoints/last.ckpt"

    ckptpath = bestckptpath if ckpt == "best" else lastckptpath

    if massfixer != "default":
        lit_module_kwargs["model_kwargs"]["massfixer"] = massfixer
    outpath = log_path / "preds" / f"ckpt={ckpt}_massfixer={massfixer}"

    is_fm = type(lit_module_kwargs['model']).__name__ == "FlowMatching" or lit_module_kwargs['model'] == "flowmatching"
    dataset = load_dataset(data_path_forecast, data_kwargs, load_obspack=not is_fm)

    if is_fm:
        logger.info("Generating %s %s CKPT", ckptpath, ckpt)
        model = NeuralTransport.load_from_checkpoint(ckptpath, **lit_module_kwargs)
        outpath.mkdir(parents=True, exist_ok=True)

        if distributional_eval:
            dist_kwargs = distributional_eval_kwargs or {}
            dist_outpath = outpath / "distributional_eval"
            logger.info("Running distributional evaluation...")
            generate_for_distributional_eval(
                model,
                dataset,
                dist_outpath,
                device=device,
                target_vars_3d=data_kwargs["target_vars"],
                generate_kwargs=generate_kwargs,
                **dist_kwargs,
            )
            return

        if generate_kwargs.get("mask_source") == "oco2":
            dataset_gen = load_dataset(generate_kwargs["data_path_generate"], generate_kwargs["generate_data_kwargs"])
            iterative_generate_oco2(
                model,
                dataset,
                dataset_gen,
                outpath,
                rollout=(freq != "singlestep"),
                device=device,
                verbose=True,
                freq=freq,
                zero_surfflux=zero_surfflux,
                remap=("latlon" not in data_kwargs["grid"]),
                target_vars_3d=data_kwargs["target_vars"],
                target_vars_2d=generate_kwargs["generate_data_kwargs"]["target_vars"],
                save_obs=save_obs,
                **generate_kwargs,
            )
        else:
            iterative_generate(
                model,
                dataset,
                outpath,
                rollout=(freq != "singlestep"),
                device=device,
                verbose=True,
                freq=freq,
                zero_surfflux=zero_surfflux,
                remap=("latlon" not in data_kwargs["grid"]),
                target_vars_3d=data_kwargs["target_vars"],
                target_vars_2d=[],
                save_obs=save_obs,
                **generate_kwargs,
            )
    else:
        logger.info("Forecasting %s %s CKPT", ckptpath, ckpt)
        model = NeuralTransport.load_from_checkpoint(ckptpath, **lit_module_kwargs)
        outpath.mkdir(parents=True, exist_ok=True)
        iterative_forecast(
            model,
            dataset,
            outpath,
            rollout=(freq != "singlestep"),
            device=device,
            verbose=True,
            freq=freq,
            remap=("latlon" not in data_kwargs["grid"]),
            target_vars_3d=data_kwargs["target_vars"],
            target_vars_2d=[],
            zero_surfflux=zero_surfflux,
        )


def load_pred_targ(target_path, pred_path):
    targ = xr.open_zarr(target_path)
    co2targ = xr.merge(
        [
            targ["variables_3d"].to_dataset("vari_3d"),
            targ["variables_2d"].to_dataset("vari_2d"),
        ]
    )

    co2pred = xr.open_zarr(pred_path)
    return co2targ, co2pred


def score(
    target_path: str,
    pred_path: str,
    obs_pred_path: str | None = None,
    freq: str | None = "QS",
    target_var: str | None = "co2massmix",
    generate_kwargs: dict | None = None,
    distributional_eval: bool = False,
    distributional_eval_path: str | None = None,
) -> None:
    # Distributional evaluation path
    if distributional_eval:
        import xarray as xr

        dist_dir = (
            Path(distributional_eval_path) if distributional_eval_path else Path(pred_path) / "distributional_eval"
        )
        if dist_dir.exists():
            gt_ds = xr.open_dataset(dist_dir / "gt_pool.nc")
            gen_ds = xr.open_dataset(dist_dir / "gen_pool.nc")
            df = compute_distributional_score_df(gt_ds, gen_ds, target_var=target_var)
            score_path = dist_dir / "scores"
            score_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(score_path / "distributional_metrics.csv", index=False)
            logger.info("Distributional metrics saved to %s", score_path / "distributional_metrics.csv")
        else:
            logger.warning("Distributional eval dir not found: %s", dist_dir)
        return

    co2targ, co2pred = load_pred_targ(target_path, pred_path)
    co2pred = co2pred.isel(time=slice(1, None))
    co2targ = co2targ.isel(time=slice(1, None)).isel(time=slice(len(co2pred.time)))

    model_name = pred_path.parent.parent.parent.parent.name
    singlestep_or_rollout = pred_path.parent.parent.parent.name
    ckpt_name = pred_path.parent.name
    score_path = pred_path.parent.parent.parent / "scores" / ckpt_name

    metrics = {}
    if generate_kwargs:
        df_full, df_global_scalars, maps = compute_score_df_generate(
            co2targ, co2pred, target_var=target_var, **generate_kwargs
        )
        df_full.index = pd.MultiIndex.from_product(
            [[f"{model_name}_{singlestep_or_rollout}_{ckpt_name}"], df_full.index],
            names=["model", "sample"],
        )
        score_path.mkdir(parents=True, exist_ok=True)
        df_full.to_csv(score_path / ("metrics_fullsamples.csv" if freq == "QS" else f"metrics_fullsamples_{freq}.csv"))

        idx = pd.IndexSlice
        df_summary = df_full.loc[idx[f"{model_name}_{singlestep_or_rollout}_{ckpt_name}", ["mean", "std"]], :]
        df_summary.to_csv(score_path / ("metrics.csv" if freq == "QS" else f"metrics_{freq}.csv"))
        df_global_scalars.to_csv(
            score_path / ("metrics_global_scalars.csv" if freq == "QS" else f"metrics_global_scalars_{freq}.csv")
        )
        maps.to_netcdf(score_path / ("metrics_maps.nc" if freq == "QS" else f"metrics_maps_{freq}.nc"))
    else:
        metrics[f"{model_name}_{singlestep_or_rollout}_{ckpt_name}"] = compute_score_df(
            co2targ,
            co2pred,
            freq=freq,
        )
        df = pd.DataFrame(metrics).T
        score_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(score_path / ("metrics.csv" if freq == "QS" else f"metrics_{freq}.csv"))

    if obs_pred_path:
        obspreds = xr.open_zarr(obs_pred_path)  # .isel(time = slice(1,None))

        logger.info("Computing score for %s", obs_pred_path)

        df = compute_local_scores(obs_preds=obspreds, freq=freq)

        df.to_csv(score_path / ("obs_metrics.csv" if freq == "QS" else f"obs_metrics_{freq}.csv"))


def plot(
    target_path,
    pred_path,
    obs_pred_path=None,
    obs_compare_path=None,
    freq="QS",
    data_path_forecast=None,
    data_kwargs=None,
    movie_interval=["2018-01-01", "2018-03-31"],
    num_workers=multiprocessing.cpu_count() // 2,
    plot_types=["metrics", "animations", "obspack"],
    generate_kwargs=None,
):
    if generate_kwargs is None:
        generate_kwargs = {}

    # For distributional-only plots, skip loading pred/target zarr
    needs_pred_targ = any(pt in plot_types for pt in ["metrics", "animations", "obspack", "samples"])

    ckpt_name = pred_path.parent.name
    plot_path = pred_path.parent.parent.parent / "plots" / ckpt_name
    score_path = pred_path.parent.parent.parent / "scores" / ckpt_name
    plot_path.mkdir(parents=True, exist_ok=True)

    if needs_pred_targ:
        co2targ, co2pred = load_pred_targ(target_path, pred_path)
        co2pred = co2pred.isel(time=slice(1, None))
        co2targ = co2targ.isel(time=slice(1, None))
        co2targ = co2targ.isel(time=slice(None, len(co2pred.time)))
        co2massmix = data_kwargs['target_vars'][0]
        co2pred["co2molemix"] = massmix_to_molemix(co2pred[co2massmix])
        co2targ["co2molemix"] = massmix_to_molemix(co2targ[co2massmix])
    else:
        co2targ = co2pred = None
    if "metrics" in plot_types:
        plot_metrics(co2pred, co2targ, plot_path, imgformats=["pdf"])

    if "samples" in plot_types:
        plot_samples(
            preds=co2pred,
            out_dir=plot_path,
            score_path=score_path,
            tests=co2targ,
            freq=freq,
            varnames=["co2molemix"],
            normalize=False,
            imgformats=["png"],
            **generate_kwargs,
        )

    t0, tend = movie_interval

    if "animations" in plot_types:
        animate_predictions(
            co2pred.sel(time=slice(t0, tend)),
            co2targ.sel(time=slice(t0, tend)),
            plot_path,
            postfix=f"3d_anim_t0={t0}-tend={tend}",
            num_workers=num_workers,
        )

    if "distributional" in plot_types:
        dist_dir = pred_path.parent / "distributional_eval"
        if dist_dir.exists():
            from neural_transport.plots.distributional_plots import (
                plot_distributional_metrics_summary,
                plot_lat_height_comparison,
                plot_marginal_distributions,
                plot_power_spectrum_comparison,
                plot_qq,
                plot_sample_grid,
                plot_spatial_pattern_comparison,
            )

            gt_ds = xr.open_dataset(dist_dir / "gt_pool.nc")
            gen_ds = xr.open_dataset(dist_dir / "gen_pool.nc")
            target_var = data_kwargs['target_vars'][0]
            gt_fields = gt_ds[target_var].values
            gen_fields = gen_ds[target_var].values
            lat_vals = gt_ds.lat.values
            lon_vals = gt_ds.lon.values
            level_values = gt_ds.level.values if "level" in gt_ds.dims else None

            dist_plot_dir = plot_path / "distributional"

            # Select representative levels closest to [1013, 843, 441, 73] hPa
            import numpy as np

            nlev = gt_fields.shape[-1]
            if level_values is not None:
                target_pressures = [1013, 843, 441, 73]
                level_indices = []
                for tp in target_pressures:
                    idx = int(np.argmin(np.abs(np.asarray(level_values) - tp)))
                    if idx not in level_indices:
                        level_indices.append(idx)
            else:
                level_indices = list(range(min(nlev, 4)))

            # Per-level plots
            for lev_idx in level_indices:
                lev_hpa = int(level_values[lev_idx]) if level_values is not None else None
                plot_spatial_pattern_comparison(
                    gt_fields,
                    gen_fields,
                    lat_vals,
                    lon_vals,
                    dist_plot_dir,
                    level_idx=lev_idx,
                    level_hpa=lev_hpa,
                    imgformats=["png"],
                )
                plot_marginal_distributions(
                    gt_fields,
                    gen_fields,
                    dist_plot_dir,
                    level_idx=lev_idx,
                    level_hpa=lev_hpa,
                    imgformats=["png"],
                )
                plot_power_spectrum_comparison(
                    gt_fields,
                    gen_fields,
                    lat_vals,
                    lon_vals,
                    dist_plot_dir,
                    level_idx=lev_idx,
                    level_hpa=lev_hpa,
                    imgformats=["png"],
                )
                plot_qq(gt_fields, gen_fields, dist_plot_dir, level_idx=lev_idx, level_hpa=lev_hpa, imgformats=["png"])

            # Total column plots (level_idx=None)
            plot_spatial_pattern_comparison(
                gt_fields,
                gen_fields,
                lat_vals,
                lon_vals,
                dist_plot_dir,
                level_idx=None,
                level_values=level_values,
                imgformats=["png"],
            )
            plot_marginal_distributions(
                gt_fields,
                gen_fields,
                dist_plot_dir,
                level_idx=None,
                level_values=level_values,
                imgformats=["png"],
            )
            plot_power_spectrum_comparison(
                gt_fields,
                gen_fields,
                lat_vals,
                lon_vals,
                dist_plot_dir,
                level_idx=None,
                level_values=level_values,
                imgformats=["png"],
            )
            plot_qq(gt_fields, gen_fields, dist_plot_dir, level_idx=None, level_values=level_values, imgformats=["png"])

            # Lat-height cross-section
            plot_lat_height_comparison(
                gt_fields,
                gen_fields,
                lat_vals,
                dist_plot_dir,
                level_values=level_values,
                imgformats=["png"],
            )

            # Sample grid
            plot_sample_grid(
                gt_fields,
                gen_fields,
                lat_vals,
                lon_vals,
                dist_plot_dir,
                level_values=level_values,
                imgformats=["png"],
            )

            # Distributional metrics bar chart
            metrics_csv = dist_dir / "scores" / "distributional_metrics.csv"
            if metrics_csv.exists():
                import pandas as pd

                metrics_df = pd.read_csv(metrics_csv)
                metrics = dict(zip(metrics_df.columns, metrics_df.iloc[0]))
                plot_distributional_metrics_summary(
                    metrics,
                    dist_plot_dir,
                    level_values=level_values,
                    imgformats=["png"],
                )
            else:
                from neural_transport.inference.distributional_metrics import (
                    compute_distributional_metrics,
                )

                metrics = compute_distributional_metrics(
                    gt_fields,
                    gen_fields,
                    lat_vals,
                    lon_vals,
                )
                plot_distributional_metrics_summary(
                    metrics,
                    dist_plot_dir,
                    level_values=level_values,
                    imgformats=["png"],
                )

    if obs_pred_path and ("obspack" in plot_types):
        obspreds = xr.open_zarr(obs_pred_path)

        dataset = load_dataset(data_path_forecast, data_kwargs)

        carboscope_obspred = xr.open_zarr(obs_compare_path) if obs_compare_path else None

        plot_obspack_stations(
            obspreds,
            dataset.obspack_metadata,
            plot_path,
            compare_obs=carboscope_obspred,
            stations="all",
            imgformats=["pdf"],
        )


def train_and_eval_singlestep(
    run_dir,
    data_kwargs,
    lit_module_kwargs,
    trainer_kwargs,
    data_path_forecast,
    device="cuda",
    freq="QS",
    obs_compare_path=None,
    movie_interval=["2018-01-01", "2018-03-31"],
    num_workers=multiprocessing.cpu_count() // 2,
    train=True,
    ckpt="last",
    massfixer="default",
    pretrained_ckptpath=None,
    ckpt_kwargs=dict(
        save_top_k=1,  # -1,
        save_last=True,
        monitor="Loss/Val_rollout",
        filename="Epoch={epoch}-Step={step}-LossVal={Loss/Val_rollout:.6f}",
        auto_insert_metric_name=False,
        every_n_epochs=1,
    ),
    plot_types=["metrics", "animations", "obspack"],
    generate_kwargs=None,
    distributional_eval=False,
    distributional_eval_kwargs=None,
    extra_callbacks=None,
):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

    if generate_kwargs is None:
        generate_kwargs = {}

    if train:
        train_singlestep(
            run_dir,
            data_kwargs,
            lit_module_kwargs,
            trainer_kwargs,
            ckptpath=pretrained_ckptpath,
            ckpt_kwargs=ckpt_kwargs,
            extra_callbacks=extra_callbacks,
        )

    if ("SLURM_PROCID" in os.environ) and (int(os.environ["SLURM_PROCID"]) > 0):
        return

    predict(
        run_dir / "singlestep",
        data_path_forecast,
        data_kwargs,
        device=device,
        freq=freq,
        ckpt=ckpt,
        lit_module_kwargs=lit_module_kwargs,
        massfixer=massfixer,
        generate_kwargs=generate_kwargs,
        distributional_eval=distributional_eval,
        distributional_eval_kwargs=distributional_eval_kwargs,
    )
    target_path = (
        data_path_forecast
        / f"{data_kwargs['dataset']}_{data_kwargs['grid']}_{data_kwargs['vertical_levels']}_{data_kwargs['freq']}.zarr"
    )
    pred_path = (
        run_dir / "singlestep" / "preds" / f"ckpt={ckpt}_massfixer={massfixer}" / f"co2_pred_rollout_{freq}.zarr"
    )
    obs_pred_path = (
        run_dir / "singlestep" / "preds" / f"ckpt={ckpt}_massfixer={massfixer}" / f"obs_co2_pred_rollout_{freq}.zarr"
    )
    is_fm = type(lit_module_kwargs['model']).__name__ == "FlowMatching" or lit_module_kwargs['model'] == "flowmatching"
    if is_fm:
        obs_pred_path = None
        plot_types += ["samples"]

    if distributional_eval and is_fm:
        # Distributional-only evaluation: run score/plot on generated pools
        dist_outpath = run_dir / "singlestep" / "preds" / f"ckpt={ckpt}_massfixer={massfixer}" / "distributional_eval"
        target_var = data_kwargs['target_vars'][0]
        score(
            target_path,
            pred_path,
            obs_pred_path=obs_pred_path,
            freq=freq,
            target_var=target_var,
            generate_kwargs=generate_kwargs,
            distributional_eval=True,
            distributional_eval_path=dist_outpath,
        )
        plot(
            target_path,
            pred_path,
            obs_pred_path=obs_pred_path,
            obs_compare_path=obs_compare_path,
            freq=freq,
            data_path_forecast=data_path_forecast,
            data_kwargs=data_kwargs,
            movie_interval=movie_interval,
            num_workers=num_workers,
            plot_types=["distributional"],
            generate_kwargs=generate_kwargs,
        )
    else:
        target_var = data_kwargs['target_vars'][0]
        score(
            target_path,
            pred_path,
            obs_pred_path=obs_pred_path,
            freq=freq,
            target_var=target_var,
            generate_kwargs=generate_kwargs,
        )
        plot(
            target_path,
            pred_path,
            obs_pred_path=obs_pred_path,
            obs_compare_path=obs_compare_path,
            freq=freq,
            data_path_forecast=data_path_forecast,
            data_kwargs=data_kwargs,
            movie_interval=movie_interval,
            num_workers=num_workers,
            plot_types=plot_types,
            generate_kwargs=generate_kwargs,
        )


def train_and_eval_rollout(
    run_dir,
    data_kwargs,
    lit_module_kwargs,
    rollout_trainer_kwargs,
    data_path_forecast,
    device="cuda",
    freq="QS",
    obs_compare_path=None,
    movie_interval=["2018-01-01", "2018-03-31"],
    num_workers=multiprocessing.cpu_count() // 2,
    rollout_constant_lr=1e-5,
    timesteps=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
    train=True,
    ckpt="last",
    massfixers=["default"],
    run_scoring=True,
    run_plotting=True,
    run_forecast=True,
    plot_types=["metrics", "animations", "obspack"],
    zero_surfflux=False,
    generate_kwargs=None,
):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

    if generate_kwargs is None:
        generate_kwargs = {}

    if train:
        train_rollout(
            run_dir,
            data_kwargs,
            lit_module_kwargs,
            rollout_trainer_kwargs,
            rollout_constant_lr=rollout_constant_lr,
            timesteps=timesteps,
            ckpt=ckpt,
        )

    if ("SLURM_PROCID" in os.environ) and (int(os.environ["SLURM_PROCID"]) > 0):
        return

    for massfixer in massfixers:
        if run_forecast:
            predict(
                run_dir / "rollout",
                data_path_forecast,
                data_kwargs,
                device=device,
                freq=freq,
                ckpt=ckpt,
                lit_module_kwargs=lit_module_kwargs,
                massfixer=massfixer,
                zero_surfflux=zero_surfflux,
                generate_kwargs=generate_kwargs,
            )
        target_path = (
            data_path_forecast
            / f"{data_kwargs['dataset']}_{data_kwargs['grid']}_{data_kwargs['vertical_levels']}_{data_kwargs['freq']}.zarr"
        )

        pred_path = (
            run_dir
            / "rollout"
            / "preds"
            / f"ckpt={ckpt}_massfixer={massfixer}"
            / (f"co2_pred_rollout_{freq}.zarr" if not zero_surfflux else f"co2_pred_zeroflux_rollout_{freq}.zarr")
        )
        obs_pred_path = (
            run_dir
            / "rollout"
            / "preds"
            / f"ckpt={ckpt}_massfixer={massfixer}"
            / (
                f"obs_co2_pred_rollout_{freq}.zarr"
                if not zero_surfflux
                else f"obs_co2_pred_zeroflux_rollout_{freq}.zarr"
            )
        )

        if run_scoring:
            logger.info("Starting Scoring")
            score(target_path, pred_path, obs_pred_path=obs_pred_path, freq=freq, generate_kwargs=generate_kwargs)
        if run_plotting:
            logger.info("Starting Plotting")
            plot(
                target_path,
                pred_path,
                obs_pred_path=obs_pred_path,
                obs_compare_path=obs_compare_path,
                freq=freq,
                data_path_forecast=data_path_forecast,
                data_kwargs=data_kwargs,
                movie_interval=movie_interval,
                num_workers=num_workers,
                plot_types=plot_types,
                generate_kwargs=generate_kwargs,
            )
