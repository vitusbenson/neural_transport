import multiprocessing
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import xarray as xr

from neural_transport.datamodule import CarbonDataModule, CarbonDataset
from neural_transport.inference.analyse import compute_local_scores, compute_score_df
from neural_transport.inference.forecast import iterative_forecast
from neural_transport.inference.plot_results import (
    animate_predictions,
    plot_metrics,
    plot_obspack_stations,
)
from neural_transport.litmodule import NeuralTransport


def train_singlestep(run_dir, data_kwargs, lit_module_kwargs, trainer_kwargs):
    run_dir = Path(run_dir)

    logger = pl.loggers.tensorboard.TensorBoardLogger(
        run_dir, name="", version="singlestep"
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=2,  # -1,
        save_last=True,
        monitor="Loss/Val_rollout",
        filename="Epoch={epoch}-Step={step}-LossVal={Loss/Val_rollout:.6f}",
        auto_insert_metric_name=False,
        every_n_epochs=1,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor()

    dset = CarbonDataModule(**data_kwargs)

    model = NeuralTransport(**lit_module_kwargs)

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        **trainer_kwargs,
    )

    print("Starting singlestep training")
    trainer.fit(model, dset)


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
        lit_module_kwargs["lr_shedule_kwargs"] = dict(
            warmup_steps=1, halfcosine_steps=100000, min_lr=1, max_lr=1
        )

    log_path = run_dir / "singlestep"
    ckptdir = log_path / "checkpoints"
    bestckptpath = sorted(
        [p for p in ckptdir.glob("*.ckpt") if "LossVal" in p.name],
        key=lambda p: float(p.name.split("=")[-1].split(".c")[0]),
    )[0]

    lastckptpath = log_path / "checkpoints/last.ckpt"

    ckptpath = bestckptpath if ckpt == "best" else lastckptpath

    model = NeuralTransport.load_from_checkpoint(
        ckptpath,
        map_location="cpu",
        **lit_module_kwargs,
    )

    logger = pl.loggers.tensorboard.TensorBoardLogger(
        run_dir, name="", version="rollout"
    )
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

        print(f"Starting training {n_timesteps} timesteps")
        trainer.fit(model, dset)
        trainer.fit_loop.max_epochs += 2
        trainer.fit_loop.epoch_loop.val_loop._results.clear()


def load_dataset(data_path, data_kwargs):

    dataset = CarbonDataset(
        data_path=data_path,
        dataset=data_kwargs["dataset"],
        grid=data_kwargs["grid"],
        vertical_levels=data_kwargs["vertical_levels"],
        freq=data_kwargs["freq"],
        n_timesteps=1,
        target_vars=data_kwargs["target_vars"],
        forcing_vars=data_kwargs["forcing_vars"],
        load_obspack=True,
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
):
    log_path = Path(log_path)

    ckptdir = log_path / "checkpoints"
    bestckptpath = sorted(
        [p for p in ckptdir.glob("*.ckpt") if "LossVal" in p.name],
        key=lambda p: float(p.name.split("=")[-1].split(".c")[0]),
    )[0]

    lastckptpath = log_path / "checkpoints/last.ckpt"

    ckptpath = bestckptpath if ckpt == "best" else lastckptpath

    if massfixer != "default":
        lit_module_kwargs["model_kwargs"]["massfixer"] = massfixer
    outpath = log_path / "preds" / f"ckpt={ckpt}_massfixer={massfixer}"

    dataset = load_dataset(data_path_forecast, data_kwargs)

    print(f"Forecasting {ckptpath} {ckpt} CKPT")
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
        target_vars_3d=["co2massmix"],
        target_vars_2d=[],
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


def score(target_path, pred_path, obs_pred_path=None):
    co2targ, co2pred = load_pred_targ(target_path, pred_path)
    co2pred = co2pred.isel(time=slice(1, None))
    co2targ = co2targ.isel(time=slice(1, None)).isel(time=slice(len(co2pred.time)))

    model_name = pred_path.parent.parent.parent.parent.name
    singlestep_or_rollout = pred_path.parent.parent.parent.name
    ckpt_name = pred_path.parent.name
    score_path = pred_path.parent.parent.parent / "scores" / ckpt_name

    metrics = {}
    metrics[f"{model_name}_{singlestep_or_rollout}_{ckpt_name}"] = compute_score_df(
        co2targ, co2pred
    )

    df = pd.DataFrame(metrics).T
    score_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(score_path / "metrics.csv")

    if obs_pred_path:
        obspreds = xr.open_zarr(obs_pred_path)  # .isel(time = slice(1,None))

        print(f"Computing score for {obs_pred_path}")

        df = compute_local_scores(obs_preds=obspreds)

        df.to_csv(score_path / "obs_metrics.csv")


def plot(
    target_path,
    pred_path,
    obs_pred_path=None,
    obs_compare_path=None,
    data_path_forecast=None,
    data_kwargs=None,
    movie_interval=["2018-01-01", "2018-03-31"],
    num_workers=multiprocessing.cpu_count() // 2,
):
    co2targ, co2pred = load_pred_targ(target_path, pred_path)

    ckpt_name = pred_path.parent.name
    plot_path = pred_path.parent.parent.parent / "plots" / ckpt_name

    co2pred = co2pred.isel(time=slice(1, None))
    co2targ = co2targ.isel(time=slice(1, None))
    co2targ = co2targ.isel(time=slice(None, len(co2pred.time)))

    plot_path.mkdir(parents=True, exist_ok=True)
    plot_metrics(co2pred, co2targ, plot_path, imgformats=["pdf"])

    t0, tend = movie_interval

    animate_predictions(
        co2pred.sel(time=slice(t0, tend)),
        co2targ.sel(time=slice(t0, tend)),
        plot_path,
        postfix=f"3d_anim_t0={t0}-tend={tend}",
        num_workers=num_workers,
    )

    if obs_pred_path:
        obspreds = xr.open_zarr(obs_pred_path)

        dataset = load_dataset(data_path_forecast, data_kwargs)

        carboscope_obspred = (
            xr.open_zarr(obs_compare_path) if obs_compare_path else None
        )

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
):
    if train:
        train_singlestep(run_dir, data_kwargs, lit_module_kwargs, trainer_kwargs)
    predict(
        run_dir / "singlestep",
        data_path_forecast,
        data_kwargs,
        device=device,
        freq=freq,
        ckpt=ckpt,
        lit_module_kwargs=lit_module_kwargs,
    )
    target_path = (
        data_path_forecast
        / f"{data_kwargs['dataset']}_{data_kwargs['grid']}_{data_kwargs['vertical_levels']}_{data_kwargs['freq']}.zarr"
    )
    pred_path = (
        run_dir / "singlestep" / "preds" / ckpt / f"co2_pred_rollout_{freq}.zarr"
    )
    obs_pred_path = (
        run_dir / "singlestep" / "preds" / ckpt / f"obs_co2_pred_rollout_{freq}.zarr"
    )
    score(target_path, pred_path, obs_pred_path)
    plot(
        target_path,
        pred_path,
        obs_pred_path,
        obs_compare_path,
        data_path_forecast,
        data_kwargs,
        movie_interval,
        num_workers,
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
):
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

    for massfixer in massfixers:
        predict(
            run_dir / "rollout",
            data_path_forecast,
            data_kwargs,
            device=device,
            freq=freq,
            ckpt=ckpt,
            lit_module_kwargs=lit_module_kwargs,
            massfixer=massfixer,
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
            / f"co2_pred_rollout_{freq}.zarr"
        )
        obs_pred_path = (
            run_dir
            / "rollout"
            / "preds"
            / f"ckpt={ckpt}_massfixer={massfixer}"
            / f"obs_co2_pred_rollout_{freq}.zarr"
        )

        print("Starting Scoring")
        score(target_path, pred_path, obs_pred_path)
        plot(
            target_path,
            pred_path,
            obs_pred_path,
            obs_compare_path,
            data_path_forecast,
            data_kwargs,
            movie_interval,
            num_workers,
        )
