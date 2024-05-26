import math

import numpy as np
import pytorch_lightning as pl
import torch

from neural_transport.models import MODELS
from neural_transport.tools.loss import LOSSES
from neural_transport.tools.metrics import ManyMetrics
from neural_transport.tools.plot import plots_val_step_v2


class CarbonFM(pl.LightningModule):
    def __init__(
        self,
        model="gnn",
        model_kwargs={},
        loss="mse",
        loss_kwargs={},
        metrics=[
            {"name": "rmse", "kwargs": {"weights": {"co2massmix": np.ones((1, 1, 1))}}}
        ],
        no_grad_step_shedule=None,
        lr=1e-3,
        weight_decay=0.1,
        lr_shedule_kwargs=dict(
            warmup_steps=1000, halfcosine_steps=299000, min_lr=3e-7, max_lr=1.0
        ),
        val_dataloader_names=["singlestep", "rollout"],
        plot_kwargs=dict(
            variables=["co2molemix"],
            layer_idxs=[0, 1, 9, 15],
            n_samples=4,
            grid="latlon1",
            max_workers=32,
        ),
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MODELS[model](**model_kwargs)
        self.loss = LOSSES[loss](**loss_kwargs)
        self.metrics = ManyMetrics(metrics)

    def forward(self, batch):
        T = max(batch[v].shape[1] for v in batch if isinstance(batch[v], torch.Tensor))

        for t in range(T):
            if t == 0:
                curr_preds = {}  # {batch[v][:, t] for v in self.hparams.target_vars}

            curr_data = {
                v: batch[v][:, t] if batch[v].shape[1] == T else batch[v][:, 0]
                for v in batch
                if isinstance(batch[v], torch.Tensor)
            }

            curr_data |= curr_preds

            if self.no_grad_shedule(self.global_step, t):
                with torch.no_grad():
                    curr_preds = self.model(curr_data)
            else:
                curr_preds = self.model(curr_data)

            if t == 0:
                preds = {k: torch.empty_like(batch[k]) for k in curr_preds}

            for v in preds:
                preds[v][:, t] = curr_preds[v]

        return preds

    def no_grad_shedule(self, global_step, t):
        return (
            self.hparams.no_grad_step_shedule
            and (global_step > self.hparams.no_grad_step_shedule["from_step"])
            and (t in self.hparams.no_grad_step_shedule["t_no_grad"])
        )

    def common_step(self, batch):
        preds = self(batch)

        loss, losses = self.loss(preds, batch)

        return loss, losses, preds

    def training_step(self, batch, batch_idx):
        loss, losses, preds = self.common_step(batch)

        self.log("Loss/Train", loss, prog_bar=True)
        self.log_dict(losses)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        dataloader_name = self.hparams.val_dataloader_names[dataloader_idx]

        loss, losses, preds = self.common_step(batch)

        self.log(
            f"Loss/Val_{dataloader_name}",
            loss,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        metrics = self.metrics(preds, batch)

        self.log_dict(
            {f"{k}_Val_{dataloader_name}": v for k, v in metrics.items()},
            sync_dist=True,
            add_dataloader_idx=False,
        )

        self.plots(preds, batch, batch_idx, dataloader_idx)

    def plots(self, preds, batch, batch_idx, dataloader_idx):
        if (batch_idx < 1) and (dataloader_idx == 0) and (self.global_rank == 0):
            plots_val_step_v2(
                self.logger.experiment,
                self.current_epoch,
                preds,
                batch,
                batch_idx=batch_idx,
                **self.hparams.plot_kwargs,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(warmup_steps, halfcosine_steps, min_lr=3e-7, max_lr=1.0):
            def ret_lambda(current_step):
                if current_step <= warmup_steps:
                    return min_lr + (max_lr - min_lr) * current_step / warmup_steps
                elif current_step <= warmup_steps + halfcosine_steps:
                    return min_lr + (max_lr - min_lr) * (
                        (
                            math.cos(
                                ((current_step - warmup_steps) / (halfcosine_steps))
                                * math.pi
                            )
                            + 1
                        )
                        / 2
                    )
                else:
                    return min_lr

            return ret_lambda

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda(**self.hparams.lr_shedule_kwargs)
        )
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
