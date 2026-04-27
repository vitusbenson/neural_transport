import math

import numpy as np
import pytorch_lightning as pl
import torch

from neural_transport.models import MODELS
from neural_transport.models.wrappers_registry import MODELWRAPPERS
from neural_transport.tools.loss import LOSSES
from neural_transport.tools.metrics import ManyMetrics
from neural_transport.tools.plot import plots_val_step


class NeuralTransport(pl.LightningModule):
    def __init__(
        self,
        model="gnn",
        model_kwargs={},
        loss="mse",
        loss_kwargs={},
        metrics=[{"name": "rmse", "kwargs": {"weights": {"co2massmix": np.ones((1, 1, 1))}}}],
        no_grad_step_shedule=None,
        lr=1e-3,
        weight_decay=0.1,
        lr_shedule_kwargs=dict(warmup_steps=1000, halfcosine_steps=299000, min_lr=3e-7, max_lr=1.0),
        val_dataloader_names=["singlestep", "rollout"],
        plot_kwargs=dict(
            variables=["co2molemix"],
            layer_idxs=[0, 1, 9, 15],
            n_samples=4,
            grid="latlon1",
            max_workers=32,
        ),
        pretrained_ckptpath=None,
        pushforward_kwargs=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Phase 25c v1: K-step pushforward (ArchesWeatherGen-style; for FM: K-1
        # no-grad inference steps that chain the prior, followed by a single
        # FM training-loss step on the drifted prior).
        self.pushforward_kwargs = pushforward_kwargs or {}
        if model in MODELS:
            self.model = MODELS[model](**model_kwargs)
        elif model in MODELWRAPPERS:
            self.model = MODELWRAPPERS[model](**model_kwargs)
        else:
            self.model = model
        if pretrained_ckptpath is not None:
            ckpt = torch.load(pretrained_ckptpath, map_location="cpu", weights_only=False)
            model_state_dict = {
                k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")
            }
            for key in [
                "multiscale_encoder.position_feats",
                "multiscale_decoder.position_feats",
            ]:
                model_state_dict.pop(key, None)

            self.model.load_state_dict(model_state_dict, strict=False)

        self.loss = LOSSES[loss](**loss_kwargs)
        self.metrics = ManyMetrics(metrics)

    def forward(self, batch, *, mode=None):
        T = max(batch[v].shape[1] for v in batch if isinstance(batch[v], torch.Tensor))

        # Only FlowMatching accepts the mode kwarg; other models ignore it.
        extra_kwargs = {"mode": mode} if mode is not None else {}

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
                    curr_preds = self.model(curr_data, **extra_kwargs)
            else:
                curr_preds = self.model(curr_data, **extra_kwargs)
            if t == 0:
                preds = {
                    k: torch.empty((curr_preds[k].shape[0], T, *curr_preds[k].shape[1:]), device=curr_preds[k].device)
                    for k in curr_preds
                }

            for v in preds:
                preds[v][:, t] = curr_preds[v]

        if T == 1 and "trajectory" in preds:
            preds["trajectory"] = preds["trajectory"].squeeze(1)  # [T_Flow T B N C] -> [T_Flow B N C]
            preds["trajectory"] = preds["trajectory"].permute(1, 0, 2, 3)
        return preds  # [B, T, Nlat*Nlon, C]

    def no_grad_shedule(self, global_step, t):
        return (
            self.hparams.no_grad_step_shedule
            and (global_step > self.hparams.no_grad_step_shedule["from_step"])
            and (t in self.hparams.no_grad_step_shedule["t_no_grad"])
        )

    def common_step(self, batch, *, mode=None):
        preds = self(batch, mode=mode)

        loss, losses = self.loss(preds, batch)

        return loss, losses, preds

    def _pushforward_chain_prior(self, batch, K, target_var, inference_steps, method, grad_through_inference=False):
        """Run K-1 no-grad inference steps to obtain a drifted prior at step K-1.

        Returns a single-timestep `sub_batch` ready for the standard training
        forward. The chained prior replaces `batch[target_var][:, K-1]`, and
        all other tensors keep their step-(K-1) slice. Tensors with time-dim 1
        broadcast unchanged.
        """
        is_fm = isinstance(self.model, MODELWRAPPERS["flowmatching"])
        if not is_fm:
            return batch  # only FM supports the inference chain

        T_dim = batch[target_var].shape[1]
        K = min(K, T_dim)
        if K < 2:
            return batch

        inner = self.model
        prev_kwargs = getattr(inner, "generate_kwargs", {}) or {}
        cheap_kwargs = {
            "steps": int(inference_steps),
            "method": method,
            "n_samples": 1,
            "masking": False,
        }

        def slice_t(d, t):
            out = {}
            for v, val in d.items():
                if isinstance(val, torch.Tensor) and val.ndim >= 3:
                    # Mimic NeuralTransport.forward: slice [:, t] when time dim
                    # equals T_dim, else [:, 0] (broadcasts the stationary stat).
                    if val.shape[1] == T_dim:
                        out[v] = val[:, t]
                    elif val.shape[1] == 1:
                        out[v] = val[:, 0]
                    else:
                        out[v] = val
                else:
                    out[v] = val
            return out

        prior = batch[target_var][:, 0]
        # Phase 25c v2/v3: disable bf16 autocast for the chaining inference;
        # v3 enables grad through the ODE solve (AWG-style backprop).
        grad_ctx = torch.enable_grad() if grad_through_inference else torch.no_grad()
        with grad_ctx, torch.autocast(device_type="cuda" if prior.is_cuda else "cpu", enabled=False):
            inner.generate_kwargs = cheap_kwargs
            try:
                for t in range(K - 1):
                    sub = slice_t(batch, t)
                    sub[target_var] = prior
                    next_key = f"{target_var}_next"
                    if next_key in sub:
                        sub[next_key] = prior.clone()
                    pred = inner(sub, mode="generate")
                    prior = pred[target_var]
                    if not grad_through_inference:
                        prior = prior.detach()
            finally:
                inner.generate_kwargs = prev_kwargs

        # Build sub_batch at step K-1 with the drifted prior, but keep a leading
        # time-dim of 1 so the litmodule forward loop runs T=1.
        sub_batch = {}
        for v, val in batch.items():
            if isinstance(val, torch.Tensor) and val.ndim >= 3:
                if val.shape[1] == T_dim:
                    sub_batch[v] = val[:, K - 1 : K]
                else:
                    sub_batch[v] = val
            else:
                sub_batch[v] = val
        sub_batch[target_var] = prior.unsqueeze(1)
        return sub_batch

    def training_step(self, batch, batch_idx):
        applied_pushforward = False
        if self.pushforward_kwargs:
            K = int(self.pushforward_kwargs.get("K", 2))
            K_max = int(self.pushforward_kwargs.get("K_max", K))
            from_step = int(self.pushforward_kwargs.get("from_step", 0))
            target_var = self.pushforward_kwargs.get("target_var", "co2massmix")
            inference_steps = int(self.pushforward_kwargs.get("inference_steps", 5))
            method = self.pushforward_kwargs.get("method", "euler")
            grad_through_inf = bool(self.pushforward_kwargs.get("grad_through_inference", False))

            # Phase 25c v2: linear curriculum on prob between
            # `curriculum_start_step` and `curriculum_end_step`. If both unset,
            # falls back to the constant `prob` from v1.
            curr_start = self.pushforward_kwargs.get("curriculum_start_step")
            curr_end = self.pushforward_kwargs.get("curriculum_end_step")
            prob_max = float(self.pushforward_kwargs.get("prob", 0.5))
            prob_min = float(self.pushforward_kwargs.get("prob_min", 0.0))
            if curr_start is not None and curr_end is not None and curr_end > curr_start:
                gs = self.global_step
                if gs <= curr_start:
                    prob = prob_min
                elif gs >= curr_end:
                    prob = prob_max
                else:
                    frac = (gs - curr_start) / (curr_end - curr_start)
                    prob = prob_min + frac * (prob_max - prob_min)
            else:
                prob = prob_max

            self.log("Pushforward/prob", prob, prog_bar=False)

            apply = (
                target_var in batch
                and self.global_step >= from_step
                and (torch.rand(()).item() < prob)
                and batch[target_var].ndim >= 3
                and batch[target_var].shape[1] >= 2
            )
            if apply:
                # Randomize K (lead time) per iteration in [K, K_max].
                if K_max > K:
                    T_dim = batch[target_var].shape[1]
                    K_actual = int(torch.randint(K, min(K_max, T_dim) + 1, (1,)).item())
                else:
                    K_actual = K
                self.log("Pushforward/K", float(K_actual), prog_bar=False)
                batch = self._pushforward_chain_prior(
                    batch,
                    K_actual,
                    target_var,
                    inference_steps,
                    method,
                    grad_through_inference=grad_through_inf,
                )
                applied_pushforward = True

        loss, losses, preds = self.common_step(batch)

        # Phase 25c v2: down-weight pushforward-batch loss so standard FM
        # batches dominate gradients (default 1.0 keeps v1 behaviour).
        if applied_pushforward and self.pushforward_kwargs:
            lw = float(self.pushforward_kwargs.get("loss_weight", 1.0))
            if lw != 1.0:
                loss = loss * lw

        self.log("Loss/Train", loss, prog_bar=True)
        self.log_dict(losses)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        dataloader_name = self.hparams.val_dataloader_names[dataloader_idx]

        is_fm = isinstance(self.model, MODELWRAPPERS["flowmatching"])

        # Use mode="train" to dispatch FlowMatching to training_forward()
        # without calling model.train(), which would corrupt BatchNorm stats.
        loss, losses, preds = self.common_step(batch, mode="train" if is_fm else None)

        if is_fm:
            for v in preds:
                if v not in ("dx_t", "time_loss_weight"):
                    preds[v] = preds[v] * batch[f"{v}_scale"] + batch[f"{v}_offset"]

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
        if (batch_idx < 1) and (dataloader_idx == 0) and (self.global_rank == 0) and self.logger is not None:
            plots_val_step(
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
                        (math.cos(((current_step - warmup_steps) / (halfcosine_steps)) * math.pi) + 1) / 2
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
