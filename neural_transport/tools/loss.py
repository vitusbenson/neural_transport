import torch
import torch.nn as nn


class MAE(nn.Module):
    def __init__(self, weights={}):
        super().__init__()

        self.vars = list(weights.keys())

        for variable, weight in weights.items():
            self.register_buffer(
                f"weights_{variable}", torch.from_numpy(weight.astype("float32"))
            )  # N, C

    def forward(self, preds, batch):
        loss = 0
        losses = {}
        for v in self.vars:
            se = (preds[v] - batch[f"{v}_next"]).abs()
            wmse = torch.mean(se * getattr(self, f"weights_{v}"))
            losses[f"Loss_Vari/weighted_{v}"] = wmse
            losses[f"Loss_Vari/unweighted{v}"] = torch.mean(se)
            # if getattr(self, f"weights_{v}").shape[-1] > se.shape[-1]:
            #     print(
            #         "shape mismatch loss", v, se.shape, getattr(self, f"weights_{v}").shape
            #     )
            if self.training and not wmse.requires_grad:
                print("no grad", v, preds[v].requires_grad, preds[v].shape)
            if not torch.isfinite(wmse).all():
                print(
                    v,
                    wmse,
                    preds[v].min(),
                    preds[v].max(),
                    batch[v].min(),
                    batch[v].max(),
                    getattr(self, f"weights_{v}").min(),
                    getattr(self, f"weights_{v}").max(),
                )

            loss = loss + wmse

        return loss, losses


class MSE(nn.Module):
    def __init__(self, weights={}):
        super().__init__()

        self.vars = list(weights.keys())

        for variable, weight in weights.items():
            self.register_buffer(
                f"weights_{variable}", torch.from_numpy(weight.astype("float32"))
            )  # N, C

    def forward(self, preds, batch):
        loss = 0
        losses = {}
        for v in self.vars:
            se = (preds[v] - batch[f"{v}_next"]) ** 2
            wmse = torch.mean(se * getattr(self, f"weights_{v}"))
            losses[f"Loss_Vari/weighted_{v}"] = wmse
            losses[f"Loss_Vari/unweighted{v}"] = torch.mean(se)
            # if getattr(self, f"weights_{v}").shape[-1] > se.shape[-1]:
            #     print(
            #         "shape mismatch loss", v, se.shape, getattr(self, f"weights_{v}").shape
            #     )
            if self.training and not wmse.requires_grad:
                print("no grad", v, preds[v].requires_grad, preds[v].shape)
            if not torch.isfinite(wmse).all():
                print(
                    v,
                    wmse,
                    preds[v].min(),
                    preds[v].max(),
                    batch[v].min(),
                    batch[v].max(),
                    getattr(self, f"weights_{v}").min(),
                    getattr(self, f"weights_{v}").max(),
                )

            loss = loss + wmse

        return loss, losses


LOSSES = {
    "mse": MSE,
    "mae": MAE,
}
