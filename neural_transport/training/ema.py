"""Exponential Moving Average (EMA) callback for PyTorch Lightning.

Standard in flow matching / diffusion training. Maintains a shadow copy
of model weights with exponential decay, swaps in for validation/generation.
Typically gives 10-20% quality improvement for free.
"""

import copy

import pytorch_lightning as pl
import torch


class EMACallback(pl.Callback):
    """Exponential Moving Average of model weights.

    Maintains shadow EMA weights and swaps them in during validation
    and test steps for improved generation quality.

    Args:
        decay: EMA decay rate. Higher = smoother averaging. Default 0.9999.
        ema_start_step: Don't update EMA until this training step. Default 0.
    """

    def __init__(self, decay=0.9999, ema_start_step=0):
        super().__init__()
        self.decay = decay
        self.ema_start_step = ema_start_step
        self.ema_state_dict = None
        self.original_state_dict = None

    def on_fit_start(self, trainer, pl_module):
        """Initialize EMA state dict as a copy of model weights."""
        self.ema_state_dict = copy.deepcopy({k: v.clone() for k, v in pl_module.state_dict().items()})

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update EMA weights after each training step."""
        if self.ema_state_dict is None:
            return
        if trainer.global_step < self.ema_start_step:
            # Before start step, just copy weights directly
            for key, param in pl_module.state_dict().items():
                self.ema_state_dict[key].copy_(param)
            return

        decay = self.decay
        for key, param in pl_module.state_dict().items():
            if param.is_floating_point():
                self.ema_state_dict[key].mul_(decay).add_(param, alpha=1 - decay)
            else:
                # Non-float params (e.g. batch norm num_batches_tracked): copy directly
                self.ema_state_dict[key].copy_(param)

    def on_validation_start(self, trainer, pl_module):
        """Swap in EMA weights for validation."""
        if self.ema_state_dict is None:
            return
        self.original_state_dict = {k: v.clone() for k, v in pl_module.state_dict().items()}
        pl_module.load_state_dict(self.ema_state_dict)

    def on_validation_end(self, trainer, pl_module):
        """Restore original weights after validation."""
        if self.original_state_dict is None:
            return
        pl_module.load_state_dict(self.original_state_dict)
        self.original_state_dict = None

    def on_test_start(self, trainer, pl_module):
        """Swap in EMA weights for testing."""
        self.on_validation_start(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        """Restore original weights after testing."""
        self.on_validation_end(trainer, pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Save EMA state dict alongside regular checkpoint."""
        checkpoint["ema_state_dict"] = self.ema_state_dict

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """Restore EMA state dict from checkpoint."""
        if "ema_state_dict" in checkpoint:
            self.ema_state_dict = checkpoint["ema_state_dict"]
