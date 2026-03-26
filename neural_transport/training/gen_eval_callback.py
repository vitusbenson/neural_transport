"""Generation quality evaluation callback for PyTorch Lightning.

Periodically generates a small ensemble during validation, computes
ensemble-mean RMSE and energy distance, and logs metrics for checkpoint
selection based on actual generation quality rather than velocity MSE.
"""

import logging

import numpy as np
import pytorch_lightning as pl
import torch

logger = logging.getLogger(__name__)

from neural_transport.evaluation import energy_distance
from neural_transport.evaluation.suite import EvaluationSuite
from neural_transport.inference.generation import is_bad_sample


class GenerationQualityCallback(pl.Callback):
    """Evaluate generation quality during training.

    Every `eval_every_n_epochs` epochs: generate a small ensemble from the
    validation set, compute ensemble-mean RMSE + energy distance, and log
    as GenEval/* metrics.

    Args:
        val_dataset: CarbonDataset for validation data.
        n_gt_samples: Number of GT samples to draw from val set.
        n_gen_samples: Number of unconditional samples to generate.
        eval_every_n_epochs: Run evaluation every N epochs.
        target_var: Target variable name.
        generate_kwargs: kwargs for generation (steps, method, etc.).
    """

    def __init__(
        self,
        val_dataset,
        n_gt_samples=5,
        n_gen_samples=20,
        eval_every_n_epochs=5,
        target_var="co2massmix",
        generate_kwargs=None,
        eval_config=None,
    ):
        super().__init__()
        self.val_dataset = val_dataset
        self.n_gt_samples = n_gt_samples
        self.n_gen_samples = n_gen_samples
        self.eval_every_n_epochs = eval_every_n_epochs
        self.target_var = target_var
        # Fast ODE defaults for training-time evaluation
        _fast_defaults = {"method": "euler", "steps": 5}
        _fast_defaults.update(generate_kwargs or {})
        self.generate_kwargs = _fast_defaults
        self.suite = EvaluationSuite(eval_config)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate samples and compute quality metrics."""
        current_epoch = trainer.current_epoch
        if current_epoch % self.eval_every_n_epochs != 0:
            return

        device = pl_module.device
        model = pl_module.model

        # Check if model is FlowMatching
        if not hasattr(model, 'inference_forward'):
            return

        try:
            # Collect GT samples
            n_gt = min(self.n_gt_samples, len(self.val_dataset))
            indices = np.random.choice(len(self.val_dataset), n_gt, replace=False)

            gt_fields = []
            for idx in indices:
                batch = self.val_dataset[idx]
                field = batch[self.target_var]  # [N, C] or [T, N, C]
                if field.ndim == 3:
                    field = field[0]  # Take first timestep
                gt_fields.append(field.numpy())

            gt_fields = np.stack(gt_fields)  # [n_gt, N, C]

            # Generate unconditional samples
            was_generating = getattr(model, 'generating', False)
            was_training = model.training
            model.generating = True
            model.eval()

            gen_fields = []
            n_valid = 0
            n_total = self.n_gen_samples

            # Use a single batch from val set as template
            # Squeeze time dim if present (T=1), then add batch dim
            raw_sample = self.val_dataset[0]
            template_batch = {}
            for k, v in raw_sample.items():
                if not isinstance(v, torch.Tensor):
                    continue
                if v.ndim >= 2 and v.shape[0] == 1:
                    v = v.squeeze(0)  # [1, N, C] -> [N, C]
                template_batch[k] = v.unsqueeze(0).to(device)  # [N, C] -> [1, N, C]

            model.generate_kwargs = self.generate_kwargs

            for _ in range(n_total):
                batch_i = {k: v.clone() for k, v in template_batch.items()}
                with torch.no_grad():
                    preds = model(batch_i)
                if self.target_var in preds:
                    field = preds[self.target_var].cpu().numpy()
                    if field.ndim == 4:  # [B, T, N, C]
                        field = field[0, -1]  # Last timestep, first batch
                    elif field.ndim == 3:  # [B, N, C]
                        field = field[0]
                    if not is_bad_sample(field):
                        gen_fields.append(field)
                        n_valid += 1

            # Restore model state
            model.generating = was_generating
            if was_training:
                model.train()

            valid_fraction = n_valid / n_total

            # Always log valid_fraction
            pl_module.log("GenEval/valid_fraction", valid_fraction)

            if n_valid == 0:
                logger.warning("GenEval: 0/%d samples valid", n_total)
                pl_module.log("GenEval/RMSE", float("inf"))
                pl_module.log("GenEval/energy_distance", float("inf"))
                pl_module.log("GenEval/gen_std", 0.0)
                return

            if n_valid < 2:
                logger.warning("GenEval: only %d/%d valid samples, metrics may be noisy", n_valid, n_total)

            gen_fields = np.stack(gen_fields)  # [n_valid, N, C]

            # Compute RMSE via EvaluationSuite
            gt_mean = gt_fields.mean(axis=0)
            gen_mean = gen_fields.mean(axis=0)
            result = self.suite.evaluate_deterministic(gen_mean, gt_mean)

            # Compute energy distance on anomalies
            gt_anom = gt_fields - gt_fields.mean(axis=(1, 2), keepdims=True)
            gen_anom = gen_fields - gen_fields.mean(axis=(1, 2), keepdims=True)
            e_dist = energy_distance(gt_anom, gen_anom)

            # Log metrics
            pl_module.log("GenEval/RMSE", result.pointwise["rmse"], prog_bar=True)
            pl_module.log("GenEval/energy_distance", e_dist)
            pl_module.log("GenEval/gen_std", float(gen_fields.std()))

        except Exception as e:
            # Don't crash training if eval fails
            logger.warning("GenerationQualityCallback error: %s", e, exc_info=True)
