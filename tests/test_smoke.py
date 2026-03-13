"""End-to-end smoke test for the neural_transport pipeline.

Exercises: generate data -> train FM -> sample unconditional -> evaluate.
Runs by default (not marked slow) and marked quick for explicit selection.
"""

import pytest
import torch

from neural_transport.experiments.toy_column_osse import (
    ToyVelocityWrapper,
    create_column_obs,
    evaluate,
    generate_toy_data,
    get_column_weights,
    sample_unconditional,
    train_flow_matching,
)


@pytest.mark.quick
def test_e2e_smoke():
    """Minimal E2E: generate data, train 2 epochs, sample, evaluate — no NaN."""
    nlat, nlon, nlev = 4, 8, 4
    device = "cpu"

    # Generate tiny dataset
    data = generate_toy_data(n_samples=100, nlat=nlat, nlon=nlon, nlev=nlev, seed=42)
    assert data.shape == (100, nlev, nlat, nlon)

    # Train for 2 epochs
    column_weights = get_column_weights(nlev)
    conv_model, data_mean, data_std = train_flow_matching(data, nlev=nlev, epochs=2, lr=1e-3, device=device)
    velocity_wrapper = ToyVelocityWrapper(conv_model, nlev=nlev).to(device)

    # Sample 2 unconditional samples
    torch.manual_seed(0)
    samples = sample_unconditional(velocity_wrapper, 2, nlev, nlat, nlon, device, steps=10)
    assert samples.shape == (2, nlev, nlat, nlon)
    assert not torch.isnan(samples).any(), "NaN in smoke test samples"
    assert not torch.isinf(samples).any(), "Inf in smoke test samples"

    # Evaluate
    gt_norm = ((data[0:1] - data_mean) / data_std).to(device)
    obs_mask, obs_values = create_column_obs(data[0:1].to(device), column_weights, obs_fraction=0.3, seed=99)
    metrics = evaluate(samples, gt_norm, obs_mask, column_weights, data_mean, data_std)

    for key, val in metrics.items():
        assert not (val != val), f"NaN in metric {key}"  # NaN != NaN
