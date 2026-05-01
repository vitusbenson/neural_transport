import torch


def _print_stats_torch(
    name: str,
    t: torch.Tensor
    ) -> None:
    nan_mask = torch.isnan(t)
    n_nan = nan_mask.sum().item()
    total = t.numel()

    if n_nan == total:
        print(f"  {name}: ALL NaN")
        return
    
    valid = t[~nan_mask]

    print(
        f"  {name}: "
        f"min={valid.min().item():.2f}, "
        f"max={valid.max().item():.2f}, "
        f"mean={torch.nanmean(t).item():.2f}, "
        f"std={valid.std().item():.2f}, "
        f"NaNs={n_nan}/{total}"
    )