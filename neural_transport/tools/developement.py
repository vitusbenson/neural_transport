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

    print(
        f"  {name}: "
        f"min={torch.nanmin(t).item():.3e}, "
        f"max={torch.nanmax(t).item():.3e}, "
        f"mean={torch.nanmean(t).item():.3e}, "
        f"std={torch.nanstd(t).item():.3e}, "
        f"NaNs={n_nan}/{total}"
    )