import tempfile
import time
from pathlib import Path

import pandas as pd
import torch


def iterative_benchmark(
    model,
    dataset,
    out_path=None,
    n_steps=120,
    n_repeats=5,
    device="cpu",
):

    model = model.eval().to(device)

    batch = {k: v.unsqueeze(0).to(device) for k, v in dataset[0].items()}

    temp_dir = tempfile.TemporaryDirectory()
    torch.save(batch, Path(temp_dir.name) / f"batch.pt")

    times = []

    for read_on_step, write_on_step in [(False, False), (True, False), (True, True)]:
        for repeat in range(n_repeats):

            print(
                f"Repeat {repeat}, read_on_step={read_on_step}, write_on_step={write_on_step}"
            )
            start = time.process_time()

            for i in range(n_steps):

                if read_on_step:
                    batch = torch.load(Path(temp_dir.name) / f"batch.pt")
                    # batch = {
                    #     k: v.unsqueeze(0).to(device) for k, v in dataset[i].items()
                    # }

                with torch.no_grad():
                    preds = model(batch)

                if write_on_step:
                    torch.save(preds, Path(temp_dir.name) / f"out.pt")

            end = time.process_time()

            elapsed = end - start

            times.append(
                {
                    "repeat": repeat,
                    "time": elapsed,
                    "read_on_step": read_on_step,
                    "write_on_step": write_on_step,
                }
            )

    df = pd.DataFrame(times)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.mkdir(exist_ok=True, parents=True)

        df.to_csv(out_path / f"benchmark_times.csv")

    temp_dir.cleanup()

    return df
