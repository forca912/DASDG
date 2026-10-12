"""Run leakage-free naive forecasting baselines on masked PEMS inputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denominator = float(np.sum(np.abs(target)))
    wape = float(np.sum(np.abs(error)) / max(denominator, 1e-8) * 100.0)
    return {"mae": mae, "rmse": rmse, "wape": wape}


def fill_history(history: np.ndarray, observed: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    filled = np.empty_like(history, dtype=np.float32)
    last = fallback.astype(np.float32).copy()
    for step in range(history.shape[0]):
        visible = observed[step]
        last[visible] = history[step, visible]
        filled[step] = last
    return filled


def evaluate(
    flow: np.ndarray,
    observed: np.ndarray,
    train_mean: np.ndarray,
    input_steps: int,
    output_steps: int,
) -> list[dict[str, float | int | str]]:
    predictions = {"persistence": [], "history_mean": []}
    targets = []
    for end in range(input_steps, len(flow) - output_steps + 1):
        history = flow[end - input_steps : end]
        history_mask = observed[end - input_steps : end]
        filled = fill_history(history, history_mask, train_mean)
        predictions["persistence"].append(np.repeat(filled[-1][None, :], output_steps, axis=0))
        predictions["history_mean"].append(np.repeat(filled.mean(axis=0)[None, :], output_steps, axis=0))
        targets.append(flow[end : end + output_steps])

    target_array = np.stack(targets)
    results: list[dict[str, float | int | str]] = []
    horizons = sorted(set([3, 6, 9, output_steps]))
    for name, values in predictions.items():
        prediction_array = np.stack(values)
        for horizon in horizons:
            score = metrics(prediction_array[:, horizon - 1], target_array[:, horizon - 1])
            results.append({"model": name, "horizon": horizon, **score})
        overall = metrics(prediction_array, target_array)
        results.append({"model": name, "horizon": "all", **overall})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["PEMS04", "PEMS08"])
    parser.add_argument("--mask-file", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/naive"))
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--output-steps", type=int, default=12)
    args = parser.parse_args()

    complete = np.load(args.processed_dir / args.dataset / "complete.npz")
    flow = complete["flow"].astype(np.float32)
    train_end, val_end, total = (int(value) for value in complete["split"])
    train_mean = flow[:train_end].mean(axis=0)
    mask = np.load(args.mask_file)
    test_observed = mask["test"].astype(bool)
    test_flow = flow[val_end:total]
    if test_observed.shape != test_flow.shape:
        raise ValueError(f"Mask shape {test_observed.shape} != test flow shape {test_flow.shape}")

    results = evaluate(test_flow, test_observed, train_mean, args.input_steps, args.output_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.dataset}_{args.mask_file.stem}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "model", "horizon", "mae", "rmse", "wape"])
        writer.writeheader()
        for row in results:
            writer.writerow({"dataset": args.dataset, **row})

    print(f"dataset={args.dataset}, test_windows={len(test_flow) - args.input_steps - args.output_steps + 1}")
    for row in results:
        print(
            f"{row['model']:>12} horizon={str(row['horizon']):>3} "
            f"MAE={row['mae']:.4f} RMSE={row['rmse']:.4f} WAPE={row['wape']:.2f}%"
        )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
