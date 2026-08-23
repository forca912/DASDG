"""Run the frozen v16B primary external-baseline confirmation matrix."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


MODELS = ("graph_wavenet_adaptive", "gru_mask_duration")
SEEDS = tuple(range(20260808, 20260814))
DATASETS = ("PEMS04", "PEMS08")
RATES = (0.05, 0.10, 0.15)
BLOCKS = (6, 12, 24)
HORIZONS = ("3", "6", "9", "12", "all")
SOURCE_DIR = Path(__file__).resolve().parent


def run(command: list[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command), flush=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}); inspect {log_path}")
    return elapsed


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if {row["horizon"] for row in rows} != set(HORIZONS):
        raise RuntimeError(f"Horizon coverage mismatch: {path}")
    return rows


def upsert(path: Path, row: dict[str, object], keys: tuple[str, ...]) -> None:
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            existing = list(csv.DictReader(handle))
    key = tuple(str(row[name]) for name in keys)
    retained = [item for item in existing if tuple(item[name] for name in keys) != key]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(retained)
        writer.writerow(row)


def mask_path(data_root: Path, dataset: str, seed: int, rate: float, block: int, fraction: float) -> Path:
    return data_root / "masks_v2" / dataset / "persistent_rate_controlled" / (
        f"seed{seed}_global{rate:g}_nodes{fraction:g}_block{block}.npz"
    )


def run_dir(root: Path, model: str, dataset: str, seed: int, rate: float, block: int) -> Path:
    parent = root / f"global{rate:g}" / f"block{block}" / model
    name = f"{dataset}_{model}_h64_lr0.001_d0.1_seed{seed}"
    return parent / name


def collect_model(
    output_root: Path,
    model: str,
    dataset: str,
    seed: int,
    rate: float,
    block: int,
    fraction: float,
    mask: Path,
    elapsed: float | None,
) -> None:
    directory = run_dir(output_root, model, dataset, seed, rate, block)
    required = (
        "history.csv", "validation_metrics.csv", "test_metrics.csv", "config.json",
        "efficiency.json", "adjacency.npz", "best.pt",
    )
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise RuntimeError(f"Missing files in {directory}: {missing}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8-sig"))
    efficiency = json.loads((directory / "efficiency.json").read_text(encoding="utf-8-sig"))
    masks = np.load(mask)
    metrics = read_metrics(directory / "test_metrics.csv")
    overall = next(row for row in metrics if row["horizon"] == "all")
    summary = {
        "dataset": dataset,
        "global_missing_rate": rate,
        "affected_node_fraction": fraction,
        "block_length": block,
        "model": model,
        "config_id": "standard",
        "seed": seed,
        "test_mae": float(overall["mae"]),
        "test_rmse": float(overall["rmse"]),
        "test_wape": float(overall["wape"]),
        "best_val_mae": float(config["best_val_mae"]),
        "parameters": int(config["parameter_count"]),
        "latency_per_sample_ms": float(config["latency_per_sample_ms"]),
        "training_seconds": float(efficiency["total_training_seconds"]),
        "wall_seconds": "" if elapsed is None else elapsed,
        "train_actual_missing_rate": float(masks["train_actual_missing_rate"]),
        "val_actual_missing_rate": float(masks["val_actual_missing_rate"]),
        "test_actual_missing_rate": float(masks["test_actual_missing_rate"]),
        "adjacency_sha256": str(config["adjacency_sha256"]),
        "run_dir": str(directory),
    }
    upsert(
        output_root / "external_summary.csv", summary,
        ("dataset", "global_missing_rate", "block_length", "model", "seed"),
    )
    for metric in metrics:
        row = {
            "dataset": dataset,
            "global_missing_rate": rate,
            "block_length": block,
            "model": model,
            "seed": seed,
            "horizon": metric["horizon"],
            "mae": float(metric["mae"]),
            "rmse": float(metric["rmse"]),
            "wape": float(metric["wape"]),
        }
        upsert(
            output_root / "external_horizon_metrics.csv", row,
            ("dataset", "global_missing_rate", "block_length", "model", "seed", "horizon"),
        )


def collect_naive(
    output_root: Path, dataset: str, seed: int, rate: float, block: int, mask: Path
) -> None:
    path = output_root / "naive_raw" / f"{dataset}_{mask.stem}.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        metrics = list(csv.DictReader(handle))
    if len(metrics) != 10:
        raise RuntimeError(f"Expected 10 naive metric rows: {path}")
    for metric in metrics:
        row = {
            "dataset": dataset,
            "global_missing_rate": rate,
            "block_length": block,
            "model": metric["model"],
            "seed": seed,
            "horizon": metric["horizon"],
            "mae": float(metric["mae"]),
            "rmse": float(metric["rmse"]),
            "wape": float(metric["wape"]),
        }
        upsert(
            output_root / "naive_horizon_metrics.csv", row,
            ("dataset", "global_missing_rate", "block_length", "model", "seed", "horizon"),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--global-missing-rates", nargs="+", type=float, default=list(RATES))
    parser.add_argument("--blocks", nargs="+", type=int, default=list(BLOCKS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--affected-node-fraction", type=float, default=0.40)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/v16b_primary"))
    parser.add_argument("--skip-naive", action="store_true")
    args = parser.parse_args()
    if tuple(args.seeds) != SEEDS and len(args.seeds) > 1:
        print("WARNING: non-formal seed subset requested", flush=True)
    if args.epochs != 100 or args.patience != 15:
        raise RuntimeError("Frozen v16A training budget is epochs=100, patience=15")

    total = len(args.models) * len(args.seeds) * len(args.datasets) * len(args.global_missing_rates) * len(args.blocks)
    index = 0
    for model in args.models:
        for seed in args.seeds:
            for dataset in args.datasets:
                for rate in args.global_missing_rates:
                    for block in args.blocks:
                        index += 1
                        mask = mask_path(args.data_root, dataset, seed, rate, block, args.affected_node_fraction)
                        if not mask.exists():
                            run(
                                [
                                    sys.executable, str(SOURCE_DIR / "generate_rate_controlled_masks.py"),
                                    "--dataset", dataset,
                                    "--processed-dir", str(args.data_root / "processed"),
                                    "--output-dir", str(args.data_root / "masks_v2"),
                                    "--global-missing-rate", str(rate),
                                    "--affected-node-fraction", str(args.affected_node_fraction),
                                    "--block-length", str(block),
                                    "--seed", str(seed),
                                ],
                                args.output_root / "logs" / "masks" / f"{dataset}_rate{rate:g}_block{block}_seed{seed}.log",
                            )
                        directory = run_dir(args.output_root, model, dataset, seed, rate, block)
                        complete = all((directory / name).exists() for name in (
                            "history.csv", "validation_metrics.csv", "test_metrics.csv", "config.json",
                            "efficiency.json", "adjacency.npz", "best.pt",
                        ))
                        prefix = f"[{index}/{total}] dataset={dataset} rate={rate:g} block={block} model={model} seed={seed}"
                        elapsed = None
                        if not complete:
                            print(prefix + " started", flush=True)
                            elapsed = run(
                                [
                                    sys.executable, str(SOURCE_DIR / "train_v16b_external.py"),
                                    "--dataset", dataset,
                                    "--processed-dir", str(args.data_root / "processed"),
                                    "--mask-file", str(mask),
                                    "--model", model,
                                    "--output-dir", str(directory.parent),
                                    "--hidden-dim", "64",
                                    "--learning-rate", "0.001",
                                    "--dropout", "0.1",
                                    "--epochs", "100",
                                    "--patience", "15",
                                    "--num-workers", str(args.num_workers),
                                    "--latency-warmup", "5",
                                    "--latency-repetitions", "20",
                                    "--seed", str(seed),
                                    "--independent-test",
                                ],
                                args.output_root / "logs" / model / f"{dataset}_rate{rate:g}_block{block}_seed{seed}.log",
                            )
                            print(prefix + f" finished elapsed={elapsed:.1f}s", flush=True)
                        else:
                            print(prefix + " skip_existing", flush=True)
                        collect_model(
                            args.output_root, model, dataset, seed, rate, block,
                            args.affected_node_fraction, mask, elapsed,
                        )

    if not args.skip_naive:
        naive_index = 0
        naive_total = len(args.seeds) * len(args.datasets) * len(args.global_missing_rates) * len(args.blocks)
        for seed in args.seeds:
            for dataset in args.datasets:
                for rate in args.global_missing_rates:
                    for block in args.blocks:
                        naive_index += 1
                        mask = mask_path(args.data_root, dataset, seed, rate, block, args.affected_node_fraction)
                        output = args.output_root / "naive_raw" / f"{dataset}_{mask.stem}.csv"
                        if not output.exists():
                            run(
                                [
                                    sys.executable, str(SOURCE_DIR / "run_naive_single.py"),
                                    "--dataset", dataset,
                                    "--mask-file", str(mask),
                                    "--processed-dir", str(args.data_root / "processed"),
                                    "--output-dir", str(args.output_root / "naive_raw"),
                                ],
                                args.output_root / "logs" / "naive" / f"{dataset}_rate{rate:g}_block{block}_seed{seed}.log",
                            )
                        collect_naive(args.output_root, dataset, seed, rate, block, mask)
                        print(f"[naive {naive_index}/{naive_total}] collected", flush=True)
    print(f"V16B_PRIMARY_RUN_COMPLETE learning_rows={total} naive_conditions={0 if args.skip_naive else naive_total}", flush=True)


if __name__ == "__main__":
    main()
