"""Run the locked DASDG release-reproduction conditions."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


LOCKED_VARIANT = "graph_no_reliability_channel"
DEFAULT_DATASETS = ("PEMS04", "PEMS08")
DEFAULT_SEEDS = (20260808, 20260809, 20260810)
HORIZONS = ("3", "6", "9", "12", "all")
SOURCE_DIR = Path(__file__).resolve().parent


def mask_path(data_root: Path, dataset: str, seed: int, rate: float, block: int, fraction: float) -> Path:
    return data_root / "masks_v2" / dataset / "persistent_rate_controlled" / (
        f"seed{seed}_global{rate:g}_nodes{fraction:g}_block{block}.npz"
    )


def run_dir(output_root: Path, dataset: str, seed: int, rate: float, block: int) -> Path:
    return output_root / f"global{rate:g}" / f"block{block}" / (
        f"{dataset}_{LOCKED_VARIANT}_h64_g32_k20_seed{seed}"
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(output_root: Path, dataset: str, seed: int, rate: float, block: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    directory = run_dir(output_root, dataset, seed, rate, block)
    required = (
        "best.pt", "history.csv", "test_metrics.csv", "config.json", "efficiency.json",
        "model_diagnostics.json", "reconstruction_metrics.json",
    )
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise RuntimeError(f"Incomplete DASDG run {directory}: {missing}")
    metrics = read_rows(directory / "test_metrics.csv")
    if {row["horizon"] for row in metrics} != set(HORIZONS):
        raise RuntimeError(f"Horizon coverage mismatch: {directory / 'test_metrics.csv'}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8-sig"))
    efficiency = json.loads((directory / "efficiency.json").read_text(encoding="utf-8-sig"))
    if config.get("variant") != LOCKED_VARIANT:
        raise RuntimeError(f"Variant mismatch in {directory / 'config.json'}")
    overall = next(row for row in metrics if row["horizon"] == "all")
    summary = {
        "dataset": dataset,
        "global_missing_rate": rate,
        "block_length": block,
        "variant": LOCKED_VARIANT,
        "seed": seed,
        "test_mae": float(overall["mae"]),
        "test_rmse": float(overall["rmse"]),
        "test_wape": float(overall["wape"]),
        "best_val_mae": float(config["best_val_mae"]),
        "parameters": int(config["parameter_count"]),
        "latency_per_sample_ms": float(config["latency_per_sample_ms"]),
        "training_seconds": float(efficiency["total_training_seconds"]),
        "run_dir": str(directory),
    }
    horizon_rows = [
        {
            "dataset": dataset,
            "global_missing_rate": rate,
            "block_length": block,
            "variant": LOCKED_VARIANT,
            "seed": seed,
            "horizon": row["horizon"],
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "wape": float(row["wape"]),
        }
        for row in metrics
    ]
    return summary, horizon_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--global-missing-rate", type=float, default=0.10)
    parser.add_argument("--block-length", type=int, default=12)
    parser.add_argument("--affected-node-fraction", type=float, default=0.40)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/dasdg_repro_v1_0"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.epochs != 150 or args.patience != 25:
        raise RuntimeError("The locked release budget is epochs=150 and patience=25")

    total = len(args.datasets) * len(args.seeds)
    summaries: list[dict[str, object]] = []
    horizons: list[dict[str, object]] = []
    index = 0
    for seed in args.seeds:
        for dataset in args.datasets:
            index += 1
            mask = mask_path(
                args.data_root, dataset, seed, args.global_missing_rate,
                args.block_length, args.affected_node_fraction,
            )
            if not mask.exists():
                raise FileNotFoundError(f"Missing frozen mask: {mask}")
            directory = run_dir(
                args.output_root, dataset, seed, args.global_missing_rate, args.block_length
            )
            complete = all((directory / name).exists() for name in (
                "best.pt", "history.csv", "test_metrics.csv", "config.json", "efficiency.json",
                "model_diagnostics.json", "reconstruction_metrics.json",
            ))
            prefix = f"[{index}/{total}] dataset={dataset} variant={LOCKED_VARIANT} seed={seed}"
            if not complete:
                directory.parent.mkdir(parents=True, exist_ok=True)
                log_path = args.output_root / "logs" / f"{dataset}_seed{seed}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable, str(SOURCE_DIR / "train_fdr_stgnn.py"),
                    "--dataset", dataset,
                    "--processed-dir", str(args.data_root / "processed"),
                    "--mask-file", str(mask),
                    "--variant", LOCKED_VARIANT,
                    "--epochs", "150",
                    "--patience", "25",
                    "--num-workers", str(args.num_workers),
                    "--latency-repetitions", "50",
                    "--seed", str(seed),
                    "--output-dir", str(directory.parent),
                ]
                print(prefix + " started", flush=True)
                print("$ " + " ".join(command), flush=True)
                started = time.perf_counter()
                with log_path.open("w", encoding="utf-8") as handle:
                    completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
                if completed.returncode != 0:
                    raise RuntimeError(f"Training failed; inspect {log_path}")
                print(prefix + f" finished elapsed={time.perf_counter() - started:.1f}s", flush=True)
            else:
                print(prefix + " skip_existing", flush=True)
            summary, horizon_rows = collect(
                args.output_root, dataset, seed, args.global_missing_rate, args.block_length
            )
            summaries.append(summary)
            horizons.extend(horizon_rows)

    write_rows(args.output_root / "dasdg_summary.csv", summaries)
    write_rows(args.output_root / "dasdg_horizon_metrics.csv", horizons)
    print(f"DASDG_REPRODUCTION_COMPLETE learning_rows={len(summaries)}", flush=True)


if __name__ == "__main__":
    main()
