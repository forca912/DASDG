"""Run the preregistered independent test-set confirmation experiment."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


VARIANTS = (
    "graph_base",
    "graph_no_reliability_channel",
    "graph_mask_only",
)


def run_checked(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def read_overall(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["horizon"] == "all":
                return {name: float(row[name]) for name in ("mae", "rmse", "wape")}
    raise RuntimeError(f"Missing overall row in {path}")


def upsert_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    key_names = ("dataset", "global_missing_rate", "block_length", "variant", "seed")
    key = tuple(str(row[name]) for name in key_names)
    retained = [
        item for item in existing if tuple(item[name] for name in key_names) != key
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(retained)
        writer.writerow(row)


def collect_row(
    dataset: str,
    rate: float,
    node_fraction: float,
    block: int,
    variant: str,
    seed: int,
    mask_file: Path,
    run_dir: Path,
    wall_seconds: float | None,
) -> dict[str, object]:
    required = [
        "test_metrics.csv",
        "efficiency.json",
        "config.json",
        "model_diagnostics.json",
        "history.csv",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing result files in {run_dir}: {missing}")
    metrics = read_overall(run_dir / "test_metrics.csv")
    efficiency = json.loads((run_dir / "efficiency.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((run_dir / "model_diagnostics.json").read_text(encoding="utf-8"))
    masks = np.load(mask_file)
    return {
        "dataset": dataset,
        "global_missing_rate": rate,
        "affected_node_fraction": node_fraction,
        "block_length": block,
        "variant": variant,
        "seed": seed,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "wape": metrics["wape"],
        "best_val_mae": float(config["best_val_mae"]),
        "parameters": int(efficiency["parameter_count"]),
        "latency_per_sample_ms": float(efficiency["latency_per_sample_ms"]),
        "latency_p90_ms": float(efficiency["latency_p90_ms"]),
        "peak_inference_memory_mb": float(efficiency["peak_inference_memory_mb"]),
        "training_seconds": float(efficiency["total_training_seconds"]),
        "wall_seconds": wall_seconds if wall_seconds is not None else "",
        "train_actual_missing_rate": float(masks["train_actual_missing_rate"]),
        "val_actual_missing_rate": float(masks["val_actual_missing_rate"]),
        "test_actual_missing_rate": float(masks["test_actual_missing_rate"]),
        "mean_missing_duration_channel": float(
            diagnostics.get("mean_missing_duration_channel", 0.0)
        ),
        "mean_missing_reliability_channel": float(
            diagnostics.get("mean_missing_reliability_channel", 0.0)
        ),
        "result_dir": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20260808, 20260814)))
    parser.add_argument(
        "--datasets", nargs="+", choices=["PEMS04", "PEMS08"], default=["PEMS04", "PEMS08"]
    )
    parser.add_argument("--global-missing-rates", type=float, nargs="+", default=[0.05, 0.10, 0.15])
    parser.add_argument("--affected-node-fraction", type=float, default=0.40)
    parser.add_argument("--blocks", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latency-repetitions", type=int, default=50)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/fdr_representation_v15_confirm")
    )
    args = parser.parse_args()
    if not args.seeds or not args.blocks or not args.global_missing_rates:
        raise ValueError("Seeds, blocks, and rates must be non-empty")
    if any(block <= 0 for block in args.blocks):
        raise ValueError("Block lengths must be positive")
    if not 0 < args.affected_node_fraction <= 1:
        raise ValueError("Affected node fraction must be in (0, 1]")
    if any(not 0 < rate < args.affected_node_fraction for rate in args.global_missing_rates):
        raise ValueError("Each global missing rate must be below affected node fraction")

    summary_path = args.output_root / "summary.csv"
    total = len(args.seeds) * len(args.datasets) * len(args.global_missing_rates) * len(args.blocks) * len(VARIANTS)
    completed = 0
    for seed in args.seeds:
        for dataset in args.datasets:
            for rate in args.global_missing_rates:
                for block in args.blocks:
                    mask_file = (
                        Path("data/masks_v2")
                        / dataset
                        / "persistent_rate_controlled"
                        / f"seed{seed}_global{rate:g}_nodes{args.affected_node_fraction:g}_block{block}.npz"
                    )
                    if not mask_file.exists():
                        run_checked(
                            [
                                sys.executable,
                                "generate_rate_controlled_masks.py",
                                "--dataset", dataset,
                                "--global-missing-rate", str(rate),
                                "--affected-node-fraction", str(args.affected_node_fraction),
                                "--block-length", str(block),
                                "--seed", str(seed),
                            ]
                        )
                    condition_root = args.output_root / f"global{rate:g}" / f"block{block}"
                    for variant in VARIANTS:
                        completed += 1
                        run_name = f"{dataset}_{variant}_h64_g32_k20_seed{seed}"
                        run_dir = condition_root / run_name
                        prefix = f"[{completed}/{total}] {dataset} rate={rate:g} block={block} variant={variant} seed={seed}"
                        if (run_dir / "test_metrics.csv").exists():
                            upsert_row(summary_path, collect_row(dataset, rate, args.affected_node_fraction, block, variant, seed, mask_file, run_dir, None))
                            print(f"{prefix} skip_existing", flush=True)
                            continue
                        log_path = args.output_root / "logs" / dataset / f"global{rate:g}" / f"block{block}" / f"{variant}_seed{seed}.log"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        command = [
                            sys.executable,
                            "train_fdr_stgnn.py",
                            "--dataset", dataset,
                            "--mask-file", str(mask_file),
                            "--variant", variant,
                            "--epochs", str(args.epochs),
                            "--patience", str(args.patience),
                            "--num-workers", str(args.num_workers),
                            "--latency-repetitions", str(args.latency_repetitions),
                            "--seed", str(seed),
                            "--output-dir", str(condition_root),
                        ]
                        print(f"{prefix} started", flush=True)
                        started = time.perf_counter()
                        with log_path.open("w", encoding="utf-8") as handle:
                            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
                        elapsed = time.perf_counter() - started
                        upsert_row(summary_path, collect_row(dataset, rate, args.affected_node_fraction, block, variant, seed, mask_file, run_dir, elapsed))
                        print(f"{prefix} finished elapsed={elapsed:.1f}s", flush=True)
    print(f"independent confirmation completed: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
