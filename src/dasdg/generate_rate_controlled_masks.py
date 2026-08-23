"""Generate deterministic persistent masks with a controlled global missing rate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def persistent_mask(
    rng: np.random.Generator,
    length: int,
    nodes: int,
    global_missing_rate: float,
    affected_node_fraction: float,
    block_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < global_missing_rate < affected_node_fraction <= 1.0:
        raise ValueError(
            "Require 0 < global_missing_rate < affected_node_fraction <= 1"
        )
    observed = np.ones((length, nodes), dtype=bool)
    affected_count = max(1, int(round(nodes * affected_node_fraction)))
    affected_nodes = np.sort(
        rng.choice(nodes, size=affected_count, replace=False).astype(np.int64)
    )

    target_missing_per_node = global_missing_rate * length * nodes / affected_count
    blocks_per_node = int(round(target_missing_per_node / block_length))
    candidate_starts = np.arange(0, length - block_length + 1, block_length)
    blocks_per_node = min(max(blocks_per_node, 1), len(candidate_starts))
    for node in affected_nodes:
        starts = rng.choice(candidate_starts, size=blocks_per_node, replace=False)
        for start in starts:
            observed[start : start + block_length, node] = False
    return observed, affected_nodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["PEMS04", "PEMS08"])
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/masks_v2"))
    parser.add_argument("--global-missing-rate", type=float, default=0.10)
    parser.add_argument("--affected-node-fraction", type=float, default=0.40)
    parser.add_argument("--block-length", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    if args.block_length <= 0:
        raise ValueError("--block-length must be positive")

    complete = np.load(args.processed_dir / args.dataset / "complete.npz")
    total = int(complete["features"].shape[0])
    nodes = int(complete["features"].shape[1])
    bounds = complete["split"]
    rng = np.random.default_rng(args.seed)
    masks: dict[str, np.ndarray] = {}
    affected: dict[str, np.ndarray] = {}
    actual_rates: dict[str, float] = {}
    start = 0
    for name, end_value in zip(("train", "val", "test"), bounds):
        end = int(end_value)
        mask, affected_nodes = persistent_mask(
            rng,
            end - start,
            nodes,
            args.global_missing_rate,
            args.affected_node_fraction,
            args.block_length,
        )
        masks[name] = mask
        affected[name] = affected_nodes
        actual_rates[name] = float(1.0 - mask.mean())
        print(
            f"{name}: missing={actual_rates[name]:.6f}, observed={mask.mean():.6f}, "
            f"affected_nodes={len(affected_nodes)}, shape={mask.shape}"
        )
        start = end
    if start != total:
        raise RuntimeError(f"Split boundary mismatch: final={start}, total={total}")

    output_dir = args.output_dir / args.dataset / "persistent_rate_controlled"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"seed{args.seed}_global{args.global_missing_rate:g}_"
        f"nodes{args.affected_node_fraction:g}_block{args.block_length}.npz"
    )
    np.savez_compressed(
        path,
        **masks,
        train_affected_nodes=affected["train"],
        val_affected_nodes=affected["val"],
        test_affected_nodes=affected["test"],
        intended_global_missing_rate=np.array(args.global_missing_rate),
        affected_node_fraction=np.array(args.affected_node_fraction),
        block_length=np.array(args.block_length),
        seed=np.array(args.seed),
        train_actual_missing_rate=np.array(actual_rates["train"]),
        val_actual_missing_rate=np.array(actual_rates["val"]),
        test_actual_missing_rate=np.array(actual_rates["test"]),
        observed_is_one=np.array(True),
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
