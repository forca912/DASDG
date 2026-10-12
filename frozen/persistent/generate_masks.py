"""Generate auditable robustness masks for the SoftwareX revision experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from protocol import mask_path, read_json, write_json

MASK_GENERATOR_VERSION = "r1-robustness-mask-v1"


def _load_complete(data_root: Path, dataset: str):
    path = data_root / "processed" / dataset / "complete.npz"
    with np.load(path, allow_pickle=False) as z:
        flow = z["flow"].astype(np.float32)
        split = [int(v) for v in z["split"].tolist()]
    if flow.ndim != 2 or not np.isfinite(flow).all():
        raise ValueError(f"Invalid complete flow: {path}")
    return flow, split


def _select_nodes(flow: np.ndarray, train_end: int, fraction: float, seed: int,
                  pattern: str) -> tuple[np.ndarray, dict[str, object]]:
    nodes = flow.shape[1]
    count = max(1, int(round(nodes * fraction)))
    rng = np.random.default_rng(seed)
    if pattern != "spatial_cluster":
        selected = np.sort(rng.choice(nodes, size=count, replace=False))
        return selected, {"node_selection": "seeded_uniform", "cluster_basis": None}

    # No physical coordinates are distributed with the frozen release data.
    # Use training-only absolute correlation as an auditable spatial proxy.
    train = flow[:train_end].astype(np.float64)
    corr = np.corrcoef(train, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
    anchor = int(rng.integers(nodes))
    scores = corr[anchor].copy()
    scores[anchor] = np.inf
    order = np.argsort(-scores, kind="stable")
    # The anchor already ranks first because its score is infinite. Adding it
    # again would duplicate a node and reduce the actual affected-node count.
    selected = np.sort(order[:count])
    return selected, {"node_selection": "training_absolute_correlation", "cluster_anchor": anchor,
                      "cluster_basis": "training_only_absolute_correlation_proxy"}


def _free_intervals(occupied: np.ndarray, length: int) -> np.ndarray:
    if length <= 0 or length > len(occupied):
        return np.empty(0, dtype=np.int64)
    free = (~occupied).astype(np.int8)
    run = np.convolve(free, np.ones(length, dtype=np.int8), mode="valid")
    return np.flatnonzero(run == length)


def _node_mask(length: int, missing_count: int, block_length: int, pattern: str,
               rng: np.random.Generator) -> np.ndarray:
    missing = np.zeros(length, dtype=bool)
    remaining = int(missing_count)
    while remaining > 0:
        if pattern == "irregular_duration":
            low = max(2, block_length // 2)
            high = max(low, 2 * block_length)
            duration = int(rng.integers(low, high + 1))
        else:
            duration = int(block_length)
        duration = min(duration, remaining, length)
        starts = _free_intervals(missing, duration)
        if len(starts) == 0:
            # Deterministic fallback: complete the quota in the longest free run.
            free = (~missing).astype(np.int8)
            changes = np.diff(np.r_[0, free, 0])
            begins = np.flatnonzero(changes == 1)
            ends = np.flatnonzero(changes == -1)
            if len(begins) == 0:
                raise RuntimeError("Cannot place the requested missing quota")
            idx = int(np.argmax(ends - begins))
            start = int(begins[idx])
            duration = min(duration, int(ends[idx] - begins[idx]))
        else:
            start = int(starts[rng.integers(len(starts))])
        missing[start:start + duration] = True
        remaining -= duration
    return ~missing


def _make_split(length: int, nodes: int, selected: np.ndarray, rate: float,
                block_length: int, pattern: str, rng: np.random.Generator) -> np.ndarray:
    observed = np.ones((length, nodes), dtype=bool)
    total = int(round(length * nodes * rate))
    quotas = np.full(len(selected), total // len(selected), dtype=np.int64)
    quotas[: total % len(selected)] += 1
    for node, quota in zip(selected, quotas):
        observed[:, int(node)] = _node_mask(length, int(quota), block_length, pattern, rng)
    return observed


def generate_one(data_root: Path, dataset: str, seed: int, config: dict) -> dict[str, object]:
    flow, split = _load_complete(data_root, dataset)
    train_end, val_end, total = split
    pattern = config.get("mask_pattern", "regular")
    if pattern not in {"regular", "irregular_duration", "spatial_cluster"}:
        raise ValueError(f"Unsupported mask pattern: {pattern}")
    rate = float(config["global_missing_rate"])
    affected_fraction = float(config["affected_node_fraction"])
    block_length = int(config["block_length"])
    if not 0.0 < rate < 1.0:
        raise ValueError("global_missing_rate must be in (0, 1)")
    if not 0.0 < affected_fraction <= 1.0:
        raise ValueError("affected_node_fraction must be in (0, 1]")
    if block_length < 1:
        raise ValueError("block_length must be positive")
    selected, selection_meta = _select_nodes(
        flow, train_end, affected_fraction, seed, pattern
    )
    expected_count = max(1, int(round(flow.shape[1] * affected_fraction)))
    if len(selected) != expected_count or len(np.unique(selected)) != expected_count:
        raise ValueError("Selected nodes must be unique and match the requested count")
    generator_version = ("r1-robustness-mask-v2-spatial-unique"
                         if pattern == "spatial_cluster" else MASK_GENERATOR_VERSION)
    rng = np.random.default_rng(seed + 1000003 * (list(config.get("datasets", [])).index(dataset) + 1))
    lengths = (train_end, val_end - train_end, total - val_end)
    masks = {}
    start = 0
    for name, length in zip(("train", "val", "test"), lengths):
        masks[name] = _make_split(
            length, flow.shape[1], selected, rate, block_length, pattern, rng
        )
        start += length
    path = mask_path(data_root, dataset, seed, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **masks, seed=np.int64(seed), block_length=np.int64(config["block_length"]),
        intended_global_missing_rate=np.float64(config["global_missing_rate"]),
        affected_node_fraction=np.float64(config["affected_node_fraction"]),
        mask_pattern=np.asarray(pattern), mask_generator_version=np.asarray(generator_version),
        selected_nodes=selected.astype(np.int64), observed_is_one=np.asarray(True),
        **{k: np.asarray(v) for k, v in selection_meta.items() if v is not None},
    )
    manifest = {
        "dataset": dataset, "seed": seed, "mask_path": str(path),
        "mask_generator_version": generator_version, "mask_pattern": pattern,
        "global_missing_rate": rate,
        "affected_node_fraction": affected_fraction,
        "block_length": block_length,
        "selected_nodes": selected.tolist(), **selection_meta,
        "split_lengths": dict(zip(("train", "val", "test"), lengths)),
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=["PEMS03", "PEMS04", "PEMS08"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    datasets = [args.dataset] if args.dataset else config["datasets"]
    seeds = [args.seed] if args.seed is not None else config["seeds"]
    manifests = []
    for seed in seeds:
        for dataset in datasets:
            manifests.append(generate_one(args.data_root.resolve(), dataset, int(seed), config))
    output = args.config.parent / f"{args.config.stem}_mask_manifest.json"
    write_json(output, {"generator_version": MASK_GENERATOR_VERSION, "masks": manifests})
    for item in manifests:
        print(f"MASK_GENERATED dataset={item['dataset']} seed={item['seed']} "
              f"pattern={item['mask_pattern']} path={item['mask_path']}", flush=True)
    print(f"MASK_MANIFEST {output}", flush=True)


if __name__ == "__main__":
    main()
