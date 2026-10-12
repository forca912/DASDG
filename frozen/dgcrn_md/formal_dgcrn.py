"""Formal DGCRN-MD-adapted run for one dataset/seed; no test leakage."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from compat import build_compat, sha256  # noqa: E402
from inputs import causal_inputs, assert_input_invariance  # noqa: E402
from protocol import atomic_checkpoint, environment, write_csv, write_json  # noqa: E402
from vendor.train_v16b_external import build_correlation_adjacency, set_seed  # noqa: E402


HISTORY = 12
HORIZON = 12
SEEDS = (20260808, 20260809, 20260810)


def cfg_from_file() -> dict:
    return json.loads((ROOT / "formal_config.json").read_text(encoding="utf-8"))


class WindowDataset(Dataset):
    def __init__(self, flow, observed, mean, std, start, end):
        self.inputs = causal_inputs(flow, observed, mean, std)
        self.targets = ((flow.astype(np.float32) - mean) / std).astype(np.float32)
        self.starts = np.arange(start, end - HISTORY - HORIZON + 1, dtype=np.int64)
        if len(self.starts) == 0:
            raise ValueError("No complete forecasting windows in split")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        s = int(self.starts[index])
        return (torch.from_numpy(self.inputs[s:s + HISTORY]),
                torch.from_numpy(self.targets[s + HISTORY:s + HISTORY + HORIZON]))


def load_contract(data_root: Path, dataset: str, seed: int, cfg: dict):
    complete = data_root / "processed" / dataset / "complete.npz"
    mask = data_root / "masks_v2" / dataset / "persistent_rate_controlled" / (
        f"seed{seed}_global0.1_nodes0.4_block12.npz")
    if not complete.is_file() or not mask.is_file():
        raise FileNotFoundError(f"Missing complete data or mask: {complete} / {mask}")
    expected_shape = {"PEMS04": (16992, 307), "PEMS08": (17856, 170)}[dataset]
    with np.load(complete, allow_pickle=False) as z:
        flow = z["flow"].astype(np.float32)
        split = [int(x) for x in z["split"].tolist()]
        mean, std = float(z["mean"].flat[0]), float(z["std"].flat[0])
        features = z["features"] if "features" in z else None
    if flow.shape != expected_shape or split != [int(len(flow) * .6), int(len(flow) * .8), len(flow)]:
        raise ValueError(f"Unexpected {dataset} shape/split: {flow.shape}, {split}")
    if not np.isfinite(flow).all() or std <= 0:
        raise ValueError("Non-finite flow or invalid normalization")
    if features is not None and not np.array_equal(flow, features[..., 0]):
        raise ValueError("Flow does not match features[...,0]")
    with np.load(mask, allow_pickle=False) as z:
        if not bool(z["observed_is_one"]) or int(z["seed"]) != seed:
            raise ValueError("Mask metadata mismatch")
        masks = {name: z[name].copy() for name in ("train", "val", "test")}
    for name, (a, b) in {"train": (0, split[0]), "val": (split[0], split[1]),
                         "test": (split[1], split[2])}.items():
        if masks[name].dtype != np.bool_ or masks[name].shape != (b - a, flow.shape[1]):
            raise ValueError(f"Mask shape/dtype mismatch: {name}")
    # The source package already audited these identities in expected_data.json.
    records = json.loads((ROOT / "expected_data.json").read_text(encoding="utf-8"))["legacy_protocol_matches"]
    record = next(r for r in records if r["dataset"] == dataset and r["seed"] == seed)
    if sha256(complete) != record["complete_sha256"] or sha256(mask) != record["mask_sha256"]:
        raise ValueError("Complete data or mask hash differs from audited identity")
    if mean != record["mean"] or std != record["std"]:
        raise ValueError("Frozen normalization differs from audited identity")
    return flow, masks, mean, std, split, {"complete_sha256": sha256(complete),
        "mask_sha256": sha256(mask), "dataset": dataset, "seed": seed,
        "shape": list(flow.shape), "split": split, "mean": mean, "std": std,
        "test_targets_used": False, "graph_source": cfg["graph_source"]}


def make_model(nodes, adjacency, device, compat_dir):
    sys.path.insert(0, str(compat_dir))
    from pf_dgcrn.net import DGCRN
    a = torch.tensor(adjacency, dtype=torch.float32, device=device)
    reverse = a.T / a.T.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return DGCRN(2, nodes, device, predefined_A=[a, reverse], seq_length=12,
                 in_dim=3, out_dim=12, rnn_size=64, node_dim=40,
                 hyperGNN_dim=16, dropout=.1).to(device)


def forecast(model, x, y=None, training=False, batches_seen=0):
    features = x.permute(0, 3, 2, 1).contiguous()
    ycl = torch.zeros((len(x), 3, x.shape[2], HORIZON), dtype=x.dtype, device=x.device)
    if training:
        if y is None:
            raise ValueError("Teacher forcing requires training targets")
        ycl[:, 0] = y.permute(0, 2, 1)
    return model(features, ycl=ycl, batches_seen=batches_seen, task_level=HORIZON).squeeze(-1)


@torch.no_grad()
def future_target_probe(model, x, seed):
    """Evaluation must be invariant to values in the unknown future target."""
    model.eval()
    features = x.permute(0, 3, 2, 1).contiguous()
    zeros = torch.zeros((len(x), 3, x.shape[2], HORIZON), dtype=x.dtype, device=x.device)
    changed = zeros.clone()
    changed[:, 0] = 123456.0
    # The upstream implementation orthogonally initializes a fresh hidden
    # state on every forward, including eval mode. Resetting the RNG isolates
    # target dependence from that stochastic initialization.
    set_seed(seed)
    first = model(features, ycl=zeros, batches_seen=0, task_level=HORIZON)
    set_seed(seed)
    second = model(features, ycl=changed, batches_seen=0, task_level=HORIZON)
    difference = float((first - second).abs().max().cpu())
    if not torch.isfinite(first).all() or not torch.isfinite(second).all() or difference != 0.0:
        raise RuntimeError(f"Future target invariance failed: max_abs_difference={difference}")
    return {"status": "PASS", "max_abs_difference": difference,
            "known_future_mask_or_duration_used": False}


@torch.no_grad()
def evaluate(model, loader, device, mean, std, seed):
    model.eval()
    sums = np.zeros((3, HORIZON), dtype=np.float64)
    count = 0
    for index, (x, y) in enumerate(loader):
        set_seed(seed + index)
        pred = forecast(model, x.to(device), training=False).cpu().numpy() * std + mean
        target = y.numpy() * std + mean
        err = pred - target
        sums[0] += np.abs(err).sum(axis=(0, 2))
        sums[1] += np.square(err).sum(axis=(0, 2))
        sums[2] += np.abs(target).sum(axis=(0, 2))
        count += len(y) * y.shape[2]
    rows = []
    for h in (3, 6, 9, 12):
        rows.append({"horizon": str(h), "mae": float(sums[0, h - 1] / count),
                     "rmse": float(np.sqrt(sums[1, h - 1] / count)),
                     "wape": float(100 * sums[0, h - 1] / max(sums[2, h - 1], 1e-8))})
    rows.append({"horizon": "all", "mae": float(sums.sum(axis=1)[0] / (count * HORIZON)),
                 "rmse": float(np.sqrt(sums.sum(axis=1)[1] / (count * HORIZON))),
                 "wape": float(100 * sums.sum(axis=1)[0] / max(sums.sum(axis=1)[2], 1e-8))})
    return rows


def run(args):
    cfg = cfg_from_file()
    if args.seed not in SEEDS or args.dataset not in cfg["datasets"]:
        raise ValueError("Dataset or seed is outside the frozen matrix")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    set_seed(args.seed)
    flow, masks, mean, std, split, evidence = load_contract(args.data_root, args.dataset, args.seed, cfg)
    invariance = [assert_input_invariance(flow[:split[0]], masks["train"], mean, std, c)
                  for c in (12, split[0] // 2, split[0] - 12)]
    train_flow, val_flow, test_flow = flow[:split[0]], flow[split[0]:split[1]], flow[split[1]:]
    adjacency = build_correlation_adjacency((train_flow - mean) / std, cfg["top_k"], masks["train"])
    adjacency_hash = hashlib.sha256(adjacency.tobytes()).hexdigest()
    train_ds = WindowDataset(train_flow, masks["train"], mean, std, 0, len(train_flow))
    val_ds = WindowDataset(val_flow, masks["val"], mean, std, 0, len(val_flow))
    test_ds = WindowDataset(test_flow, masks["test"], mean, std, 0, len(test_flow))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              generator=generator, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    compat_dir = out / "compat"
    build_compat(ROOT, compat_dir)
    model = make_model(flow.shape[1], adjacency, device, compat_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=cfg["lr_plateau_patience"], min_lr=cfg["min_learning_rate"])
    best_path = out / "best.pt"
    best = float("inf")
    stale = 0
    rows = []
    batches_seen = 0
    probe_x, _ = train_ds[0]
    probe = future_target_probe(model, probe_x[None].to(device), args.seed + 700000)
    write_json(out / "data_contract.json", evidence | {"causal_invariance": invariance,
        "adjacency_sha256": adjacency_hash, "adjacency_shape": list(adjacency.shape),
        "future_target_invariance": probe})
    write_json(out / "run_config.json", cfg | {"dataset": args.dataset, "seed": args.seed,
        "identity": hashlib.sha256(json.dumps(cfg | {"dataset": args.dataset, "seed": args.seed}, sort_keys=True).encode()).hexdigest(),
        "environment": environment(device), "formal_result": True, "test_targets_used": False,
        "compatibility_variant": "DGCRN-MD-adapted", "future_auxiliary": "zero_eval_zero_val_training_target_value_only",
        "adjacency_sha256": adjacency_hash})
    for epoch in range(1, cfg["epochs"] + 1):
        started = time.perf_counter(); model.train(); total = 0.; n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = forecast(model, x, y, training=True, batches_seen=batches_seen)
            loss = torch.nn.functional.l1_loss(pred, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"]); optimizer.step()
            batches_seen += 1; total += float(loss.detach()) * len(x); n += len(x)
        val = evaluate(model, val_loader, device, mean, std, args.seed + 100000 + epoch); score = val[-1]["mae"]
        scheduler.step(score)
        row = {"epoch": epoch, "train_l1_normalized": total / max(n, 1),
               "val_mae_raw": score, "seconds": time.perf_counter() - started,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        rows.append(row); write_csv(out / "history.csv", rows)
        write_json(out / "progress.json", {"status": "RUNNING", "epoch": epoch,
            "epochs": cfg["epochs"], "dataset": args.dataset, "seed": args.seed})
        print(f"FORMAL dataset={args.dataset} seed={args.seed} epoch={epoch}/150 val_MAE={score:.5f}", flush=True)
        if score < best - 1e-5:
            best, stale = score, 0; atomic_checkpoint(best_path, model.state_dict())
        else:
            stale += 1
            if stale >= cfg["early_stopping_patience"]:
                print(f"EARLY_STOP epoch={epoch} best_val_MAE={best:.5f}", flush=True); break
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    write_csv(out / "validation_metrics.csv", evaluate(model, val_loader, device, mean, std, args.seed + 800000))
    write_csv(out / "test_metrics.csv", evaluate(model, test_loader, device, mean, std, args.seed + 900000))
    write_json(out / "progress.json", {"status": "FORMAL_COMPLETE", "dataset": args.dataset,
        "seed": args.seed, "formal_training_completed": True, "test_evaluation_performed": True})
    files = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*"))
             if p.is_file() and p.name != "DONE.json" and p.suffix not in {".tmp"}}
    write_json(out / "DONE.json", {"status": "FORMAL_COMPLETE", "identity": json.loads((out / "run_config.json").read_text())["identity"], "files": files})
    print(f"FORMAL_COMPLETE {args.dataset} seed={args.seed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["PEMS04", "PEMS08"], required=True)
    parser.add_argument("--seed", type=int, choices=list(SEEDS), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    run(parser.parse_args())
