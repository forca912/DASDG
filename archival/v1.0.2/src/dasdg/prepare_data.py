"""Prepare PEMS flow data without changing the original observations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def prepare_dataset(raw_path: Path, output_root: Path) -> None:
    dataset = raw_path.stem.upper()
    raw = np.load(raw_path)["data"].astype(np.float32)
    if raw.ndim != 3 or raw.shape[-1] < 1:
        raise ValueError(f"Unexpected shape for {raw_path}: {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"Non-finite value found in {raw_path}")

    n_steps = raw.shape[0]
    train_end = int(n_steps * 0.6)
    val_end = int(n_steps * 0.8)
    train = raw[:train_end]
    mean = train.reshape(-1, raw.shape[-1]).mean(axis=0)
    std = train.reshape(-1, raw.shape[-1]).std(axis=0)
    std = np.maximum(std, 1e-6)

    out = output_root / dataset
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "complete.npz",
        features=raw,
        flow=raw[..., 0],
        split=np.array([train_end, val_end, n_steps], dtype=np.int64),
        mean=mean,
        std=std,
    )
    print(
        f"{dataset}: shape={raw.shape}, splits="
        f"train[0:{train_end}), val[{train_end}:{val_end}), "
        f"test[{val_end}:{n_steps})"
    )
    print(f"saved: {out / 'complete.npz'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    files = sorted(args.raw_dir.glob("PEMS*.npz"))
    if not files:
        raise FileNotFoundError(f"No PEMS npz files found in {args.raw_dir}")
    for path in files:
        prepare_dataset(path, args.output_dir)


if __name__ == "__main__":
    main()
