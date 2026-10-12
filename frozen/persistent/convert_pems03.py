"""Validate and convert the frozen PEMS03 archive without modifying its source."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np

EXPECTED_BYTES = 15_800_415
EXPECTED_GIT_BLOB = "b4487154fc9611a3bf9856b0aad34efa610ea8ae"
EXPECTED_SHAPE = (26208, 358, 1)
EXPECTED_MIN = 0.0
EXPECTED_MAX = 1852.0


def _digests(raw: bytes) -> tuple[str, str]:
    git_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    return git_blob, hashlib.sha256(raw).hexdigest()


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def convert(source: Path, data_root: Path) -> tuple[Path, Path, dict[str, object]]:
    source = source.expanduser().resolve(strict=True)
    data_root = data_root.expanduser().resolve()
    out_dir = data_root / "processed" / "PEMS03"
    complete_path = out_dir / "complete.npz"
    manifest_path = out_dir / "source_manifest.json"
    if complete_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing PEMS03 output: {out_dir}")

    raw = source.read_bytes()
    git_blob, source_sha256 = _digests(raw)
    if len(raw) != EXPECTED_BYTES:
        raise ValueError(f"Source byte count mismatch: expected {EXPECTED_BYTES}, found {len(raw)}")
    if git_blob != EXPECTED_GIT_BLOB:
        raise ValueError(f"Source Git blob SHA-1 mismatch: {git_blob}")

    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if "data" not in archive.files:
            raise ValueError(f"NPZ must contain key 'data'; found keys={archive.files}")
        data = archive["data"]
        keys = list(archive.files)
    if data.shape != EXPECTED_SHAPE:
        raise ValueError(f"PEMS03 shape mismatch: expected {EXPECTED_SHAPE}, found {data.shape}")
    if data.dtype != np.float64:
        raise ValueError(f"PEMS03 dtype mismatch: expected float64, found {data.dtype}")
    if not np.isfinite(data).all():
        raise ValueError("PEMS03 contains NaN or infinite values")
    data_min, data_max = float(data.min()), float(data.max())
    if data_min != EXPECTED_MIN or data_max != EXPECTED_MAX or np.any(data < 0):
        raise ValueError(f"PEMS03 value range mismatch: expected [{EXPECTED_MIN}, {EXPECTED_MAX}], found [{data_min}, {data_max}]")

    flow = data[..., 0].astype(np.float32)
    split = [int(len(flow) * 0.6), int(len(flow) * 0.8), len(flow)]
    train = flow[:split[0]]
    mean = np.asarray([train.mean(dtype=np.float64)], dtype=np.float64)
    std = np.asarray([max(train.std(dtype=np.float64), 1e-6)], dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or float(std[0]) <= 0:
        raise ValueError("Invalid training-only normalization statistics")

    if complete_path.exists() or manifest_path.exists():
        raise FileExistsError(f"PEMS03 output appeared during conversion: {out_dir}")
    _atomic_npz(complete_path, {
        "flow": flow,
        "split": np.asarray(split, dtype=np.int64),
        "mean": mean,
        "std": std,
    })
    complete_sha256 = hashlib.sha256(complete_path.read_bytes()).hexdigest()
    manifest: dict[str, object] = {
        "dataset": "PEMS03",
        "source_file_name": source.name,
        "source_bytes": len(raw),
        "source_git_blob_sha1": git_blob,
        "source_sha256": source_sha256,
        "source_npz_keys": keys,
        "source_key_used": "data",
        "source_shape": list(data.shape),
        "source_dtype": str(data.dtype),
        "source_min": data_min,
        "source_max": data_max,
        "flow_shape": list(flow.shape),
        "flow_dtype": str(flow.dtype),
        "split_protocol": "chronological_60_20_20",
        "split_boundaries": split,
        "normalization": "float64 mean/std computed from float32 flow training segment only; std floor=1e-6",
        "training_mean": float(mean[0]),
        "training_std": float(std[0]),
        "complete_npz_sha256": complete_sha256,
        "raw_source_copied": False,
    }
    try:
        _atomic_json(manifest_path, manifest)
    except BaseException:
        # Keep incomplete conversion artifacts visible; never silently delete evidence.
        raise
    return complete_path, manifest_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Validated server-side PEMS03.npz")
    parser.add_argument("--data-root", type=Path, required=True, help="Experiment data root, not the raw-data directory")
    args = parser.parse_args()
    complete, manifest_path, manifest = convert(args.source, args.data_root)
    print(f"SOURCE_CHECK_PASS git_blob={manifest['source_git_blob_sha1']} bytes={manifest['source_bytes']}", flush=True)
    print(f"CONVERSION_PASS complete={complete} manifest={manifest_path}", flush=True)
    print(f"shape={manifest['flow_shape']} split={manifest['split_boundaries']} mean={manifest['training_mean']:.12g} std={manifest['training_std']:.12g}", flush=True)


if __name__ == "__main__":
    main()
