"""Fail-closed data registration checks for a new, local dataset directory.

This validates the frozen persistent engine's input contract without importing
Torch or silently converting arbitrary data into a benchmark condition.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


class ContractError(ValueError):
    pass


EXPECTED = {"PEMS03": (26208, 358), "PEMS04": (16992, 307), "PEMS08": (17856, 170)}


def validate_complete(path, dataset, fixture=False):
    path = Path(path).resolve()
    if dataset not in EXPECTED:
        raise ContractError(f"Unsupported frozen dataset name: {dataset}")
    if not path.is_file() or path.name != "complete.npz":
        raise ContractError("Expected a file named complete.npz")
    try:
        with np.load(path, allow_pickle=False) as z:
            required = {"flow", "split", "mean", "std"}
            missing = required - set(z.files)
            if missing:
                raise ContractError(f"Missing arrays: {sorted(missing)}")
            flow = z["flow"]
            split = z["split"].tolist()
            mean = float(z["mean"].flat[0])
            std = float(z["std"].flat[0])
            if flow.ndim != 2 or not np.issubdtype(flow.dtype, np.number) or not np.isfinite(flow).all():
                raise ContractError("flow must be finite numeric 2-D")
            if not fixture and tuple(flow.shape) != EXPECTED[dataset]:
                raise ContractError(f"{dataset} requires shape {EXPECTED[dataset]}, found {tuple(flow.shape)}")
            if len(split) != 3 or any(type(v) is not int for v in split) or not (0 < split[0] < split[1] < split[2] == len(flow)):
                raise ContractError("split must be chronological integer 60/20/20 boundaries")
            if not fixture and split != [int(len(flow) * .6), int(len(flow) * .8), len(flow)]:
                raise ContractError("split differs from the frozen chronological 60/20/20 protocol")
            if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
                raise ContractError("mean/std must be finite and std > 0")
            return {"status": "PASS", "dataset": dataset, "path": str(path),
                    "shape": list(flow.shape), "split": split, "dtype": str(flow.dtype),
                    "mean": mean, "std": std, "fixture_only": fixture,
                    "raw_data_hash_recomputed": False,
                    "note": "Input contract only; this does not authorize a formal benchmark claim."}
    except ContractError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise ContractError(f"Invalid NPZ contract: {exc}") from exc


def validate_mask(path, split, nodes, input_steps=12, output_steps=12):
    path = Path(path).resolve()
    if not path.is_file() or path.suffix != ".npz":
        raise ContractError("Expected a mask NPZ file")
    with np.load(path, allow_pickle=False) as z:
        result = {"status": "PASS", "path": str(path), "splits": {},
                  "metadata_checked": []}
        for key, expected_len in zip(("train", "val", "test"),
                                     (split[0], split[1] - split[0], split[2] - split[1])):
            if key not in z:
                raise ContractError(f"Mask is missing {key}")
            arr = z[key]
            if arr.dtype != np.bool_ or arr.shape != (expected_len, nodes):
                raise ContractError(f"{key} mask must have shape {(expected_len, nodes)} and bool dtype")
            if expected_len < input_steps + output_steps:
                raise ContractError(f"{key} split is too short for forecasting windows")
            result["splits"][key] = {"shape": list(arr.shape), "realized_missing_rate": float(1.0 - arr.mean())}
        for key in ("seed", "block_length", "intended_global_missing_rate", "affected_node_fraction", "observed_is_one"):
            if key in z:
                result["metadata_checked"].append(key)
        if "observed_is_one" not in z or not bool(z["observed_is_one"]):
            raise ContractError("Mask must declare observed_is_one=True")
        return result
