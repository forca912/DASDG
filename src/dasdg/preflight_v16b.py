"""Fail-closed preflight for the v16B independent-test experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DATASETS = ("PEMS04", "PEMS08")
RATES = (0.05, 0.10, 0.15)
BLOCKS = (6, 12, 24)
SEEDS = tuple(range(20260808, 20260814))
SOURCE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=SOURCE_DIR)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--mask-dir", type=Path, default=Path("data/masks_v2"))
    parser.add_argument("--affected-node-fraction", type=float, default=0.40)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    required_scripts = ("run_v16b_primary.py", "train_v16b_external.py", "run_naive_single.py", "generate_rate_controlled_masks.py")
    missing_scripts = [name for name in required_scripts if not (args.package_root / name).exists()]
    if missing_scripts:
        raise RuntimeError(f"Missing scripts: {missing_scripts}")

    report: dict[str, object] = {"package_root": str(args.package_root.resolve()), "datasets": {}}
    for dataset in DATASETS:
        complete_path = args.processed_dir / dataset / "complete.npz"
        if not complete_path.exists():
            raise FileNotFoundError(complete_path)
        complete = np.load(complete_path)
        if "flow" not in complete or "split" not in complete:
            raise RuntimeError(f"{complete_path} must contain flow and split")
        flow = complete["flow"]
        split = [int(value) for value in complete["split"]]
        if flow.ndim != 2 or split != sorted(split) or split[-1] != len(flow):
            raise RuntimeError(f"Invalid {dataset} shape/split: shape={flow.shape}, split={split}")
        report["datasets"][dataset] = {"shape": list(flow.shape), "split": split}

    missing_masks = []
    checked_masks = 0
    for dataset in DATASETS:
        for rate in RATES:
            for block in BLOCKS:
                for seed in SEEDS:
                    path = args.mask_dir / dataset / "persistent_rate_controlled" / f"seed{seed}_global{rate:g}_nodes{args.affected_node_fraction:g}_block{block}.npz"
                    if not path.exists():
                        missing_masks.append(str(path))
                        continue
                    checked_masks += 1
                    mask = np.load(path)
                    for split in ("train", "val", "test"):
                        if split not in mask:
                            raise RuntimeError(f"{path} missing {split} mask")
                        if mask[split].dtype != np.bool_:
                            raise RuntimeError(f"{path}:{split} must be boolean")
                    actual = float(mask["test_actual_missing_rate"])
                    if abs(actual - rate) > 0.005:
                        raise RuntimeError(f"{path} test missing-rate error {abs(actual-rate):.6f} > 0.005")
    report["mask_files_checked"] = checked_masks
    report["mask_files_expected"] = len(DATASETS) * len(RATES) * len(BLOCKS) * len(SEEDS)
    report["missing_mask_count"] = len(missing_masks)
    if missing_masks:
        report["missing_masks_first"] = missing_masks[:10]
        raise RuntimeError(json.dumps(report, indent=2))

    try:
        import torch
        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:  # pragma: no cover
        report["torch_check_error"] = repr(exc)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
