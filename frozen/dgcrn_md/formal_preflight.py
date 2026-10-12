"""Fail-closed server preflight for the frozen DGCRN-MD matrix."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from protocol import load_data, write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Output already contains files; choose a new directory")
    cfg = json.loads((ROOT / "formal_config.json").read_text(encoding="utf-8"))
    if cfg["adapter_name"] != "DGCRN-MD-adapted" or cfg["formal_result"] is not True:
        raise RuntimeError("Formal config identity is not frozen")
    for dataset in cfg["datasets"]:
        for seed in cfg["seeds"]:
            # load_data performs the complete shape, split, mask, normalization,
            # and audited-hash checks without opening a model checkpoint.
            _, _, _, _, _, evidence = load_data(args.data_root.resolve(), dataset, seed,
                                                  {"mask_root": "masks_v2",
                                                   "mask_pattern": "persistent_rate_controlled",
                                                   "global_missing_rate": .1,
                                                   "affected_node_fraction": .4,
                                                   "block_length": 12,
                                                   "training": {"input_steps": 12, "output_steps": 12}},
                                                  fixture=False)
            print(f"DATA_PASS {dataset} seed={seed} complete={evidence['complete_sha256'][:12]} mask={evidence['mask_sha256'][:12]}")
    if args.device == "cuda":
        probe = subprocess.run([sys.executable, "-c",
            "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"],
            check=False, capture_output=True, text=True)
        if probe.returncode:
            raise RuntimeError("CUDA unavailable: " + probe.stderr.strip())
        print("CUDA_PASS " + probe.stdout.strip())
    compile_result = subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / "formal_dgcrn.py")], check=False)
    if compile_result.returncode:
        raise RuntimeError("formal_dgcrn.py syntax check failed")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "PREFLIGHT_PASS.json", {
        "status": "PREFLIGHT_PASS",
        "protocol_id": cfg["protocol_id"],
        "datasets": cfg["datasets"], "seeds": cfg["seeds"],
        "formal_training_started": False, "test_evaluation_performed": False,
    })
    print("PREFLIGHT_PASS formal_training_started=false test_evaluation_performed=false")


if __name__ == "__main__":
    main()
