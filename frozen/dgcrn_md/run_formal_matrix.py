"""Run the frozen six-job DGCRN-MD matrix sequentially."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    package = args.package.resolve()
    cfg = json.loads((package / "formal_config.json").read_text(encoding="utf-8"))
    if args.output.exists() and not (args.output / "PREFLIGHT_PASS.json").is_file():
        raise RuntimeError("Run formal_preflight.py first; output has no PREFLIGHT_PASS.json")
    for dataset in cfg["datasets"]:
        for seed in cfg["seeds"]:
            job = args.output / "runs" / dataset / f"seed{seed}"
            done = job / "DONE.json"
            if done.is_file():
                print(f"SKIP_DONE {dataset} seed={seed}", flush=True)
                continue
            if job.exists() and any(job.iterdir()):
                raise RuntimeError(f"Partial job exists; preserve it and choose a new output: {job}")
            command = [sys.executable, "-u", str(package / "formal_dgcrn.py"),
                       "--dataset", dataset, "--seed", str(seed),
                       "--data-root", str(args.data_root.resolve()),
                       "--output", str(job), "--device", args.device]
            print("JOB_START", dataset, seed, flush=True)
            subprocess.run(command, check=True)
            print("JOB_DONE", dataset, seed, flush=True)
    print("FORMAL_MATRIX_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
