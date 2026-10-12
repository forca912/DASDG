"""Plan, generate, run, audit, and bundle the R1.4 robustness suite.

The suite is deliberately sequential.  A failed condition stops the command;
completed conditions remain verifiable and can be resumed by rerunning the same
command after the cause has been fixed.  Each condition has its own config,
mask root, and output directory so that sensitivity and pattern experiments
cannot silently reuse one another's artifacts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from generate_masks import generate_one
from protocol import ROOT, read_json, write_json


def _slug(value: object) -> str:
    text = str(value)
    return text.replace(".", "p").replace("%", "pct").replace("/", "-")


def _condition_specs(base: dict, suite: str) -> list[dict]:
    if suite == "sensitivity":
        specs = []
        for rate in (0.05, 0.10, 0.20):
            for block in (6, 12, 24):
                specs.append({
                    "condition_id": f"rate{_slug(rate)}_block{block}",
                    "global_missing_rate": rate,
                    "block_length": block,
                    "mask_pattern": "regular",
                    "models": ["dasdg"],
                })
        return specs
    if suite == "patterns":
        return [
            {"condition_id": pattern, "global_missing_rate": 0.10,
             "block_length": 12, "mask_pattern": pattern,
             "models": ["dasdg", "gwn_duration"]}
            for pattern in ("regular", "irregular_duration", "spatial_cluster")
        ]
    raise ValueError(f"Unknown suite: {suite}")


def _build_config(base: dict, spec: dict, suite: str) -> dict:
    config = json.loads(json.dumps(base))
    config["protocol"] = f"softwarex-r1-robustness-v1-{suite}-{spec['condition_id']}"
    config["global_missing_rate"] = spec["global_missing_rate"]
    config["block_length"] = spec["block_length"]
    config["mask_pattern"] = spec["mask_pattern"]
    config["mask_root"] = str(
        Path("masks_r1_robustness") / suite / spec["condition_id"]
    ).replace("\\", "/")
    config["stages"] = {"robustness": spec["models"]}
    config["suite"] = suite
    config["condition_id"] = spec["condition_id"]
    return config


def plan(args: argparse.Namespace) -> tuple[Path, dict]:
    base = read_json(args.base_config)
    specs = _condition_specs(base, args.suite)
    suite_root = args.output.resolve()
    manifest_path = suite_root / "suite_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (manifest.get("suite") != args.suite
                or manifest.get("data_root") != str(args.data_root.resolve())
                or manifest.get("base_config") != str(args.base_config.resolve())
                or manifest.get("fixture_only") != bool(args.fixture)
                or [c.get("condition_id") for c in manifest.get("conditions", [])]
                != [s["condition_id"] for s in specs]):
            raise ValueError(f"Existing suite manifest conflicts with current arguments: {manifest_path}")
        for condition in manifest["conditions"]:
            config_path = Path(condition["config"])
            if read_json(config_path) != _build_config(base, condition, args.suite):
                raise ValueError(f"Existing suite config differs from current base: {config_path}")
        print(f"SUITE_PLAN_REUSED {manifest_path}", flush=True)
        print(f"SUITE_EXPECTED_RUNS {manifest['expected_runs']}", flush=True)
        return manifest_path, manifest
    config_root = suite_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    conditions = []
    for spec in specs:
        config = _build_config(base, spec, args.suite)
        config_path = config_root / f"{spec['condition_id']}.json"
        write_json(config_path, config)
        conditions.append({
            **spec,
            "config": str(config_path),
            "output": str(suite_root / "outputs" / args.suite / spec["condition_id"]),
            "expected_runs": len(config["datasets"]) * len(config["seeds"]) * len(spec["models"]),
        })
    manifest = {
        "suite": args.suite,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_config": str(args.base_config.resolve()),
        "data_root": str(args.data_root.resolve()),
        "conditions": conditions,
        "expected_runs": sum(c["expected_runs"] for c in conditions),
        "fixture_only": bool(args.fixture),
    }
    write_json(manifest_path, manifest)
    print(f"SUITE_PLAN {manifest_path}", flush=True)
    print(f"SUITE_EXPECTED_RUNS {manifest['expected_runs']}", flush=True)
    return manifest_path, manifest


def _run_python(args: argparse.Namespace, action: str, condition: dict, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, "-u", str(ROOT / "run.py"), action,
           "--config", condition["config"], "--data-root", str(args.data_root.resolve()),
           "--output", condition["output"], "--stage", "robustness", "--device", args.device]
    if args.fixture:
        cmd.append("--fixture")
    if extra:
        cmd.extend(extra)
    print("SUITE_COMMAND", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def generate(args: argparse.Namespace, manifest: dict) -> None:
    for condition in manifest["conditions"]:
        config = read_json(condition["config"])
        mask_manifest = []
        for seed in config["seeds"]:
            for dataset in config["datasets"]:
                mask_manifest.append(generate_one(args.data_root.resolve(), dataset, int(seed), config))
        path = Path(condition["config"]).with_name(Path(condition["config"]).stem + "_mask_manifest.json")
        write_json(path, {"masks": mask_manifest})
        print(f"MASK_CONDITION_GENERATED {condition['condition_id']} masks={len(mask_manifest)}", flush=True)


def run(args: argparse.Namespace, manifest: dict) -> None:
    for index, condition in enumerate(manifest["conditions"], 1):
        print(f"SUITE_CONDITION [{index}/{len(manifest['conditions'])}] {condition['condition_id']}", flush=True)
        _run_python(args, "preflight", condition)
        _run_python(args, "run", condition)
        _run_python(args, "audit", condition)
    print("SUITE_COMPLETE", args.suite, flush=True)


def status(args: argparse.Namespace, manifest: dict) -> None:
    for condition in manifest["conditions"]:
        _run_python(args, "status", condition)


def audit(args: argparse.Namespace, manifest: dict) -> None:
    failures = []
    for condition in manifest["conditions"]:
        try:
            _run_python(args, "audit", condition)
        except subprocess.CalledProcessError:
            failures.append(condition["condition_id"])
    if failures:
        raise SystemExit("SUITE_AUDIT_INCOMPLETE " + ",".join(failures))
    print("SUITE_AUDIT_PASS", args.suite, flush=True)


def bundle(args: argparse.Namespace, manifest: dict) -> None:
    audit(args, manifest)
    archive = args.output.resolve().parent / (
        f"{args.output.resolve().name}_{args.suite}_results_{time.strftime('%Y%m%d_%H%M%S')}.tar.gz"
    )
    with tarfile.open(archive, "w:gz") as tar:
        root = args.output.resolve()
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix in {".pt", ".tmp", ".lock"}:
                continue
            tar.add(path, arcname=str(path.relative_to(root)))
    print(f"RESULT_BUNDLE {archive}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "generate", "run", "status", "audit", "bundle"])
    parser.add_argument("--suite", choices=["sensitivity", "patterns"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs/robustness.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/r1_robustness_suite")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    if args.fixture and "SMOKE" not in str(args.output):
        raise ValueError("Fixture output must contain SMOKE")
    _, manifest = plan(args)
    if args.action == "plan":
        return
    if args.action == "generate":
        generate(args, manifest)
    elif args.action == "run":
        run(args, manifest)
    elif args.action == "status":
        status(args, manifest)
    elif args.action == "audit":
        audit(args, manifest)
    else:
        bundle(args, manifest)


if __name__ == "__main__":
    main()
