"""Standard-library CLI; no Torch import or GPU is required for evidence checks."""
from __future__ import annotations
import argparse
import json
import sys
from importlib.resources import files
from pathlib import Path
from .evidence import EvidenceError, audit_archive, file_sha, repair_archive, safe_path, write_new_json


def registry():
    return json.loads(files("dasdg_revision_tools").joinpath("engine_registry.json").read_text(encoding="utf-8"))


def check_engine(name, root):
    spec = registry()["engines"][name]
    expected = spec["source_files_sha256"]
    checked = 0
    for relative, digest in expected.items():
        safe_path(relative)
        path = root / relative
        if path.is_symlink() or not path.is_file() or file_sha(path) != digest:
            raise EvidenceError(f"Frozen engine source differs or is missing: {relative}")
        checked += 1
    actual = {p.relative_to(root).as_posix() for p in root.glob("*.py")}
    actual |= {p.relative_to(root).as_posix() for p in (root / "vendor").glob("*.py")}
    expected_code = {n for n in expected if n.endswith(".py") and
                     ("/" not in n or n.startswith("vendor/"))}
    if actual != expected_code:
        raise EvidenceError("Extra or missing protocol-hashed root/vendor source files")
    return {"status": "FROZEN_SOURCE_MATCH", "engine": name, "files_checked": checked,
            "training_rerun": False, "formal_training_environment_checked": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description="DASDG revision evidence tools (not the complete training application)")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("audit-dgcrn", "repair-dgcrn"):
        command = sub.add_parser(action)
        command.add_argument("archive", type=Path)
        command.add_argument("--report", type=Path)
        if action == "repair-dgcrn":
            command.add_argument("--output", type=Path, required=True)
    sub.add_parser("registry", help="Print actual tested dataset/model scopes and remaining integration limits")
    command = sub.add_parser("check-input", help="Validate complete.npz and an optional mask without training")
    command.add_argument("dataset", choices=("PEMS03", "PEMS04", "PEMS08"))
    command.add_argument("complete", type=Path)
    command.add_argument("--fixture", action="store_true")
    command.add_argument("--mask", type=Path)
    command = sub.add_parser("check-engine")
    command.add_argument("name", choices=("persistent", "dgcrn-md"))
    command.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    try:
        if getattr(args, "report", None) and args.report.exists():
            raise FileExistsError("Refusing to overwrite an existing report")
        if args.action == "registry":
            result = registry()
        elif args.action == "check-input":
            try:
                from .contract import ContractError, validate_complete, validate_mask
            except ModuleNotFoundError as exc:
                if exc.name == "numpy":
                    raise RuntimeError("check-input requires optional NumPy; install numpy for NPZ contract checks") from exc
                raise
            result = validate_complete(args.complete, args.dataset, args.fixture)
            if args.mask:
                result["mask"] = validate_mask(args.mask, result["split"], result["shape"][1])
        elif args.action == "check-engine":
            result = check_engine(args.name, args.root.resolve())
        elif args.action == "audit-dgcrn":
            result = audit_archive(args.archive)
        else:
            result = repair_archive(args.archive, args.output)
        if getattr(args, "report", None):
            write_new_json(args.report, result)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0 if result.get("publication_metadata_status") != "NEEDS_REPAIR" else 2
    except (EvidenceError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(f"EVIDENCE_CHECK_FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
