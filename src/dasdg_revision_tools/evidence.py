"""Verify/repair the fixed six-run DGCRN-MD public archive without extraction.

The repair changes publication metadata only. Original DONE.json records,
metrics, source files and all other payload bytes remain unchanged.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


class EvidenceError(ValueError):
    """The archive does not satisfy its evidence contract."""


EXPECTED = {(d, s) for d in ("PEMS04", "PEMS08")
            for s in (20260808, 20260809, 20260810)}
GLOBAL = "PUBLIC_ARCHIVE_MANIFEST.json"
PROVENANCE = "provenance/original_public_manifests/"
REQUIRED = {"run_config.json", "data_contract.json", "history.csv",
            "progress.json", "test_metrics.csv", "validation_metrics.csv",
            "compat/compatibility_manifest.json", "compat/compatibility.diff",
            "compat/pf_dgcrn/net.py", "compat/pf_dgcrn/layer.py"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def safe_path(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise EvidenceError(f"Unsafe archive path: {name!r}")
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts) or PurePosixPath(name).is_absolute():
        raise EvidenceError(f"Unsafe archive path: {name!r}")
    return name


def read_archive(path):
    """Bounded, duplicate/link-safe in-memory read; never deserialize checkpoints."""
    payload, names, total = {}, set(), 0
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = safe_path(member.name.rstrip("/") if member.isdir() else member.name)
            key = name.casefold()
            if key in names:
                raise EvidenceError(f"Duplicate or case-colliding member: {name}")
            names.add(key)
            if member.isdir():
                continue
            if not member.isfile():
                raise EvidenceError(f"Links/special files are not allowed: {name}")
            total += member.size
            if member.size > 64 * 1024**2 or total > 256 * 1024**2:
                raise EvidenceError("Public evidence archive exceeds the bounded read limit")
            stream = archive.extractfile(member)
            payload[name] = stream.read()
            if len(payload[name]) != member.size:
                raise EvidenceError(f"Truncated member: {name}")
    return payload


def get_json(payload, name):
    try:
        return json.loads(payload[name].decode("utf-8-sig"))
    except (KeyError, ValueError, UnicodeError) as exc:
        raise EvidenceError(f"Missing or invalid JSON: {name}") from exc


def check_hash(payload, name, expected):
    safe_path(name)
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise EvidenceError(f"Invalid SHA256 for {name}")
    if name not in payload or sha(payload[name]) != expected.lower():
        raise EvidenceError(f"Missing file or hash mismatch: {name}")


def omitted_kind(name):
    if name == "best.pt":
        return "checkpoint"
    if "__pycache__" in PurePosixPath(name).parts and name.endswith(".pyc"):
        return "python_bytecode_cache"
    raise EvidenceError(f"Unexpected missing artifact: {name}")


def analyze(payload, corrected=False):
    global_manifest = get_json(payload, GLOBAL)
    manifest_paths = sorted(n for n in payload if re.fullmatch(
        r"runs/PEMS0[48]/seed[0-9]+/DONE_public\.json", n))
    indexed = global_manifest.get("files", [])
    if len(indexed) != len(manifest_paths) or {r["path"] for r in indexed} != set(manifest_paths):
        raise EvidenceError("Global public manifest index is incomplete or duplicated")
    if global_manifest.get("public_manifest_count") != 6:
        raise EvidenceError("Expected six public manifests")
    for row in indexed:
        check_hash(payload, row["path"], row["sha256"])
    # Extra/misnamed source runs must not be silently ignored.
    source_done = {n for n in payload if n.endswith("/DONE.json") and n.startswith("runs/")}
    if source_done != {p.rsplit("/", 1)[0] + "/DONE.json" for p in manifest_paths}:
        raise EvidenceError("Unexpected or missing original completion records")
    runs, conditions, present_count, missing_count = [], set(), 0, 0
    for public_path in manifest_paths:
        root = public_path.rsplit("/", 1)[0]
        public = get_json(payload, public_path)
        done = get_json(payload, root + "/DONE.json")
        config = get_json(payload, root + "/run_config.json")
        key = (config.get("dataset"), config.get("seed"))
        if type(key[1]) is not int or key not in EXPECTED or key in conditions:
            raise EvidenceError(f"Unexpected or duplicate dataset/seed: {key}")
        if root != f"runs/{key[0]}/seed{key[1]}":
            raise EvidenceError(f"Directory/config condition mismatch: {root}")
        conditions.add(key)
        if done.get("status") != "FORMAL_COMPLETE" or public.get("status") != "FORMAL_COMPLETE":
            raise EvidenceError(f"Non-formal completion: {root}")
        identity = config.get("identity")
        if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
            raise EvidenceError(f"Invalid run identity: {root}")
        if identity != done.get("identity") or identity != public.get("run_identity"):
            raise EvidenceError(f"Run identity mismatch: {root}")
        if config.get("formal_result") is not True or config.get("compatibility_variant") != "DGCRN-MD-adapted":
            raise EvidenceError(f"Unexpected DGCRN protocol: {root}")
        original_files = done.get("files", {})
        if not REQUIRED <= set(original_files):
            raise EvidenceError(f"Required source artifacts not indexed: {root}")
        present, omitted = {}, {}
        for relative, expected in original_files.items():
            safe_path(relative)
            name = root + "/" + relative
            if name in payload:
                check_hash(payload, name, expected)
                present[relative] = expected
                present_count += 1
            else:
                kind = omitted_kind(relative)
                if not re.fullmatch(r"[0-9a-f]{64}", expected):
                    raise EvidenceError(f"Invalid omitted artifact hash: {relative}")
                omitted[relative] = {"sha256": expected, "kind": kind,
                                     "availability": "not_locally_verified",
                                     "reason": "omitted_from_source_public_archive"}
                missing_count += 1
        if not REQUIRED <= set(present):
            raise EvidenceError(f"Missing required metrics/configuration/source: {root}")
        old_public = public.get("public_files", {})
        if corrected:
            # Index all run files, including the original DONE record, but never itself.
            expected_public = {n[len(root) + 1:]: sha(b) for n, b in payload.items()
                               if n.startswith(root + "/") and n != public_path}
            if old_public != expected_public:
                raise EvidenceError(f"Corrected public file coverage mismatch: {root}")
            if public.get("omitted_artifacts") != omitted:
                raise EvidenceError(f"Omitted artifact disclosure mismatch: {root}")
            if (public.get("dataset"), public.get("seed")) != key:
                raise EvidenceError(f"Corrected condition mismatch: {root}")
            for rel, digest in old_public.items():
                check_hash(payload, root + "/" + rel, digest)
        elif old_public != original_files:
            raise EvidenceError(f"Source public/source DONE file maps disagree: {root}")
        runs.append({"dataset": key[0], "seed": key[1], "identity": identity,
                     "root": root, "present_source_files": len(present),
                     "omitted_artifacts": omitted,
                     "manifest_condition": {"dataset": public.get("dataset"),
                                            "seed": public.get("seed")}})
    if conditions != EXPECTED:
        raise EvidenceError("Incomplete frozen six-run matrix")
    if corrected:
        coverage = global_manifest.get("archive_files", {})
        expected_coverage = {n: sha(b) for n, b in payload.items() if n != GLOBAL}
        if coverage != expected_coverage:
            raise EvidenceError("Corrected global archive coverage mismatch")
        for name, digest in coverage.items():
            check_hash(payload, name, digest)
        source_hashes = global_manifest.get("unchanged_source_files", {})
        if not source_hashes:
            raise EvidenceError("Missing original payload preservation record")
        for name, digest in source_hashes.items():
            check_hash(payload, name, digest)
        original_manifests = {n[len(PROVENANCE):]: b for n, b in payload.items()
                              if n.startswith(PROVENANCE)}
        original = {n: payload[n] for n in source_hashes}
        original.update(original_manifests)
        # Revalidate saved source manifests and every original required payload.
        analyze(original, corrected=False)
        if set(original_manifests) != {GLOBAL, *manifest_paths}:
            raise EvidenceError("Missing or extra preserved source manifests")
    return {"artifact_integrity_status": "PASS", "verification_status": "ANALYZED",
            "training_rerun": False, "completed_runs": len(runs), "expected_runs": 6,
            "present_source_file_hashes_checked": present_count,
            "omitted_artifact_count": missing_count, "corrected_manifest": corrected,
            "limits": ["Checkpoint availability on the powered-off server is not verified.",
                       "Original data/mask files are not rehashed by this archive check.",
                       "This is artifact validation, not a benchmark reproduction rerun."],
            "runs": runs}


def audit_archive(path):
    payload = read_archive(path)
    corrected = get_json(payload, GLOBAL).get("manifest_version") == 2
    report = analyze(payload, corrected=corrected)
    report.update(archive_sha256=file_sha(path), archive_bytes=Path(path).stat().st_size)
    if not corrected:
        report["publication_metadata_status"] = "NEEDS_REPAIR"
    else:
        report["publication_metadata_status"] = "PASS"
    return report


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(json_bytes(value))


def repair_archive(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.exists():
        raise FileExistsError("Refusing to overwrite the source or an existing output")
    payload = read_archive(source)
    if get_json(payload, GLOBAL).get("manifest_version") == 2:
        raise EvidenceError("Archive already repaired; use audit-dgcrn")
    source_report = analyze(payload)
    preserved = {n: sha(b) for n, b in payload.items()
                 if n != GLOBAL and not n.endswith("/DONE_public.json")}
    original_sha = file_sha(source)
    created = datetime.now(timezone.utc).isoformat()
    original_global = payload[GLOBAL]
    payload[PROVENANCE + GLOBAL] = original_global
    for run in source_report["runs"]:
        root = run["root"]
        path = root + "/DONE_public.json"
        payload[PROVENANCE + path] = payload[path]
        public_files = {n[len(root) + 1:]: sha(b) for n, b in payload.items()
                        if n.startswith(root + "/") and n != path}
        payload[path] = json_bytes({"manifest_type": "public_result_manifest",
            "manifest_version": 2, "created_at_utc": created,
            "run_identity": run["identity"], "dataset": run["dataset"],
            "seed": run["seed"], "status": "FORMAL_COMPLETE",
            "public_files": public_files, "omitted_artifacts": run["omitted_artifacts"],
            "source_done_json": "DONE.json",
            "note": "Original completion record is unchanged. Omitted artifacts have source hashes only; server retention is not verified."})
    manifest_paths = [r["root"] + "/DONE_public.json" for r in source_report["runs"]]
    payload[GLOBAL] = json_bytes({"archive_type": "dgcrn_md_formal_public_results",
        "manifest_version": 2, "created_at_utc": created, "public_manifest_count": 6,
        "source_archive_sha256": original_sha,
        "repair_scope": "public_manifests_only_original_payload_unchanged",
        "files": [{"path": n, "sha256": sha(payload[n])} for n in manifest_paths],
        "unchanged_source_files": preserved,
        "archive_files": {n: sha(b) for n, b in payload.items() if n != GLOBAL},
        "checkpoint_policy": "omitted; source hashes preserved; current server availability not locally verified"})
    analyze(payload, corrected=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents an overwrite race, even between parallel invocations.
    created_output = False
    try:
        with output.open("xb") as stream:
            created_output = True
            with tarfile.open(fileobj=stream, mode="w:gz") as archive:
                for name, data in sorted(payload.items()):
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(data), 0o644, 0
                    archive.addfile(info, io.BytesIO(data))
        reread = read_archive(output)
        if reread != payload:
            raise EvidenceError("Output archive readback changed payload bytes")
        report = audit_archive(output)
        if file_sha(source) != original_sha:
            raise EvidenceError("Source archive changed during repair")
        report.update(source_archive_sha256=original_sha,
                      preserved_original_payload_files=len(preserved),
                      preserved_original_manifests=7)
        return report
    except BaseException:
        if created_output:
            output.unlink(missing_ok=True)
        raise
