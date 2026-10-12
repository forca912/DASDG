"""Fail-closed audit for the locked DASDG v1.0.0 validation experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DATASETS = ("PEMS04", "PEMS08")
SEEDS = (20260808, 20260809, 20260810)
BASELINES = ("graph_wavenet_adaptive", "gru_mask_duration")
HORIZONS = ("3", "6", "9", "12", "all")
METRICS = ("mae", "rmse", "wape")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def key_main(row: dict[str, str]) -> tuple[str, int]:
    return row["dataset"], int(row["seed"])


def key_baseline(row: dict[str, str]) -> tuple[str, str, int]:
    return row["dataset"], row["model"], int(row["seed"])


def indexed(rows: list[dict[str, str]], key_fn, name: str):
    result = {}
    for row in rows:
        key = key_fn(row)
        if key in result:
            raise RuntimeError(f"Duplicate {name} key: {key}")
        result[key] = row
    return result


def relative_difference(actual: float, reference: float) -> float:
    return abs(actual - reference) / max(abs(reference), 1e-12)


def check_metrics(
    candidate: dict,
    reference: dict,
    prefix: str,
    tolerance: float,
    failures: list[str],
    comparisons: list[dict[str, object]],
) -> None:
    for name in ("test_mae", "test_rmse", "test_wape"):
        actual = float(candidate[name])
        expected = float(reference[name])
        delta = relative_difference(actual, expected)
        comparisons.append({
            "condition": prefix,
            "metric": name,
            "actual": actual,
            "reference": expected,
            "relative_difference": delta,
        })
        if not math.isfinite(actual):
            failures.append(f"{prefix}: non-finite {name}")
        elif delta > tolerance:
            failures.append(
                f"{prefix}: {name} relative difference {delta:.6f} > {tolerance:.6f}"
            )
    if int(candidate["parameters"]) != int(reference["parameters"]):
        failures.append(
            f"{prefix}: parameter count {candidate['parameters']} != {reference['parameters']}"
        )


def check_horizons(
    rows: list[dict[str, str]],
    key_fields: tuple[str, ...],
    expected_conditions: int,
    name: str,
    failures: list[str],
) -> None:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        grouped.setdefault(key, []).append(row)
        for metric in METRICS:
            if not math.isfinite(float(row[metric])):
                failures.append(f"{name} {key}: non-finite {metric}")
    if len(grouped) != expected_conditions:
        failures.append(f"{name}: conditions={len(grouped)} expected={expected_conditions}")
    for key, values in grouped.items():
        found = [row["horizon"] for row in values]
        if sorted(found) != sorted(HORIZONS):
            failures.append(f"{name} {key}: horizon coverage={found}")


def require_marker(path: Path, marker: str, failures: list[str]) -> None:
    if not path.exists():
        failures.append(f"Missing log: {path}")
        return
    if marker not in path.read_text(encoding="utf-8", errors="replace"):
        failures.append(f"Missing completion marker in {path}: {marker}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--reference-main", type=Path, required=True)
    parser.add_argument("--reference-baseline", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    comparisons: list[dict[str, object]] = []
    main_rows = read_csv(args.main_root / "dasdg_summary.csv")
    baseline_rows = read_csv(args.baseline_root / "external_summary.csv")
    main_horizons = read_csv(args.main_root / "dasdg_horizon_metrics.csv")
    baseline_horizons = read_csv(args.baseline_root / "external_horizon_metrics.csv")
    reference_main = indexed(read_csv(args.reference_main), key_main, "reference main")
    reference_baseline = indexed(
        read_csv(args.reference_baseline), key_baseline, "reference baseline"
    )
    candidate_main = indexed(main_rows, key_main, "candidate main")
    candidate_baseline = indexed(baseline_rows, key_baseline, "candidate baseline")

    expected_main = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    expected_baseline = {
        (dataset, model, seed)
        for dataset in DATASETS for model in BASELINES for seed in SEEDS
    }
    if set(candidate_main) != expected_main:
        failures.append(f"Main keys mismatch: {sorted(set(candidate_main) ^ expected_main)}")
    if set(candidate_baseline) != expected_baseline:
        failures.append(
            f"Baseline keys mismatch: {sorted(set(candidate_baseline) ^ expected_baseline)}"
        )

    for key in sorted(expected_main & set(candidate_main) & set(reference_main)):
        row = candidate_main[key]
        if row.get("variant") != "graph_no_reliability_channel":
            failures.append(f"Main {key}: variant={row.get('variant')}")
        check_metrics(
            row, reference_main[key], f"main:{key}", args.tolerance,
            failures, comparisons,
        )
    for key in sorted(
        expected_baseline & set(candidate_baseline) & set(reference_baseline)
    ):
        check_metrics(
            candidate_baseline[key], reference_baseline[key], f"baseline:{key}",
            args.tolerance, failures, comparisons,
        )

    check_horizons(main_horizons, ("dataset", "seed"), 6, "main horizons", failures)
    check_horizons(
        baseline_horizons, ("dataset", "model", "seed"), 12,
        "baseline horizons", failures,
    )
    require_marker(
        args.main_root / "run.log",
        "DASDG_REPRODUCTION_COMPLETE learning_rows=6",
        failures,
    )
    require_marker(
        args.baseline_root / "run.log",
        "V16B_PRIMARY_RUN_COMPLETE learning_rows=12",
        failures,
    )

    report = {
        "status": "PASS" if not failures else "FAIL",
        "tolerance": args.tolerance,
        "main_rows": len(main_rows),
        "baseline_rows": len(baseline_rows),
        "main_horizon_rows": len(main_horizons),
        "baseline_horizon_rows": len(baseline_horizons),
        "comparisons": comparisons,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)
    print("DASDG_RELEASE_AUDIT_COMPLETE status=PASS")


if __name__ == "__main__":
    main()
