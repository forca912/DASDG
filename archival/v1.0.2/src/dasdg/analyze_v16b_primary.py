"""Audit and analyze the frozen v16B independent-test results.

This script never reconstructs missing horizon metrics.  The v15 FDR summary
is used only for its reported all-horizon test MAE/RMSE/WAPE, while v16B
external models are required to provide explicit horizon rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


DATASETS = ("PEMS04", "PEMS08")
RATES = (0.05, 0.10, 0.15)
BLOCKS = (6, 12, 24)
SEEDS = tuple(range(20260808, 20260814))
EXPECTED_KEYS = {
    (dataset, rate, block, seed)
    for dataset in DATASETS
    for rate in RATES
    for block in BLOCKS
    for seed in SEEDS
}
LEARNING_MODELS = ("graph_wavenet_adaptive", "gru_mask_duration")
NAIVE_MODELS = ("persistence", "history_mean")
HORIZONS = ("3", "6", "9", "12", "all")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def key(row: dict[str, str]) -> tuple[str, float, int, int]:
    return (row["dataset"], float(row["global_missing_rate"]), int(row["block_length"]), int(row["seed"]))


def metric(row: dict[str, str], name: str = "mae") -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {name}: {row}")
    return value


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def sample_std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def exact_sign_p(values: list[float]) -> float | None:
    signs = [1 if value > 0 else -1 for value in values if value != 0]
    n = len(signs)
    if n == 0:
        return None
    positives = sum(sign > 0 for sign in signs)
    tail = sum(math.comb(n, i) for i in range(positives + 1)) / (2 ** n)
    upper = sum(math.comb(n, i) for i in range(positives, n + 1)) / (2 ** n)
    return min(1.0, 2.0 * min(tail, upper))


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def exact_wilcoxon_p(values: list[float]) -> float | None:
    nonzero = [value for value in values if value != 0]
    if not nonzero:
        return None
    ranks = average_ranks([abs(value) for value in nonzero])
    observed = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    totals = []
    for signs in range(1 << len(ranks)):
        totals.append(sum(rank for index, rank in enumerate(ranks) if signs & (1 << index)))
    lower = sum(total <= observed for total in totals) / len(totals)
    upper = sum(total >= observed for total in totals) / len(totals)
    return min(1.0, 2.0 * min(lower, upper))


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_ci(values: list[float], seed: int = 20260819, repetitions: int = 20000) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = [mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(repetitions)]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def validate_table(rows: list[dict[str, str]], expected_models: tuple[str, ...], expected_horizons: bool) -> dict[str, object]:
    errors: list[str] = []
    counts = Counter(row.get("model", "") for row in rows)
    for row in rows:
        try:
            item = key(row)
            if item not in EXPECTED_KEYS:
                errors.append(f"unexpected condition {item}")
            for name in ("mae", "rmse", "wape"):
                metric(row, name)
        except (KeyError, ValueError) as exc:
            errors.append(str(exc))
    for model in expected_models:
        subset = [row for row in rows if row.get("model") == model]
        expected = len(EXPECTED_KEYS)
        if expected_horizons:
            expected *= len(HORIZONS)
            seen = Counter((key(row), row.get("horizon")) for row in subset)
            duplicates = [item for item, count in seen.items() if count != 1]
            if duplicates:
                errors.append(f"{model}: duplicate/missing horizon keys {duplicates[:3]}")
            expected_pairs = {(condition, horizon) for condition in EXPECTED_KEYS for horizon in HORIZONS}
            actual_pairs = set(seen)
            if actual_pairs != expected_pairs:
                missing = sorted(expected_pairs - actual_pairs, key=str)
                unexpected = sorted(actual_pairs - expected_pairs, key=str)
                errors.append(f"{model}: horizon-key coverage mismatch missing={missing[:3]} unexpected={unexpected[:3]}")
        elif len(subset) != expected:
            errors.append(f"{model}: expected {expected} rows, got {len(subset)}")
    return {"rows": len(rows), "model_counts": dict(counts), "errors": errors, "ok": not errors}


def condition_means(rows: list[dict[str, str]], model: str) -> dict[tuple[str, float, int, int], float]:
    result: dict[tuple[str, float, int, int], float] = {}
    for row in rows:
        if row.get("model") == model and row.get("horizon", "all") == "all":
            item = key(row)
            if item in result:
                raise ValueError(f"Duplicate all-horizon key for {model}: {item}")
            result[item] = metric(row)
    return result


def compare(reference: dict[tuple[str, float, int, int], float], comparator: dict[tuple[str, float, int, int], float], name: str) -> list[dict[str, object]]:
    common = sorted(set(reference) & set(comparator))
    if set(reference) != set(comparator):
        raise ValueError(f"Condition-key mismatch for {name}: reference={len(reference)}, comparator={len(comparator)}, common={len(common)}")
    rows = []
    for item in common:
        ref, comp = reference[item], comparator[item]
        rows.append({
            "dataset": item[0], "global_missing_rate": item[1], "block_length": item[2], "seed": item[3],
            "reference_mae": ref, "comparator_mae": comp,
            "reference_minus_comparator_mae": ref - comp,
            "relative_change_pct": 100.0 * (ref - comp) / comp,
            "reference_better": ref < comp,
            "comparison": name,
        })
    return rows


def aggregate_comparison(rows: list[dict[str, object]], name: str) -> dict[str, object]:
    diffs = [float(row["reference_minus_comparator_mae"]) for row in rows]
    relative = [float(row["relative_change_pct"]) for row in rows]
    return {
        "comparison": name, "n_conditions": len(rows), "mean_relative_pct": mean(relative),
        "std_relative_pct": sample_std(relative), "mean_mae_difference": mean(diffs),
        "reference_better_count": sum(bool(row["reference_better"]) for row in rows),
        "comparator_better_count": sum(not bool(row["reference_better"]) for row in rows),
        "exact_sign_flip_p": exact_sign_p(diffs),
    }


def stratify(rows: list[dict[str, object]], dimensions: tuple[str, ...]) -> list[dict[str, object]]:
    output = []
    for dimension in dimensions:
        groups: dict[tuple[str, object], list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            groups[(str(row["comparison"]), row[dimension])].append(row)
        for (comparison, level), values in sorted(groups.items(), key=lambda item: str(item[0])):
            diffs = [float(row["reference_minus_comparator_mae"]) for row in values]
            rel = [float(row["relative_change_pct"]) for row in values]
            output.append({
                "dimension": dimension, "level": level, "comparison": comparison,
                "n": len(values), "mean_relative_pct": mean(rel), "std_relative_pct": sample_std(rel),
                "reference_better_count": sum(bool(row["reference_better"]) for row in values),
                "exact_sign_flip_p": exact_sign_p(diffs),
            })
    return output


def seed_clustered_summary(rows: list[dict[str, object]], comparison: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["seed"])].append(row)
    seed_rows = []
    for seed, values in sorted(grouped.items()):
        diffs = [float(row["reference_minus_comparator_mae"]) for row in values]
        relative = [float(row["relative_change_pct"]) for row in values]
        seed_rows.append({
            "comparison": comparison,
            "seed": seed,
            "n_conditions": len(values),
            "mean_mae_difference": mean(diffs),
            "mean_relative_pct": mean(relative),
            "reference_better_count": sum(bool(row["reference_better"]) for row in values),
        })
    differences = [float(row["mean_mae_difference"]) for row in seed_rows]
    relative = [float(row["mean_relative_pct"]) for row in seed_rows]
    diff_ci = bootstrap_mean_ci(differences)
    relative_ci = bootstrap_mean_ci(relative)
    summary = {
        "comparison": comparison,
        "n_seeds": len(seed_rows),
        "conditions_per_seed": seed_rows[0]["n_conditions"] if seed_rows else 0,
        "mean_mae_difference": mean(differences),
        "mae_difference_ci95_low": diff_ci[0],
        "mae_difference_ci95_high": diff_ci[1],
        "mean_relative_pct": mean(relative),
        "relative_pct_ci95_low": relative_ci[0],
        "relative_pct_ci95_high": relative_ci[1],
        "seed_direction_count": sum(value < 0 for value in differences),
        "exact_seed_sign_p": exact_sign_p(differences),
        "exact_seed_wilcoxon_p": exact_wilcoxon_p(differences),
    }
    return seed_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v16b-root", type=Path, required=True)
    parser.add_argument("--v15-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    out = args.output_dir or args.v16b_root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    external = read_csv(args.v16b_root / "external_horizon_metrics.csv")
    naive = read_csv(args.v16b_root / "naive_horizon_metrics.csv")
    fdr = read_csv(args.v15_summary)
    audit = {
        "external": validate_table(external, LEARNING_MODELS, True),
        "naive": validate_table(naive, NAIVE_MODELS, True),
        "v15_fdr": {"rows": len(fdr), "errors": [], "ok": True},
    }
    fdr = [row | {"model": "fdr_v15_graph_no_reliability_channel", "horizon": "all"} for row in fdr if row.get("variant") == "graph_no_reliability_channel"]
    fdr_keys = {key(row) for row in fdr}
    if fdr_keys != EXPECTED_KEYS:
        audit["v15_fdr"] = {"rows": len(fdr), "errors": [f"expected 108 FDR keys, got {len(fdr_keys)}"], "ok": False}
    audit["overall_ok"] = all(item["ok"] for item in audit.values() if isinstance(item, dict) and "ok" in item)
    (out / "integrity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not audit["overall_ok"]:
        raise RuntimeError(f"Integrity audit failed; see {out / 'integrity_audit.json'}")

    fdr_all = condition_means(fdr, "fdr_v15_graph_no_reliability_channel")
    comparisons = []
    external_all = [row for row in external if row.get("horizon") == "all"]
    naive_all = [row for row in naive if row.get("horizon") == "all"]
    for model, rows in (("graph_wavenet_adaptive", external_all), ("gru_mask_duration", external_all), ("persistence", naive_all), ("history_mean", naive_all)):
        comparisons.extend(compare(fdr_all, condition_means(rows, model), f"FDR-v15 vs {model}"))
    write_csv(out / "condition_comparisons.csv", comparisons)
    write_csv(out / "comparison_summary.csv", [aggregate_comparison([row for row in comparisons if row["comparison"] == name], name) for name in sorted({str(row["comparison"]) for row in comparisons})])
    write_csv(out / "stratified_comparisons.csv", stratify(comparisons, ("dataset", "global_missing_rate", "block_length", "seed")))
    seed_rows = []
    seed_summaries = []
    for name in sorted({str(row["comparison"]) for row in comparisons}):
        rows, summary = seed_clustered_summary([row for row in comparisons if row["comparison"] == name], name)
        seed_rows.extend(rows)
        seed_summaries.append(summary)
    write_csv(out / "seed_paired_results.csv", seed_rows)
    write_csv(out / "seed_clustered_comparison_summary.csv", seed_summaries)

    horizon_rows = [row for row in external + naive if row.get("horizon") != "all"]
    horizon_summary = []
    for model in LEARNING_MODELS + NAIVE_MODELS:
        subset = [row for row in horizon_rows if row.get("model") == model]
        for horizon in ("3", "6", "9", "12"):
            values = [metric(row) for row in subset if row.get("horizon") == horizon]
            horizon_summary.append({"model": model, "horizon": horizon, "n": len(values), "mae_mean": mean(values), "mae_std": sample_std(values), "rmse_mean": mean([metric(row, "rmse") for row in subset if row.get("horizon") == horizon]), "wape_mean": mean([metric(row, "wape") for row in subset if row.get("horizon") == horizon])})
    write_csv(out / "external_horizon_summary.csv", horizon_summary)

    efficiency = []
    for row in external_all:
        efficiency.append({k: row.get(k, "") for k in ("dataset", "global_missing_rate", "block_length", "model", "seed", "parameters", "latency_per_sample_ms", "training_seconds")})
    write_csv(out / "external_efficiency.csv", efficiency)
    print(json.dumps({"audit": audit, "output_dir": str(out), "comparisons": len(comparisons)}, indent=2))


if __name__ == "__main__":
    main()
