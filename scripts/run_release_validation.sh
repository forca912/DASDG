#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/release_validation_v1_0}"
MAIN_ROOT="$OUTPUT_ROOT/dasdg"
BASELINE_ROOT="$OUTPUT_ROOT/baselines"

mkdir -p "$OUTPUT_ROOT" "$MAIN_ROOT" "$BASELINE_ROOT"
echo "DASDG release validation root: $OUTPUT_ROOT"

python scripts/capture_environment.py > "$OUTPUT_ROOT/environment.json"

python src/dasdg/preflight_v16b.py \
  --processed-dir "$DATA_ROOT/processed" \
  --mask-dir "$DATA_ROOT/masks_v2" \
  --output-json "$OUTPUT_ROOT/preflight.json" \
  2>&1 | tee "$OUTPUT_ROOT/preflight.log"

PYTHONPATH=src python -m unittest discover -s tests -v \
  2>&1 | tee "$OUTPUT_ROOT/unit_tests.log"

python src/dasdg/run_dasdg_reproduction.py \
  --data-root "$DATA_ROOT" \
  --datasets PEMS04 PEMS08 \
  --global-missing-rate 0.1 \
  --block-length 12 \
  --seeds 20260808 20260809 20260810 \
  --output-root "$MAIN_ROOT" \
  2>&1 | tee "$MAIN_ROOT/run.log"

python src/dasdg/run_v16b_primary.py \
  --data-root "$DATA_ROOT" \
  --models graph_wavenet_adaptive gru_mask_duration \
  --datasets PEMS04 PEMS08 \
  --global-missing-rates 0.1 \
  --blocks 12 \
  --seeds 20260808 20260809 20260810 \
  --skip-naive \
  --output-root "$BASELINE_ROOT" \
  2>&1 | tee "$BASELINE_ROOT/run.log"

python scripts/audit_release.py \
  --main-root "$MAIN_ROOT" \
  --baseline-root "$BASELINE_ROOT" \
  --reference-main reproducibility/release_reproduction_20260820/dasdg_main_summary.csv \
  --reference-baseline reproducibility/baseline_reproduction_20260820/external_summary.csv \
  --tolerance 0.05 \
  --output "$OUTPUT_ROOT/integrity_audit.json" \
  2>&1 | tee "$OUTPUT_ROOT/integrity_audit.log"

python scripts/make_sha256_manifest.py "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/sha256_manifest.csv"

tar -czf "${OUTPUT_ROOT}.tar.gz" -C "$(dirname "$OUTPUT_ROOT")" "$(basename "$OUTPUT_ROOT")"
echo "DASDG_RELEASE_VALIDATION_COMPLETE output=$OUTPUT_ROOT archive=${OUTPUT_ROOT}.tar.gz"
