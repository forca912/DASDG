#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/release_validation_v1_0}"

echo "=== active processes ==="
pgrep -af 'run_release_validation|run_dasdg_reproduction|run_v16b_primary|train_fdr_stgnn|train_v16b_external' || true

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true

echo
echo "=== completion markers ==="
grep -H 'DASDG_REPRODUCTION_COMPLETE\|V16B_PRIMARY_RUN_COMPLETE\|DASDG_RELEASE_AUDIT_COMPLETE\|DASDG_RELEASE_VALIDATION_COMPLETE' \
  "$OUTPUT_ROOT"/*.log "$OUTPUT_ROOT"/*/run.log 2>/dev/null || true

echo
echo "=== collected summary rows ==="
for file in "$OUTPUT_ROOT/dasdg/dasdg_summary.csv" "$OUTPUT_ROOT/baselines/external_summary.csv"; do
  if [[ -f "$file" ]]; then
    rows=$(( $(wc -l < "$file") - 1 ))
    echo "$file: $rows rows"
  else
    echo "$file: not created"
  fi
done

echo
echo "=== latest log lines ==="
for file in "$OUTPUT_ROOT/dasdg/run.log" "$OUTPUT_ROOT/baselines/run.log" "$OUTPUT_ROOT/integrity_audit.log"; do
  if [[ -f "$file" ]]; then
    echo "--- $file"
    tail -n 8 "$file"
  fi
done
