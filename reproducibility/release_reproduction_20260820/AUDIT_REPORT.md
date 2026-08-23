# DASDG main-method reproduction record

## Verdict

PASS for the locked DASDG main-method execution path represented by the saved
release-reproduction files.

## Coverage

- Datasets: PEMS04 and PEMS08
- Variant: `graph_no_reliability_channel`
- Global missing rate: 0.10
- Persistent block length: 12
- Seeds: 20260808, 20260809, and 20260810
- Completed learning runs: 6
- Horizon metric rows: 30

## Completeness checks

- The master log ends with `DASDG_REPRODUCTION_COMPLETE learning_rows=6`.
- All six dataset-seed condition keys occur exactly once.
- Every condition contains horizons 3, 6, 9, 12, and all.
- Each collected run was required to contain its checkpoint, history, test
  metrics, configuration, efficiency, diagnostics, and reconstruction files.
- Every collected configuration records the locked release variant.

## Descriptive results

| Dataset | Runs | MAE mean | MAE SD | Parameters | Latency ms/sample |
|---|---:|---:|---:|---:|---:|
| PEMS04 | 3 | 20.696995 | 0.108323 | 49,794 | 1.751358 |
| PEMS08 | 3 | 16.580165 | 0.111920 | 47,465 | 1.717430 |

The values above are computed from `dasdg_main_summary.csv`; they are not
manually substituted estimates.

## Boundary

This record documents the already completed locked main-method run. The final
public release must still pass the clean-environment 18-run validation defined in
`docs/FINAL_REPRODUCTION_PROTOCOL.md`.
