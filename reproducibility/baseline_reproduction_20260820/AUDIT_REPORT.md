# Baseline reproduction audit

## Verdict

**PASS for the released external-baseline execution path.** This record does not
constitute a reproduction of the DASDG main model.

## Coverage

- Datasets: PEMS04 and PEMS08
- Models: Graph WaveNet adaptive and GRU-mask-duration
- Global missing rate: 0.10
- Persistent block length: 12
- Seeds: 20260808, 20260809, and 20260810
- Completed learning runs: 12
- Horizon metric rows: 60
- Total recorded wall time: 21,844.9 s (6.068 h)

## Completeness checks

- The master log ends with `V16B_PRIMARY_RUN_COMPLETE learning_rows=12`.
- All 12 expected condition keys occur exactly once in `external_summary.csv`.
- Every condition has horizons 3, 6, 9, 12, and all.
- Every run directory contains `history.csv`, `validation_metrics.csv`,
  `test_metrics.csv`, `config.json`, `efficiency.json`, `adjacency.npz`, and
  `best.pt`.
- Every configuration records `independent_test=true`.
- No incomplete run directory was found.

## Historical-result comparison

The 12 reproduced rows were joined to the frozen full v16B result table by dataset,
model, seed, global missing rate, and block length.

- Matched rows: 12/12
- Maximum absolute MAE difference: 0
- Maximum absolute RMSE difference: 0
- Maximum absolute WAPE difference: 0
- Adjacency SHA-256 matches: 12/12

The selected baseline runs are therefore exactly reproduced in the checked
environment.

## Descriptive results

| Dataset | Model | Runs | MAE mean | MAE SD |
|---|---|---:|---:|---:|
| PEMS04 | Graph WaveNet adaptive | 3 | 21.226889 | 0.232999 |
| PEMS08 | Graph WaveNet adaptive | 3 | 17.696309 | 0.102385 |
| PEMS04 | GRU-mask-duration | 3 | 25.922928 | 0.130541 |
| PEMS08 | GRU-mask-duration | 3 | 21.135379 | 0.113774 |

## Boundary of this audit

This experiment verifies the software environment, data/mask path, logging,
checkpointing, independent-test evaluation, result aggregation, and deterministic
re-execution of two external baselines. The paper's main DASDG implementation is
stored in `train_fdr_stgnn.py`, with the locked manuscript variant named
`graph_no_reliability_channel`. A separate release-level DASDG reproduction is
required before claiming that the full software release has been reproduced.
