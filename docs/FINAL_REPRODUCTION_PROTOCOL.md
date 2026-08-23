# DASDG v1.0.0 final reproduction protocol

## Objective

Verify that a clean installation can execute the released DASDG main method and
the two external baselines under the manuscript's locked release condition,
produce complete machine-readable outputs, and recover the frozen reference
metrics within an explicit stochastic tolerance.

## Fixed design

| Item | Value |
|---|---|
| Datasets | PEMS04, PEMS08 |
| Missing mechanism | Persistent rate-controlled sensor missingness |
| Global missing rate | 0.10 |
| Affected-node fraction | 0.40 |
| Block length | 12 time steps |
| Seeds | 20260808, 20260809, 20260810 |
| Main method | `graph_no_reliability_channel` |
| External baselines | `graph_wavenet_adaptive`, `gru_mask_duration` |
| Main training budget | 150 epochs, patience 25 |
| Baseline training budget | 100 epochs, patience 15 |
| Evaluation | Independent frozen test split |

The design contains six main-method runs and twelve baseline runs. Each run must
report horizons 3, 6, 9, 12, and all, yielding 30 main-method and 60 baseline
horizon rows.

## Controlled variables

All methods use the same processed data, time split, frozen persistent masks,
seeds, history length, forecast length, evaluation metrics, and independent-test
policy. Model selection uses validation data only.

## Execution stages

1. Capture Python, package, operating-system, PyTorch, CUDA, and GPU metadata.
2. Run the fail-closed preflight over both datasets and all 108 frozen masks.
3. Run dependency-light source and missing-state unit tests.
4. Execute the six locked DASDG conditions.
5. Execute the twelve external-baseline conditions.
6. Audit condition coverage, horizon coverage, completion markers, finite
   metrics, parameter counts, and deviations from the frozen reference rows.
7. Generate SHA-256 checksums for the validation record.

## Acceptance criteria

The release validation passes only if every check below passes:

- preflight reports 108 checked masks and zero missing masks;
- unit tests return exit code zero;
- the main and baseline completion markers report 6 and 12 learning rows;
- each expected condition key occurs exactly once;
- horizon coverage is exactly `{3, 6, 9, 12, all}` for every condition;
- no metric is NaN or infinite;
- parameter counts equal the frozen reference values; and
- MAE, RMSE, and WAPE for each condition differ from their reference value by no
  more than 5% relative error.

Inference latency is recorded but is not a cross-machine pass/fail criterion,
because GPU and driver differences materially affect timing.

## Failure handling

Do not edit failed result files. Preserve the output directory and log, diagnose
the failure, and start a new timestamped validation directory. A failed or
partial run is not manuscript evidence.

## Expected duration

The frozen baseline reproduction previously required 6.068 hours and the six
main-method runs approximately 1.15 hours on the checked RTX 4090 environment.
Allow 8-10 hours including preflight, testing, setup, auditing, and normal system
variation.
