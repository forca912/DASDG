# DASDG v1.0.0 release-candidate audit

## Implemented corrections

- The README now identifies DASDG `graph_no_reliability_channel` as the locked
  main method and Graph WaveNet/GRU as external baselines.
- The baseline orchestrator resolves training and mask-generation scripts from
  the released source directory instead of the caller's working directory.
- Both orchestrators accept an explicit data root.
- The preflight resolves released entry points correctly and can save JSON.
- The package includes the six-run DASDG reference summaries as well as the
  existing twelve-run baseline evidence.
- A single release-validation command now performs preflight, unit testing,
  18 learning runs, result auditing, checksumming, and result archiving.
- A progress command reports processes, GPU use, collected rows, completion
  markers, and recent logs.

## Offline validation

The audit script was exercised against the already completed, unmodified
release-reproduction result packages.

- Main summary rows: 6
- Baseline summary rows: 12
- Main horizon rows: 30
- Baseline horizon rows: 60
- Duplicate or missing condition keys: 0
- Non-finite metrics: 0
- Maximum comparison difference against the same frozen records: 0
- Offline audit verdict: PASS

## Remaining publication gates

The following items are intentionally unresolved and must not be presented as
complete:

1. clean-server execution of `scripts/run_release_validation.sh` with PyTorch
   2.3 and the PEMS data available;
2. author selection of an open-source license;
3. public repository URL and frozen `v1.0.0` release tag;
4. archive DOI and final `CITATION.cff` metadata.
