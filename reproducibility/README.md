# Reproducibility record

Store the following files for every release verification:

- `environment.txt`: Python, PyTorch, CUDA, GPU, operating system, and package versions;
- `preflight.json`: dataset shapes, split boundaries, mask counts, and device status;
- `run.log`: complete stdout/stderr from the orchestrator;
- `external_summary.csv`: one row per dataset/model/seed condition;
- `external_horizon_metrics.csv`: horizon-level MAE, RMSE, and WAPE;
- `integrity_audit.json`: final completeness and provenance check;
- SHA-256 manifests for scripts, configurations, and result files.

Do not replace a failed or incomplete record with a hand-edited summary. Preserve
the original log and create a new release directory for a rerun.

The frozen evidence bundled with this release candidate is split into two
records:

- `release_reproduction_20260820`: six locked DASDG main-method runs;
- `baseline_reproduction_20260820`: twelve Graph WaveNet and GRU baseline runs.

These records are reference evidence for the automatic audit. They do not replace
the clean-environment validation required before publication.
