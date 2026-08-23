# DASDG v1.0.0 release candidate

DASDG is executable research software for traffic-flow forecasting under
persistent sensor missingness. The locked manuscript variant is
`graph_no_reliability_channel`: it combines causal forward filling, an observed
mask, normalized missing duration, a shared GRU encoder, a Top-k sparse dynamic
graph, gated graph fusion, and multi-step residual prediction.

This directory is the v1.0.0 release candidate. It is distributed under the MIT
License. The public repository is https://github.com/forca912/DASDG. The archive
DOI will be added after Zenodo publication.

## Verified study scope

- Datasets: PEMS04 and PEMS08.
- Full frozen matrix: missing rates 5%, 10%, and 15%; block lengths 6, 12, and
  24; seeds 20260808 through 20260813.
- Release validation subset: missing rate 10%, block length 12, and seeds
  20260808, 20260809, and 20260810.
- Main method: DASDG `graph_no_reliability_channel`, 6 learning runs.
- External baselines: Graph WaveNet adaptive and GRU-mask-duration, 12 learning
  runs.
- Metrics: MAE, RMSE, and WAPE at horizons 3, 6, 9, 12, and all.

The original traffic data and generated masks are not included. Supply them
under `data/processed` and `data/masks_v2`, or pass another root with
`--data-root`.

## Environment

The checked environment used Python 3.12.3, PyTorch 2.3.0+cu121, Linux, CUDA,
and an NVIDIA RTX 4090. The package itself depends only on NumPy and PyTorch.

```bash
conda env create -f environment.yml
conda activate dasdg
python -m pip install -e .
```

## Quick structural test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Preflight

The preflight checks both processed datasets, all 108 frozen masks, their
missing rates, the released source entry points, and the visible CUDA device.

```bash
python src/dasdg/preflight_v16b.py \
  --processed-dir data/processed \
  --mask-dir data/masks_v2 \
  --output-json outputs/preflight.json
```

## One-command release validation

Run the complete 18-run validation from the repository root:

```bash
bash scripts/run_release_validation.sh
```

For a server session that must survive SSH or browser disconnection:

```bash
nohup bash scripts/run_release_validation.sh > release_validation_launcher.log 2>&1 &
echo $! > release_validation.pid
```

Check progress at any time:

```bash
bash scripts/check_release_progress.sh
```

Rerunning the validation command uses the same default output directory and
skips already complete run directories. To preserve a failed attempt separately,
rename its output directory or set a new `OUTPUT_ROOT` before restarting.

The command executes the preflight, unit tests, six locked DASDG runs, twelve
external-baseline runs, result auditing, and SHA-256 manifest generation. On
the checked RTX 4090 setup, the learning runs required about 7.2 hours in total;
allow 8-10 hours for a clean validation.

The validation passes only when:

1. the main log ends with `DASDG_REPRODUCTION_COMPLETE learning_rows=6`;
2. the baseline log ends with `V16B_PRIMARY_RUN_COMPLETE learning_rows=12`;
3. all 18 condition keys and all 90 horizon rows are present exactly once;
4. all MAE, RMSE, and WAPE values are finite;
5. parameter counts match the frozen reference records; and
6. per-condition metrics remain within the configured 5% relative tolerance.

See `docs/FINAL_REPRODUCTION_PROTOCOL.md` for the full acceptance protocol.

## Direct commands

Main DASDG method:

```bash
python src/dasdg/run_dasdg_reproduction.py \
  --data-root data \
  --datasets PEMS04 PEMS08 \
  --global-missing-rate 0.1 \
  --block-length 12 \
  --seeds 20260808 20260809 20260810 \
  --output-root outputs/dasdg_repro_v1_0
```

External baselines:

```bash
python src/dasdg/run_v16b_primary.py \
  --data-root data \
  --models graph_wavenet_adaptive gru_mask_duration \
  --datasets PEMS04 PEMS08 \
  --global-missing-rates 0.1 \
  --blocks 12 \
  --seeds 20260808 20260809 20260810 \
  --skip-naive \
  --output-root outputs/repro_v1_0
```

## Integrity rules

1. The test split is never used for model or epoch selection.
2. Existing complete run directories are not overwritten by the orchestrators.
3. Failed or incomplete validation records are preserved and never promoted to
   manuscript evidence.
4. New manuscript numbers must be traceable to saved CSV, JSON, or log files.
5. A model implementation change requires a new software version and new runs.

## License and citation

The source code is released under the MIT License; see `LICENSE`. Citation
metadata are provided in `CITATION.cff`. The repository URL is recorded there;
the archive DOI will be added after publication.
