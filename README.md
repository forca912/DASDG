# DASDG 1.1.0

DASDG provides configurable training, causal prediction, epoch-boundary resume,
and auditable experiment workflows for multivariate forecasting under persistent
missing observations. The source distribution also preserves the traffic study
protocols for PEMS03, PEMS04 and PEMS08.

The forecasting design combines forward filling, an observation mask, missing
duration, a shared recurrent encoder, sparse dynamic graphs and gated residual
prediction. These are integrated components; the software does not claim that
elapsed-time representations, recurrent units, Top-k selection or gating were
individually invented here.

## Install and run an example

Use Python 3.10–3.12 with NumPy >=1.26,<3 and PyTorch >=2.3,<3. In a separate
environment with compatible PyTorch already installed:

```bash
python -m pip install --no-deps dist/dasdg-1.1.0-py3-none-any.whl
python examples/minimal_reuse.py --output /absolute/path/new_example --device cpu
dasdg --version
dasdg status
```

Alternatively install the source tree with `python -m pip install .`. Installation
does not itself install a GPU driver. See [README.txt](README.txt) for the full API,
CLI and frozen-workflow instructions, and [中文说明](README_中文.txt) for a Chinese
overview. A wheel contains the APIs; the source archive contains the additional
frozen experiment workflows, configurations and examples.

```python
from dasdg.api import Config, fit, predict

fit("dataset.npz", "new_run", Config(), device="cuda")
forecast = predict("new_run/best.pt", values, observed, device="cuda")
```

## Input and recovery contract

Training NPZ files contain complete finite `flow[time,node]`, an equally shaped
boolean `observed` mask (`True` = observed), and chronological integer split
boundaries `[train_end,val_end,total]`. Missing historical inputs are supported;
missing training target labels are not. Prediction uses only available history
and returns `[horizon,node]` in original units.

Resume restores model, optimizer, scheduler, random state, best weights and
training history at epoch boundaries. Changed data/configuration/source/runtime
and concurrent output use are rejected. Custom-data normalization uses complete
training labels reduced in float64; it is separate from the preserved PEMS
float32 preprocessing protocol.

## Study reproduction and external baselines

`frozen/persistent` retains the common R1 comparison, ablation and missingness
protocols; `frozen/dgcrn_md` retains the formal DGCRN-MD adapter and runner.
`source_provenance.json` records preserved sources. Original data, generated
masks and trained weights are not bundled. Supply the audited processed datasets
and masks described by the configurations and data contracts.

DGCRN is an adapted external baseline, not a native generic-API model. Its pinned
upstream is `tsinghua-fib-lab/Traffic-Benchmark` at commit
`b9f8e40b4df9b58f5ad88432dc070cbbbcdc0228`. Its LICENSE says MIT but its README badge
says CC BY-NC-SA; this inconsistency has not been clarified by the rightsholder.
The MIT core does not include its author model source. Separate acquisition,
hash verification, the two formal compatibility changes and attribution are
documented in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).

## Verification evidence

The 1.1.0rc1 candidate passed 20 CPU core checks and 17 unified/GPU checks on
RTX 4090 D, Python 3.12.3, PyTorch 2.3.0+cu121 and NumPy 1.26.4. Returned continuous
and resumed checkpoints were compared independently, including all saved states.
GPU prediction and CLI prediction matched exactly in that recorded environment.

The 1.1.0 training and prediction sources are byte-identical to that accepted
candidate. Final-version installation checks and the version bridge are recorded
in `evidence/` and `release_provenance.json`. Synthetic checks demonstrate software
behavior, not real nontraffic prediction performance or determinism across all
hardware/software environments. The embedded reuse engine retains its original
module identifier to preserve checkpoint identity; the distribution is 1.1.0.

`archival/v1.0.2` contains previous release materials. Its original baseline
budgets and metric tables are historical, not the R1 fair-input protocol.

## Licence and citation

The DASDG core is MIT-licensed; external sources remain governed by their own
terms. Cite the versioned software using [CITATION.cff](CITATION.cff).

The existing concept DOI is [10.5281/zenodo.22070217](https://doi.org/10.5281/zenodo.22070217).
The previous v1.0.2 version DOI is
[10.5281/zenodo.22074995](https://doi.org/10.5281/zenodo.22074995).
The new version DOI is assigned separately when 1.1.0 is archived. This source
tree does not claim that a new DOI has already been issued.
