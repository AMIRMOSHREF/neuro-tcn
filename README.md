# NeuroContext

NeuroContext is a complete Python pipeline for four-region spike-population
analysis. It uses delay-period activity from left/right ALM and left/right
striatum to:

1. forecast each recorded neuron's response-period spike counts; and
2. classify the eventual action as Ignore, Left, or Right.

The proposed SCOPE-DCC model combines dilated causal convolutions, per-neuron
temporal attention, a log-STFT branch, sparse neuron gates, and cross-region
attention. See [METHODOLOGY.md](METHODOLOGY.md) for the claim, controls, and
statistical requirements.

## Install

Python 3.10 or newer is required.

```powershell
cd C:\PythonProject\Rodent\NeuroContext
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Linux/macOS activation is `source .venv/bin/activate`.

## Expected data

The loader supports both supplied layouts:

```text
Data/
  Session1/
    Rasters/{Ignore,Left,Right}/trial_*.npz
Data2/
  sub-440957/
    sub-440957_ses-.../
      NPZ/{Ignore,Left,Right}/trial*.npz
      behavioral_master_log_audited.csv
```

The Data2 CSV is authoritative when passed with `--metadata-csv`. Multiple
metadata files can be supplied by repeating that argument. The combined audited
CSV can be used once if it covers all sessions.

## Run on the real datasets

First audit all NPZ keys, region labels, metadata matches, and class coverage:

```powershell
neurocontext audit `
  --data-root "C:\PythonProject\Rodent\Data" `
  --data2-root "C:\PythonProject\Rodent\Data2" `
  --metadata-csv "C:\PythonProject\Rodent\Data2\combined_audited_master_log.csv" `
  --output results\audit.json
```

Train all outer held-out-session folds:

```powershell
neurocontext train `
  --config config\default.yaml `
  --data-root "C:\PythonProject\Rodent\Data" `
  --data2-root "C:\PythonProject\Rodent\Data2" `
  --metadata-csv "C:\PythonProject\Rodent\Data2\combined_audited_master_log.csv" `
  --fold all `
  --output-dir results
```

For one fold (useful for initial checks), replace `--fold all` with `--fold 0`.
CUDA is selected automatically; force CPU with `--device cpu`.

Rank neurons using a trained fold. Use only records belonging to that fold's
training groups for a strict paper analysis; the checkpoint records those group
names:

```powershell
neurocontext select `
  --checkpoint results\fold_0.pt `
  --data-root "C:\PythonProject\Rodent\Data" `
  --data2-root "C:\PythonProject\Rodent\Data2" `
  --metadata-csv "C:\PythonProject\Rodent\Data2\combined_audited_master_log.csv" `
  --output results\fold_0_neuron_selection.csv
```

Create the 300-DPI PNG and vector PDF publication figure:

```powershell
neurocontext figure `
  --config config\default.yaml `
  --npz "C:\PythonProject\Rodent\Data\Session1\Rasters\Left\trial_32.npz" `
  --selection-csv results\fold_0_neuron_selection.csv `
  --output results\figure_selected_ensemble.png
```

The figure shows all neurons in all four regions beside only selected neurons,
with delay and held-out response epochs shaded. Its footer states the selection
criteria and per-region counts. If `--selection-csv` is omitted, the command
enters clearly labelled QA mode and highlights the most active delay units; that
fallback must not be used as a scientific result.

## Reproducible smoke test without private data

```bash
neurocontext make-demo --output-dir demo
neurocontext audit --config demo/demo_config.yaml \
  --data-root demo/Data --data2-root demo/Data2 \
  --metadata-csv demo/audited_metadata.csv
neurocontext train --config demo/demo_config.yaml \
  --data-root demo/Data --data2-root demo/Data2 \
  --metadata-csv demo/audited_metadata.csv --fold 0 --output-dir demo/results
neurocontext select --checkpoint demo/results/fold_0.pt \
  --data-root demo/Data --data2-root demo/Data2 \
  --metadata-csv demo/audited_metadata.csv \
  --output demo/results/neuron_selection.csv
neurocontext figure --config demo/demo_config.yaml \
  --npz demo/Data/Session1/Rasters/Left/trial_3.npz \
  --selection-csv demo/results/neuron_selection.csv \
  --output demo/results/neuron_selection.png
```

## Outputs

- `fold_N.pt`: model weights, exact configuration, train groups, held-out groups
- `fold_N_history.json`: optimization history
- `fold_N_metrics.json`: balanced accuracy, macro F1, confusion matrix, Poisson NLL
- `neuron_selection.csv`: gate stability, class effect size, preferred delay bin,
  and explicit selection reasons
- `neuron_selection.png/.pdf`: publication raster figure

## Important interpretation limits

The uploaded files contain CSV metadata and an NPZ schema, but no actual NPZ
sample, so repository figures generated during development use synthetic data.
Replace them with real data outputs before publication. Predictive attention is
not evidence that a neuron is causally necessary. STFT features summarize
binned spike-rate dynamics and should not be described as LFP oscillations.
