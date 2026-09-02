# SPEC-TCNN — delay-period neuron selection for lick-time prediction

Use the **delay** epoch in four populations (left ALM, right ALM, left striatum, right striatum) to (1) reconstruct those same signals in the **lick** epoch and (2) classify **Ignore / Left / Right**. The model is a temporal CNN with **dilated causal convolution**, **neuron + temporal attention**, and a **wavelet / STFT** branch.

Both corpora are first-class:

| Corpus | Layout | Labels |
|---|---|---|
| **Data** | `Session*/Rasters/{Ignore,Left,Right}/trial_*.npz` | folder name |
| **Data2** | `sub-*/sub-*_ses-*/NPZ/{Ignore,Left,Right}/*.npz` | audited CSV, with photostim / early-lick / bilateral-lick dropped |

NPZ keys match the attached extraction README (`unit_ids`, `brain_region`, `spike_times`, delay/go timestamps, lick times).

## Scientific claim (one sentence)

A sparse cortico-striatal ensemble already present in the last few hundred milliseconds of delay is sufficient to forecast lick-period population activity and the upcoming action; SPEC-TCNN names those neurons and the past context they use.

Full claims and tests: [`paper/CLAIMS.md`](paper/CLAIMS.md). Methods: [`paper/METHODOLOGY.md`](paper/METHODOLOGY.md).

## Commands

```bash
pip install -r requirements.txt

# Point at your disks
#   configs/default.yaml → data_root, data2_root
# Windows defaults:
#   C:\PythonProject\Rodent\Data
#   C:\PythonProject\Rodent\Data2

# If those folders are missing, build a schema-identical demo
python scripts/prepare_demo.py

# Discover both datasets, train SPEC-TCNN, select neurons, write Figure 1
python scripts/run_pipeline.py --epochs 10

# Train only
python scripts/train.py --epochs 20

# Heavier: wavelet/STFT inside the train loop
python scripts/run_pipeline.py --tf --epochs 8
```

Outputs:

- `figures/fig1_neuron_selection.png` — all units, then selected units, with reasons
- `figures/fig0_spec_tcnn_schematic.png`
- `outputs/trial_catalog.csv`
- `outputs/selection_figure_trial.csv`
- `outputs/laterality.json`
- `outputs/checkpoints/spec_tcnn.pt`

## Paper companion

```bash
cd dashboard
npm install
npm run dev -- --port 43173
```

The dashboard shows Figure 1, per-neuron reasons, the six claims, and the run book.

## Neuron selection (what the gold ticks mean)

Inside each region, every unit is z-scored on five criteria and the top 18% are kept:

1. **SPEC-TCNN neuron attention** — the model actually used this unit as past context
2. **Prediction gain** — occluding it raises lick-raster error
3. **Delay-rate d′** — Left vs Right (and vs Ignore)
4. **Delay→lick coupling** — does this cell’s delay PSTH predict its own lick PSTH?
5. **TF selectivity** — class-modulated 12–45 Hz Morlet / STFT energy

Silent units (&lt; 0.4 Hz in delay) are penalized. A loud tonic cell loses to a quieter delay-choice or ramping cell.

## Real data

Copy the audited logs next to Data2 or into `data/metadata/`:

- `combined_audited_master_log.csv`
- `combined_behavioral_master_log.csv`
- `audit_summary.csv`

The loader already understands those filenames. No code change is required when you swap the demo tree for `C:\PythonProject\Rodent\...`.
