# Rodent delay → lick: two pipelines in one repository

This repository contains two independent implementations of the same study (predict response-epoch
activity and the Ignore / Left / Right action from delay-epoch activity of left/right ALM and
left/right striatum, using `Data` and `Data2`):

| pipeline | code | config | docs | entry point |
|---|---|---|---|---|
| **DelayCAST** (criterion-based neuron selection → dilated causal TCN + neuron / temporal / cross-region attention + STFT branch, joint forecasting + classification) | `src/delaycast/` | `configs/delaycast.yaml` | [`paper/DELAYCAST_METHODOLOGY.md`](paper/DELAYCAST_METHODOLOGY.md) | `python -m delaycast …` |
| **SPEC-TCNN** | `src/rodent_tcnn/` | `configs/default.yaml` | [`paper/METHODOLOGY.md`](paper/METHODOLOGY.md) | `python scripts/run_pipeline.py` |

---

## DelayCAST — quick start

```bash
pip install -r requirements.txt          # CPU torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .                         # makes `python -m delaycast` / `delaycast` available

# 1. point the config at your disks (defaults are C:/PythonProject/Rodent/Data and .../Data2)
python -m delaycast inspect --npz-detail --set data.data_a_root=C:/PythonProject/Rodent/Data --set data.data_b_root=C:/PythonProject/Rodent/Data2

# 2. bin all trials (QC + per-session cache), run the neuron-selection criteria (tables with reasons)
python -m delaycast cache
python -m delaycast select

# 3. train + evaluate; `--modes` gives the neuron-set ablation (criteria-selected vs top-K by rate vs random K)
python -m delaycast train --modes criteria,rate,random

# 4. figures (Fig1 raster+selection, Fig2 time-frequency, Fig3 attention, Fig4 results)
python -m delaycast figures --all-sessions

# or everything at once
python -m delaycast all

# leave-one-session-out with adapter-only fitting on 20 % of the held-out session
python -m delaycast train --set train.eval_mode=cross_session

# no real data on this machine? build a layout-identical synthetic tree and run on it
python -m delaycast synth --out synthetic_data
python -m delaycast all --set data.data_a_root=synthetic_data/Data --set data.data_b_root=synthetic_data/Data2 \
       --set data.cache_dir=cache_synth --set output_dir=outputs_synth --set selection.top_k_per_region=16 --set model.d_model=32
```

Any config value can be overridden with `--set key.path=value`. Outputs go to `outputs/delaycast/`:
`selection/*.csv` (per-unit criteria + reasons), `run_<mode>/results.json`, `history.csv`,
`test_predictions.csv`, `attention.npz`, `model.pt`, and `figures/*.png`. Example figures produced
from the synthetic data are in `figures/delaycast_synthetic_example/`. GPU is used automatically when
available (`train.device: auto`).

---

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
