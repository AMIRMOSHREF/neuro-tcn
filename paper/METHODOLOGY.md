# SPEC-TCNN methodology

**Selective Predictive Epoch Context** — a causal temporal network that reads delay-period spikes in four cortico-striatal populations and (i) reconstructs lick-period rasters in those same populations, (ii) classifies the upcoming action as Ignore, Left, or Right, and (iii) names the neurons whose past context was actually used.

## Why this is not a standard decoder

Most delayed-response papers decode choice from delay firing rates. That asks “is the plan there?” It does not ask **which past context, in which neurons, is required to generate the future population trajectory**. SPEC-TCNN forces that question:

- The encoder is **dilated and causal**. A bin at delay time *t* cannot see *t+1*, and it cannot see the go cue or any lick.
- One head must **predict the lick-period raster**, not only a label. A unit that is class-discriminative but irrelevant to future population dynamics is down-weighted.
- Neuron attention is **sparse**. The model is penalized for spreading weight across the whole probe.
- Wavelet/STFT features test whether **time-frequency structure** in the delay, not only spike counts, marks the same units.

## Datasets

### Data (`C:\PythonProject\Rodent\Data`)

Four sessions. Each session has `Rasters/{Ignore,Left,Right}/trial_*.npz` and matching `Videos/`. Class labels come from the folder name. Epoch times come from the NPZ (`delay_start_times`, `go_start_times`, `left_lick_times`, `right_lick_times`).

### Data2 (`C:\PythonProject\Rodent\Data2`)

Three subjects (`sub-440957`, `sub-440958`, `sub-442571x`) with one or more `*_ses-*_behavior+ecephys+image+ogen` folders. Rasters live in `NPZ/{Ignore,Left,Right}/`. Ground truth is the audited master log:

- keep `early_lick == no early`
- drop photostim (`photostim_onset` not N/A, or `photostim_start_times` set)
- drop auto/free water
- drop bilateral licks
- action class = observed lick side, or Ignore if none

`audit_summary` documents the same exclusion reasons (miss, photostim, early lick, auto/free water, both-side licks).

### NPZ schema (both corpora)

`unit_ids`, `brain_region`, `spike_times` (object array), `trial_start` / `trial_stop`, presample / sample / delay / go start–stop, `left_lick_times` / `right_lick_times`. Regions: left ALM, right ALM, left Striatum, right Striatum.

## Epochs

Typical audio-delay trial from the attached CSVs:

| Epoch | Duration | Role |
|---|---|---|
| Presample | ~0.7–0.9 s | Baseline |
| Sample | 0.65 s | Instruction tone |
| **Delay** | **1.20 s** | **Only input to SPEC-TCNN** |
| Go / lick | ~1.5 s | Target raster (800 ms from first lick) |

Ignore trials have no lick; the lick window is go-aligned so the model must predict the *absence* of a lick-locked burst.

Bins: 10 ms. Delay tensor `X ∈ R^{4 × N × 120}`. Lick tensor `Y ∈ R^{4 × N × 80}`.

## Architecture

```
delay rasters (4 regions)
        │
        ▼
  4 × gated DCC stacks (dilations 1,2,4,8)
        │
        ├──────── wavelet CWT + STFT (4–80 Hz) ──► spectral encoder
        │                                              │
        ▼                                              ▼
  neuron attention (softmax over units)      cross-fuse in d_model
        │
        ▼
  causal temporal attention (no future bins)
        │
        ├────────► Poisson lick-raster head
        └────────► 3-class Ignore / Left / Right head
```

Loss:

`L = λ_pred · PoissonNLL(Ŷ, Y) + λ_cls · CE(class) + λ_sparse · mean(attn) + 0.02 · H(attn)`

## Neuron selection (the figure)

Inside each region, every unit gets a composite z-scored score:

| Criterion | What it measures | Weight |
|---|---|---|
| Attention | Mean SPEC-TCNN neuron gate | 0.28 |
| Prediction gain | Occlusion ΔMSE on lick rasters (when a trained model exists) | 0.22 |
| d′ | Delay-rate separability of Left vs Right (and vs Ignore) | 0.22 |
| Delay→lick *r* | Correlation of the unit’s delay PSTH with its lick PSTH | 0.16 |
| TF selectivity | Class modulation of 12–45 Hz CWT energy | 0.12 |

Keep the top 18% per region. Penalize units with delay rate &lt; 0.4 Hz.

**Selection is not “high firing.”** A loud tonic cell loses to a quieter delay-choice cell that (a) predicts its own lick-period burst, (b) is class-discriminative, and (c) carries β/low-γ structure the attention gate uses.

## Validation required before the claim is a result

These are design rules, not optional extras:

- Outer split holds out **entire sessions** (and animals when Data2 allows). Hyperparameters and the 18% threshold are chosen only on inner training sessions.
- Unit IDs are never pooled across sessions. Row 12 in Session1 is not the same cell as row 12 in `sub-440957`.
- Primary classification: macro-F1 and balanced accuracy with session-clustered bootstrap CIs.
- Primary forecast: held-out Poisson deviance vs a mean-rate baseline.
- Null: permute labels and lick rasters within session, then rerun selection.
- Ablations: rate-only logistic, non-causal TCN, spike-only (no TF), no forecast head, leave-one-region-out.
- Ignore windows are **go-aligned**. Left/Right windows start at first lick because that is the epoch being forecast; the encoder still never sees those bins.

## What to report in the paper

1. Figure 1 — full vs selected rasters + reasons (this repo).
2. Classification accuracy vs chance (33%) and vs a rate-SVM / linear baseline on delay FR.
3. Lick-raster NLL: full population vs selected vs random-matched vs time-shuffle.
4. Laterality index on selected ALM (claim C2).
5. ALM→STR vs STR→ALM transfer (claim C3).
6. Attention mass vs delay time (claim C4).
7. Spike-only vs spike+TF ablation (claim C5).
8. Train-on-Data / test-on-Data2 and reverse (claim C6).

## Demo vs real disks

If the Windows paths are absent, `prefer_demo_if_missing: true` builds and uses `data/demo`, which writes the same folder layout and NPZ keys, with planted delay-choice and ramping cells so the figure and claims are inspectable before you attach the real sessions.
