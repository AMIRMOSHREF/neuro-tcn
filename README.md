# DelayCAST v2 — which delay-epoch neurons, and which past context, predict the upcoming lick

Head-fixed mice, auditory delayed-response task, simultaneous spiking in **left / right ALM** and **left / right
striatum** (`Data`: 4 sessions of `Session*/Rasters/{Ignore,Left,Right}/trial_*.npz`; `Data2`: 3 animals × several
`sub-*_ses-*/NPZ/{Ignore,Left,Right}/*.npz` sessions with audited behavioural logs). Using **only the 1.2 s delay**
(the past context, ending at the go cue) of the four populations, DelayCAST

1. **selects** the neurons that carry information about the upcoming action with model-free, stability-checked
   single-unit criteria (choice selectivity, delay→response coupling, rate-normalised wavelet selectivity, ramping;
   information onset as a descriptor) — computed on training trials only, with a written reason for every unit;
2. **forecasts** the response-epoch activity of those neurons and **classifies** Ignore / Left / Right with a strictly
   causal network (dilated causal TCN → causal Transformer blocks → attention pooling → cross-region attention, with a
   causal Gabor filterbank of the gated population rate as the spectro-temporal branch);
3. **measures which past context is used** (context sweep → context-sufficiency index τ<sub>95</sub>, permutation
   occlusion maps over time, regions and neurons) and **tests a falsifiable claim** (`REPORT.md`, per-session paired
   statistics, negative control).

Methods: [`paper/DELAYCAST_METHODOLOGY.md`](paper/DELAYCAST_METHODOLOGY.md) · Claim and predictions:
[`paper/CLAIMS.md`](paper/CLAIMS.md) · NPZ schema: [`paper/NPZ_SCHEMA.md`](paper/NPZ_SCHEMA.md)

The legacy SPEC-TCNN pipeline (`src/rodent_tcnn/`, `scripts/run_pipeline.py`) is kept for reference but is no longer
the recommended entry point.

---

## Install

```bash
pip install -r requirements.txt      # CPU-only torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .                     # provides `python -m delaycast` and the `delaycast` console script
pytest -q tests                      # ~1 min: schema, causality, selection, leakage sentinel on synthetic data
```

### Windows / PowerShell quick start

The commands must be run **inside the repository clone** (not inside the data folder), one per line (PowerShell 5
does not accept `&&`), and the package must be installed once so that `python -m delaycast` exists:

```powershell
cd C:\PythonProject
git clone https://github.com/amirmoshref/neuro-tcn.git          # or: cd neuro-tcn ; git pull
cd C:\PythonProject\neuro-tcn
git checkout claude/neuro-tcn-rodent-signals-l44cae
pip install -r requirements.txt
pip install -e .
python -m delaycast inspect --npz-detail          # defaults already point at C:/PythonProject/Rodent/Data and /Data2
python -m delaycast all --quick                   # sanity run, then: python -m delaycast all
```

`scripts\run_delaycast.ps1` runs the whole protocol step by step (`-Quick` for the sanity run). Outputs go to
`outputs\delaycast\` and the binned cache to `cache\` inside the folder you run from. `-Population` runs the same
protocol on **both datasets** with the identity-free population representation (see below; outputs in
`outputs\delaycast_pop\`, cache in `cache_pop\`).

`cache` prints, per session, how many trials were discovered, kept and dropped (with the reasons), where the lick
record came from (`Data`: NPZ arrays; `Data2`: the audited log — its NPZ lick arrays are empty on lick trials) and how
units were aligned. It also reports **duplicate recordings**: `Data/Session2-4` are the same recordings as three
`Data2` sessions (same trials, same epoch timestamps); only one copy is used afterwards (`data.duplicate_keep`, default
the `Data` copy, which has every unit with its ID). Sessions that lose most of their trials in QC, or end with fewer
than 30 trials / 5 Left / 5 Right, are excluded from every command with a warning. The binned cache is keyed by a
loader version and rebuilds itself after a loader change; delete stale runs before re-running the protocol:

```powershell
if (Test-Path outputs\delaycast\runs) { Remove-Item -Recurse -Force outputs\delaycast\runs }
if (Test-Path cache\selection)       { Remove-Item -Recurse -Force cache\selection }
```

### The `Data2` NPZ export has no unit identity — re-export it from NWB

The `Data2` trial files (`left_ALM_spikes`, `right_ALM_spikes`, … object arrays) contain **only the units that fired
in that trial and no unit IDs**: the unit count changes from trial to trial (e.g. 1371 → 1551 within one session)
and nothing says which row of one trial is which row of the next. No per-unit analysis is possible on such files, so
`cache` **excludes** those sessions (`EXCLUDED at cache time … unit_identity_unavailable`) instead of silently mixing
units; `cache/<key>/excluded_sessions.csv` lists them. The three `Data2` sessions that duplicate `Data/Session2-4`
are covered by their `Data` copies (complete unit table with IDs). For the other seven sessions, re-export the trials
from the NWB files with the `Data` schema (`unit_ids` + `brain_region` + `spike_times` for **all** units, lick
times, epoch scalars) — either with the exporter that produced `Data/Session*`, or with the bundled one:

```powershell
pip install pynwb
python scripts\export_nwb_trials.py --nwb-dir D:\nwb_files --out C:\PythonProject\Rodent\Data2_units --log-dir C:\PythonProject\Rodent\Data2
python -m delaycast inspect --npz-detail --set data.data_b_root=C:/PythonProject/Rodent/Data2_units
python -m delaycast cache --set data.data_b_root=C:/PythonProject/Rodent/Data2_units
```

(`--log-dir` picks up each session's audited log for labels and lick times; set `data.data_b_root` in
`configs/delaycast.yaml` to the new folder afterwards.) Until then the **unit-level** corpus is the four `Data`
sessions, and `cross_dataset` transfer is skipped with a warning.

### Data + Data2 without unit identity: the population representation

What the `Data2` export does preserve is every trial's **population**: the spike counts of the units that fired,
summed in any way that does not need to know which unit is which. `data.representation: population`
(`.\scripts\run_delaycast.ps1 -Population`) uses exactly that: in every trial and region the active units are ranked
by their delay-epoch count, split into `data.population_groups` (8) equal rate-quantile groups, and each group's
summed counts become one channel — most active group first. A unit's rate rank is a stable property, so channel g
approximates the same units from trial to trial, and the channels are identical whether a file lists every recorded
unit (`Data`) or only the ones that fired (`Data2`), in any row order — `cache` verifies this on the three recordings
present in both trees and prints the fraction of trials with identical channels. The whole corpus — 4 `Data` + 7 non-duplicate
`Data2` sessions, 11 sessions of 3 animals — goes through the same pipeline: same TCN + Transformer backbone, same
spectral branch, same forecast head, same cross-dataset transfer, same negative control, same figures and report.

```powershell
git pull
.\scripts\run_delaycast.ps1 -Population            # -> outputs\delaycast_pop\REPORT.md, figures, runs
```

What it can and cannot test: the channels are *not neurons*, so there is nothing to select — every arm uses all
channels, `select` and Figure 1 are skipped (Figure 2 shows the time-frequency content of all channels) and the report
marks the unit-level predictions **P1a, P1b, P2, P4, P6 as not applicable**. The temporal (P3), regional (P5a/P5b), spectral (P7) and transfer (P8) predictions, and the negative
control, are tested on the full 11-session corpus, so they reach the ≥ 5-session rule the four `Data` sessions cannot.
The unit-level predictions on all 11 sessions still need the NWB re-export above.

## Commands (real data)

Point the config at your disks once (defaults are `C:/PythonProject/Rodent/Data` and `.../Data2`), or pass
`--set data.data_a_root=... --set data.data_b_root=...` to every command.

```bash
# 0. what is on disk (both layouts, both NPZ schemas, CSV joins)
python -m delaycast inspect --npz-detail

# 1. bin every trial once (uint8 per-session tensors, QC log with reasons)          ~ 1 min / session
python -m delaycast cache

# 2. descriptive neuron selection on all trials: tables with reasons, funnel,        ~ 2-4 min / session
#    stability null; outputs/delaycast/selection/*.csv
python -m delaycast select

# 3. train + evaluate.  Neuron sets: criteria | rate | random; model controls: popmean | nospec;
#    seeds from configs (0,1,2); every run re-selects neurons on ITS training split.  ~ 30-60 min / run (4-core CPU)
python -m delaycast train --modes criteria,rate,random --variants popmean --seeds 0,1,2

# 4. transfer: train on one dataset, adapt session read-in/read-out on 30 % of the other's trials
#    (held-out units chosen label-free), test on the rest
python -m delaycast train --modes criteria,random --seeds 0 --set train.eval_mode=cross_dataset
#    leave-one-session-out (opt-in, one holdout at a time)
python -m delaycast train --modes criteria --seeds 0 --set train.eval_mode=cross_session --holdout A/Session1,B/sub-440957_ses-20190211T143614

# 5. negative control: labels permuted within session BEFORE selection/training -> must be at chance
python -m delaycast train --modes criteria --seeds 0 --negative-control

# 6. figures (Fig 1 raster + selection for every session, Fig 2 time-frequency, Fig 3 attention/occlusion, Fig 4 results)
python -m delaycast figures --all-sessions
#    Figure 1 for one specific trial file (works for QC-dropped trials too)
python -m delaycast figure1 --npz C:/PythonProject/Rodent/Data/Session1/Rasters/Left/trial_331.npz

# 7. the verdict on every prediction of the claim -> outputs/delaycast/REPORT.md
python -m delaycast report

# everything above in one go (3 seeds, popmean control, cross-dataset, negative control, figures, report)
python -m delaycast all
python -m delaycast all --quick          # one seed, within-session only (≈ 10 min on the synthetic tree, 1-2 h on real data)
```

Any config value can be overridden with `--set key.path=value` (`configs/delaycast.yaml` documents every key).
GPU is used automatically (`train.device: auto`).

## No real data on this machine?

```bash
python -m delaycast synth --out synthetic_data
python -m delaycast all --quick --set data.data_a_root=synthetic_data/Data --set data.data_b_root=synthetic_data/Data2 \
       --set data.cache_dir=cache_synth --set output_dir=outputs_synth --set selection.top_k_per_region=16 --set model.d_model=32
```

The synthetic tree mirrors both layouts and NPZ schemas; example figures made from it are in
`figures/delaycast_synthetic_example/` (they demonstrate the pipeline, not a result).

## Outputs (`outputs/delaycast/`)

| path | content |
|---|---|
| `selection/<session>.csv` | one row per unit: rates per class, AUROC L/R, coupling ρ, wavelet band, ramp slope, onset / peak / late-window AUROC, q-values, criteria flags, stability, rank, `reasons`, `reason_short` |
| `selection/summary.csv`, `selection/funnel.csv` | per-session / per-region funnel (recorded → floor → eligible → stable → selected), φ between criteria, false-selection bound, stability null |
| `runs/within/<mode>[_<variant>]/seed<k>/` | `results.json` (classification + CIs + chance, forecast, context sweep + τ<sub>95</sub>, temporal occlusion, region ablation, baselines, importance agreement), `neuron_importance.csv`, `attention.npz`, `test_predictions.csv`, `selection_<session>.csv` (train-split selection actually used), `model.pt` |
| `runs/cross_dataset/…`, `runs/cross_session/…`, `runs/negative_control/…` | same, plus a pooled `results.json` per seed |
| `figures/fig1_raster_selection_<session>.png/.pdf` | all recorded units of one trial → selected units with evidence strip, exemplars, reasons, funnel |
| `figures/fig2_…`, `fig3_attention.png`, `fig4_results.png` | time–frequency, attention + occlusion + importance agreement, results |
| `REPORT.md`, `report.json` | verdict per prediction with numbers, CIs, n sessions |

## What the model is, how accurate it is, and how to make it more accurate

**Model** (`src/delaycast/models/delaycast_net.py`, ≈ 0.4 M parameters): per session, the K ≤ 32 selected units of
each region enter through a neuron gate and a normalised read-in; a dilated causal TCN (5 blocks, kernel 3, receptive
field 125 bins = the whole 1.2 s delay) and two causal Transformer blocks with a fixed time-to-go encoding produce
per-bin features; attention pooling over time and cross-region attention give one fused token; the class logits are
the sum of the head on that token (deep path) and a session-specific linear read-out of every unit's mean counts
(wide path, so the model contains the linear decoder); a Poisson decoder forecasts each unit's response-epoch counts
from the same token plus a persistence path; a causal Gabor filterbank of the gated population rate is the
spectro-temporal branch.

**Where the spectral and Transformer parts are.** Wavelet analysis (`selection.wavelet`, complex Morlet CWT band power
over the delay, criterion **W**) is part of the neuron selection and of Figure 2; the in-model counterpart is the causal
Gabor filterbank on the gated population rate (`model.spectral_branch: bands`, an STFT with a causal Hann window per
band), whose control is the same filterbank collapsed to the population mean (`popmean`, prediction P7); the two causal
Transformer blocks (`model.n_transformer_layers: 2`, `model.n_heads`) sit after the TCN and are what the attention
centre-of-mass and the temporal occlusion of Figure 3 are computed from (the attention map is the last block's; the
stack has at least one block). All of them are on by default; the `nospec` / `popmean` variants ablate the spectral
branch, and `model.n_transformer_layers` sets the depth of the Transformer stack.

**Accuracy on the four `Data` sessions** (3 seeds, test trials never used for selection or training; chance ≈ 0.62):
Left/Right balanced accuracy 0.968 / 0.956 / 0.878 on Sessions 2–4 with ≈ 90–104 selected units out of ≈ 2,000
(Session 1 selects no unit). Logistic regression on the same units reaches 0.975 / 0.960 / 0.891, on **all** units
0.986 / 0.970 / 0.941. Those all-unit numbers are the information ceiling of the recordings: no architecture can
decode a trial whose delay activity does not carry the upcoming lick side, and part of the residual error is
behavioural (the report splits accuracy by hit / miss where a log exists).

**Levers, in order of effect**: (1) more recordings — every verdict needs ≥ 5 sessions and the shared backbone
improves with trials (`Data2` re-export); (2) more units — `selection.top_k_per_region` (64, 128) with
`selection.fill_unstable: true` trades the sparsity claim for accuracy towards the all-unit ceiling; (3) the wide path
and CE checkpointing (defaults since the first run) close the gap to the linear decoder on the same units; (4) seeds
average out split noise but do not raise a single run. Nothing else in the architecture is expected to move the
number by more than a point on this corpus.

## Protocol revisions after the first real-data run

The first report on the four `Data` sessions changed four rules; every one is documented in
[`paper/CLAIMS.md`](paper/CLAIMS.md) and the methodology, and all of them require re-running `select` and every
`train` arm (`.\scripts\run_delaycast.ps1` does the whole protocol):

* **eligibility needs a direction criterion** (`selection.require_label_criterion`): ≥ 2 of {S, C, W, R} *including* S
  or W. C and R are label-free, so the old rule let the label-permuted stability null reach 0.88;
* **control arms of the same size** (`selection.match_k_to_criteria`): `rate` / `random` take K<sub>eff</sub> units per
  session × region, the number the criteria selection produced, not K;
* **temporal occlusion masks the window** (`evaluate.occlusion: zero`), the same intervention as the training-time
  window dropout; the permutation variant stays available;
* **P4 is tested against the units' mean response** (deviance explained > 0); persistence and the class-conditional
  oracle are reported next to it;
* **wide-and-deep classifier** (`model.linear_skip`): a gated, standardised mean-count read-out is added to the
  backbone logits, checkpoints are chosen by validation cross-entropy (`train.select_by: val_ce`) and the gate penalty
  is 0.1; the ablations `linonly` / `noskip` are part of the protocol.

The report also excludes sessions with an empty criteria set from criteria-arm comparisons (listed in its header,
with K<sub>eff</sub> per session), prints, per comparison, in how many sessions the prediction replicates on the
session's own test trials, tests P2 / P7 on Left/Right balanced accuracy (the claim's metric; with a handful of Ignore
test trials each one moves the 3-class score by several points), shows both region-ablation methods for P5a, and
reports the number of units the selection rule would pick with permuted labels as the empirical false-selection
estimate.

## The figure (what the gold-coloured rows mean)

Figure 1 shows every recorded unit of the four regions on one real trial (rows sorted by selection status with a
status strip), the K selected units of that trial rank-ordered next to their evidence (AUROC Left/Right, −log10 q of
S/C/W/R, information onset, stability, learned gate and permutation importance), the rank-1 unit of every region
(class-conditional PSTH delay → response, coupling window, onset marker, per-trial coupling scatter), the criteria
legend with a fixed-field reason line per unit, and the selection funnel per region. A unit is selected because it
passes the activity floor, satisfies at least two of {S, C, W, R} at BH-FDR q < 0.05 on the training trials, and is
re-selected in ≥ 60 % of 50 stratified half-subsamples of those trials; the `reasons` column of the selection table
spells this out with the numbers for every unit, selected or not.

## Scientific claim (one paragraph)

During the delay, a criterion-selected, stability-checked subset of ≤ 32 units per region supports decoding of the
upcoming lick direction at least as well as a tuned linear decoder on all ≈ 2000 recorded units and better than
rate-matched or random subsets; the last ≈ 500 ms before the go cue are sufficient and the last 400 ms are the most
costly to remove; the selected units' late-delay activity forecasts their own response-epoch activity beyond
persistence; ALM input matters more than striatal input for lick direction (striatal involvement in no-lick trials is
exploratory); model-based neuron importance agrees with the model-free criteria; the causal spectro-temporal branch
adds accuracy over a matched population-rate control; and the backbone transfers across datasets after adapter-only
fitting with label-free unit selection. Each clause is a row in [`paper/CLAIMS.md`](paper/CLAIMS.md) with its
comparator and failure condition; `REPORT.md` prints the verdicts.

## Runtime budget (CPU workstation, ≈ 2000 units × 350 trials × 15 sessions)

Measured on a 4-core CPU: one training step (batch 32, K = 32 × 4 regions, 0.57 M parameters) ≈ 0.45 s, i.e. ≈ 1 min
per epoch over ≈ 4500 training trials; a 60-epoch run is ≈ 30–60 min (early stopping usually halves it), evaluation
≈ 10 min. cache ≈ 15 min once; select ≈ 45 min once (cached per split afterwards). The full `all` protocol (3 seeds ×
4 arms + cross-dataset + negative control ≈ 15 runs) is therefore an overnight CPU job or ≈ 1 h on a GPU
(`train.device: auto`); `all --quick` (one seed, within-session only) is ≈ 1–2 h on CPU. RAM ≈ 1.5 GB for all
session caches (uint8) + model. The CLI uses every core unless `OMP_NUM_THREADS` is set; run one job at a time on a
CPU workstation (two PyTorch processes oversubscribing the cores slow each other down many-fold).
