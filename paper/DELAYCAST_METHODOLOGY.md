# DelayCAST — Delay-epoch Causal Attention with Spectro-Temporal neuron selection

*Methodology, scientific claim and figure guide for the ALM / striatum delay → lick study.*

Code: `src/delaycast/` · Config: `configs/delaycast.yaml` · CLI: `python -m delaycast <command>`

---

## 1. Problem statement

Head-fixed mice perform an **auditory delayed-response task** (both `Data` and `Data2` are `audio delay`,
protocol 1 in the behavioral logs). Every trial has

| epoch | duration (audited logs) | role in this study |
|---|---|---|
| presample | 0.5 – 0.9 s | – |
| sample | 0.65 s | optional extra context (`data.context.include_sample`) |
| **delay** | **1.20 s** (all kept trials) | **model input / "past context"** |
| **go / response** | 1.5 s window after the go cue; first lick ≈ 0.21 s (median) after go | **forecast target + behavioural label** |

Simultaneous spiking is recorded in four populations: **left ALM, right ALM, left striatum, right striatum**.
Trials fall into three behavioural classes: **Ignore** (no lick), **Left** (lick left), **Right** (lick right).

Goal: use only delay-epoch activity of the four populations to
(i) **forecast** the response-epoch activity of the same populations and
(ii) **classify** the upcoming action, while identifying **which neurons** and **which part of the past
context** carry that information.

## 2. Data handling (both datasets, one loader)

* **Dataset A** (`Data/Session*/Rasters/{Ignore,Left,Right}/trial_N.npz`): label = folder; all epoch
  times come from the NPZ. Videos are indexed (`Videos/...`) but not used by the model.
* **Dataset B** (`Data2/sub-*/sub-*_ses-*/NPZ/{Ignore,Left,Right}/trialN.npz`): label = folder, joined
  on `trial` with the per-session `behavioral_master_log_audited.csv` (fallback:
  `combined_audited_master_log.csv` via `session_dir`). Rows are dropped if `excluded`, early lick,
  photostim, auto/free water, or outcome not in {hit, ignore}.
* **NPZ-level QC for both datasets**: licks before the go cue → drop; folder label contradicted by the
  lick times (or licks on both sides) → drop; region labels not matching ALM/striatum × left/right →
  flagged (`n_unknown_region`). A `qc_log.csv` lists every excluded trial with its reason.
* Spikes are binned at **10 ms** in the delay (120 bins) and at **50 ms** in the response window
  (30 bins); one compressed cache file per session (`cache/`).

Class imbalance is severe in the audited logs (Ignore ≈ 4.5 % of kept trials). We therefore use
class-weighted cross-entropy, stratified splits and report **balanced accuracy / macro-F1** with a
label-shuffle chance distribution.

## 3. Stage 1 — criterion-based neuron selection (interpretable, per session)

For every unit and every session (`delaycast select`), computed on the delay epoch:

| tag | criterion | statistic | why it matters |
|---|---|---|---|
| floor | activity floor | mean rate ≥ 1 Hz and spikes on ≥ 20 % of trials | removes silent / unstable units that cannot carry single-trial information |
| **S** | choice selectivity | Kruskal-Wallis of delay spike counts across Ignore/Left/Right (+ η², Cohen's d L-R) | the unit's delay rate differs by upcoming action |
| **C** | delay → response coupling | Spearman ρ between the unit's late-delay (last 400 ms) rate and its own response-epoch rate, across trials | the unit's own past predicts its own future — exactly what the forecaster exploits |
| **W** | spectro-temporal selectivity | Kruskal-Wallis across classes of complex-Morlet CWT band power (1-4, 4-12, 12-30 Hz) of the single-trial rate | rhythmic / transient structure that mean rate cannot see |
| **R** | ramping | Spearman ρ of the trial-averaged delay PSTH vs time | the classical signature of preparatory activity in ALM |

All p-values are Benjamini-Hochberg corrected across units (q < 0.05). A unit is **eligible** if it passes
the floor and satisfies ≥ 2 criteria; eligible units are ranked by a weighted sum of −log₁₀ q and the
**top-K per region (K = 32)** are selected. Every unit gets a human-readable `reasons` string
(e.g. *"choice-selective delay rate (q=3e-7, η²=0.76, d(L-R)=+3.2); late-delay rate predicts own response rate (ρ=+0.72)"*).

## 4. Stage 2 — DelayCAST-Net

```
selected neurons (K × T)  ─► NeuronGate (session) ─► normalised read-in (session) ─┐
STFT band power (3 × T)  ───────────────────────────────────────────────────────────┴─► dilated causal TCN
                                                                                          (5 blocks, k=3, RF = 125 bins)
                                                                                     ─► causal temporal self-attention
                                                                                     ─► attention pooling (which past bins?)
      × 4 regions (shared weights)   ─► cross-region attention (which region?) ─► fused vector
                                                        ├─► 3-class head (Ignore / Left / Right)
                                                        └─► forecast decoder ─► session read-out ─► response-epoch
                                                                                                  rates of the K neurons
                                                                                                  (Poisson likelihood)
```

Design choices that make the method novel and testable:

1. **Dilated causal convolutions** guarantee that the representation at bin *t* only uses bins ≤ *t*;
   with the receptive field matched to the delay (125 bins) the network can, but need not, use the
   entire delay.
2. **Causal temporal attention + learned-query pooling** return an explicit distribution over past bins —
   the model's own answer to *"which past context is used?"*. Read-outs are compared with a
   **context-length sweep** at test time (only the last τ ms are visible) and with **context-length
   augmentation** during training so that the sweep is in-distribution.
3. **Learned neuron gates** with an L1 penalty. The read-in weights are L2-normalised per input neuron,
   so the gate is the *only* per-neuron scale — this makes the learned gate an identifiable importance
   score that can be correlated with the statistical criteria of Stage 1 (Figure 3C).
4. **Spectro-temporal branch**: STFT band powers (slow / theta / beta) of the selected population are
   appended as extra input channels; the CWT is used in Stage 1. The two representations therefore
   serve complementary roles (selection vs. modelling).
5. **Multi-task objective** `CE(class) + λ · PoissonNLL(response-epoch counts) + μ · Σ gates`, plus a
   **persistence path**: each neuron's late-delay log-rate is added to its own forecast, so the decoder
   learns *deviations* from persistence. The evaluation reports deviance explained of the full model
   *and* of persistence alone versus the training-set PSTH null.
6. **Session adapters + shared backbone**: unit identities differ across sessions and datasets, so
   read-in, gates and read-out are session-specific while the TCN / attention / heads are shared.
   This is what allows training on `Data` and `Data2` jointly and enables **cross-session
   evaluation**: the backbone is frozen and only the adapter is fitted on a small fraction
   (`train.adapt_frac`) of the held-out session's trials.

## 5. Evaluation protocol

* `within_session` (default): stratified train / val / test per session, all sessions trained jointly.
* `cross_session`: leave-one-session-out with adapter-only fitting on the held-out session.
* Reported: balanced accuracy, macro-F1, confusion matrix, per-session accuracy, label-shuffle chance;
  Poisson deviance explained per region (model vs persistence); context-length sweep; region ablation
  (zero one region); **neuron-set ablation** (`--modes criteria,rate,random`: criterion-selected vs
  top-K most active vs random K); linear baselines (multinomial logistic regression on delay mean
  rates of all units / of the selected units).

## 6. Scientific claim (to be tested on the real recordings)

> **Claim.** *Information about the upcoming action (lick left / lick right / withhold) is carried
> during the delay by a sparse cortico-striatal subpopulation whose late-delay dynamics also forecast
> their own response-epoch activity; a causal model restricted to the last ≈ 300-500 ms of the delay
> from this subpopulation performs as well as one that sees the whole delay from all units, and
> removing ALM degrades decoding more than removing striatum.*

Testable predictions and the analysis that tests each:

| prediction | analysis | supported if |
|---|---|---|
| P1 sparsity: criterion-selected K units ≥ all-units linear decoder | Fig. 4E (`criteria` vs `logreg_all_units`) | balanced accuracy not lower |
| P2 selection is not just "loud" units | `criteria` vs `rate` vs `random` runs | criteria > rate > random |
| P3 late-delay sufficiency | context sweep (Fig. 4B) | curve saturates by 300-500 ms before go |
| P4 delay → response coupling is a defining property | fraction of selected units with **C** (Fig. 3E); forecast > persistence and > PSTH null (Fig. 4D) | deviance explained > 0 and coupled units over-represented |
| P5 ALM dominance for choice, striatal contribution for Ignore | region ablation (Fig. 4C), cross-region attention per class (Fig. 3B) | ALM drop > STR drop; Ignore trials shift attention |
| P6 model-learned importance agrees with statistics | gate vs score (Fig. 3C) | positive Spearman ρ across sessions |
| P7 spectro-temporal information is real | selected units with **W** only; run with `model.use_spectral_branch=false` | accuracy drop when branch removed |

The claim is deliberately phrased so that each part can fail independently.

## 7. Figures

* **Figure 1 `fig1_raster_selection_<session>.png`** (the requested selection figure):
  **A** all recorded units of the four regions for one trial (sample / delay / response shaded, licks
  on top), selected units in colour; **B** the selected units rank-ordered, with badges **S C W R**
  showing which criteria each unit satisfies; **C** combined selection score; **D** class-conditional
  PSTH (delay → response) of the top unit of every region; **E** bullet-point reasons.
* **Figure 2 `fig2_time_frequency_<session>.png`**: Morlet scalograms of the selected population per
  region for the same trial, STFT band-power time courses per class, wavelet band power by class.
* **Figure 3 `fig3_attention.png`**: temporal attention over the delay per region and class, cross-region
  attention, learned gates vs statistical score, gate distributions, criteria satisfied by selected units.
* **Figure 4 `fig4_results.png`**: confusion matrix, context sweep with shuffle band, region ablation,
  forecast deviance explained (model vs persistence), neuron-set ablation + linear baselines,
  per-session accuracy.

## 8. Caveats

* The bundled figures in `figures/delaycast_synthetic_example/` come from the synthetic generator
  (`delaycast synth`) and only demonstrate the pipeline; all numbers in a manuscript must come from the
  real `Data` / `Data2` runs.
* Ignore trials are rare; keep `qc.min_trials_per_class` in mind, use balanced metrics and consider
  merging sessions of the same animal if a session has < 5 Ignore trials.
* `Data` has no CSV log; its QC relies on the lick times stored in each NPZ.
