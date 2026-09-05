# DelayCAST v2 — Delay-epoch Causal Attention with Spectro-Temporal neuron selection

*Methodology, scientific claim, evaluation protocol and figure guide for the ALM / striatum delay → lick study.*

Code: `src/delaycast/` · Config: `configs/delaycast.yaml` · CLI: `python -m delaycast <command>` · Claims: [`CLAIMS.md`](CLAIMS.md)

---

## 1. Problem statement

Head-fixed mice perform an **auditory delayed-response task** (`Data` and `Data2` are both `audio delay`, protocol 1
in the behavioural logs). Every kept trial has

| epoch | duration (audited logs) | role in this study |
|---|---|---|
| presample | 0.5 – 0.9 s | – |
| sample | 0.65 s | optional extra context (`data.context.include_sample`) |
| **delay** | **1.20 s** | **model input = "past context"** (120 bins of 10 ms, ending exactly at the go cue) |
| **go / response** | 1.5 s after the go cue; first lick ≈ 0.2 s after go | **forecast target** (30 bins of 50 ms) **+ behavioural label** |

Spiking is recorded simultaneously in four populations, **left ALM, right ALM, left striatum, right striatum**
(≈ 400–700 units each in `Data`, i.e. ≈ 2000 units per trial). Trials fall into three behavioural classes:
**Ignore** (no lick), **Left**, **Right**. Ignore trials are rare (≈ 4 % of kept trials).

The question is not "can choice be decoded from the delay?" (it can) but **which neurons and which part of the delay-epoch
past context carry the information needed to (a) forecast the same populations' response-epoch activity and (b) predict
the upcoming action** — and whether that answer is the same across recordings.

## 2. Data handling (both datasets, one loader)

* **Dataset A** (`Data/Session*/Rasters/{Ignore,Left,Right}/trial_N.npz`): label = folder; every epoch time comes from
  the NPZ (`delay_start_times`, `go_start_times`, lick times). Videos are indexed but not used by the model.
* **Dataset B** (`Data2/sub-*/sub-*_ses-*/NPZ/{Ignore,Left,Right}/trialN.npz`): label = folder (the **observed** lick
  side, or Ignore), joined on `trial` with the per-session `behavioral_master_log_audited.csv` (fallback
  `combined_audited_master_log.csv` via `session_dir`). Rows are dropped if `excluded`, early lick, photostim, auto/free
  water, or outcome not in `data.qc.csv_keep_outcomes` (default {hit, miss, ignore}: error trials are kept because
  the class is the action, not the instruction; `[hit, ignore]` gives an instruction-only analysis). The Data2 NPZs
  carry lick arrays that are **empty on almost every lick trial** — so a lick record is taken from the NPZ only
  when it actually contains licks, and otherwise from the log row (`left_lick_times` / `right_lick_times`, string
  lists; the record the audit used to define the class) — and they list **only the units that fired in the
  trial**, so unit counts differ from trial to trial. Both NPZ
  schemas are read (combined `brain_region`/`spike_times`, or pre-split `left_ALM_spikes`, …; object arrays, NaN-padded
  matrices and single-unit arrays are all accepted).
* **Unit identity**: within a session, rows are aligned by `unit_ids` — the cache builder first collects the union of
  IDs over all trials (order of first appearance), then places every trial's units by ID; a unit absent from a trial's
  NPZ is a row of zeros (it was silent). IDs only need to be unique within a region (probe-local cluster numbers that
  repeat across hemispheres are fine). Without usable IDs (pre-split schema, an ID repeated inside one region, or
  `unit_ids` and `brain_region` of different length) identity is positional and a trial whose unit count differs from
  the first kept trial's is dropped; when that happens in more than 20 % of a session's trials the session is
  **excluded** (`unit_identity_unavailable`) because its export lost unit identity — the `Data2` trial files are of
  this kind (units that fired in the trial only, no IDs); with `data.recover_identity` (default) their identity is
  recovered by alignment (next bullet) and the exclusion applies only when recovery is switched off — the NWB
  re-export (`scripts/export_nwb_trials.py`) stays the exact alternative; `cache` prints the reason and the fix. The cache JSON records the alignment mode and how many units each
  trial contributed versus the union.
* **Identity recovery** (`data.recover_identity`): an export without IDs that lists every session's units in one
  fixed order minus the silent ones (the `Data2` files: order consistency 0.998 across trials on the twin
  recordings) has its identity recovered by sequence alignment before anything else. Per region, the master slot
  list starts as the trial with the most rows; every trial is aligned to it by a monotone dynamic programme (a
  profile alignment: rows assigned to strictly increasing slots, a skipped slot costs the log-probability that the
  unit is silent, a row that fits no slot is inserted). The match score is the row's *fingerprint* against the
  slot's profile: the trial log rate (`rate`), the PSTH shape — log rate in six task windows (pre-sample, sample,
  early / late delay, early / late response) relative to the trial rate (`windows`) — and two ISI statistics,
  median-over-mean ISI and log ISI CV (`isi`; `data.identity_features`), as a Gaussian with per-slot means and
  sds (shrunk toward the typical slot sd of the feature) and one pooled correlation matrix shared by all slots —
  the features are not independent (compositional shape, the two ISI measures, a trial-wide gain), and with an
  independent-feature score the correct rows had a far heavier tail than chi-square and were rejected; a row is
  inserted when its best fit is beyond the 1 − `identity_p_insert` chi-square quantile of the features it has, a
  missing statistic contributing its expectation. Slot profiles are re-estimated after every pass, slots are
  added where at least `identity_support_frac` of the trials insert a consistent row at the same place (never from
  the first pass, whose master is one trial), slots with fewer rows than that are pruned, and adjacent slots that
  are never co-assigned in a trial and share a profile are merged (a split unit; the profile test is looser when
  the two are populated enough that independent units would have co-occurred). The recovery never uses the trial
  label. The recovered slots are the unit IDs of the session. On the twin recordings the recovery is scored
  against the true IDs (`row_accuracy`, by rate tercile; `frac_units_one_slot`), `cache` prints that table and
  `python -m delaycast identity` re-scores the fingerprint settings side by side — the sparsest units are the
  hardest to align and the least likely to pass the selection floor. Simulation at the real scale (400 units,
  300 trials, log-normal rates centred at 0.6 Hz with sd 0.9, trial-wide log-normal gain sd 0.7, task-modulated
  and selective PSTHs, bursty / regular / Poisson trains, 5 % of units silent per trial): rate alone 0.88 row
  accuracy, the full fingerprint 0.94 (0.92 / 0.94 / 0.96 by rate tercile), 95–98 % of units in one slot.
* **Population representation** (`data.representation: population`, `data.population_groups` = 8): the alternative
  that needs no identity. Per trial and region, the units with at least one spike in the context or target window
  are ranked by their delay-epoch count (ties broken by the full count vector, so the ranking is a function of the
  multiset of rows and not of their order), split into 8 equal rate-quantile groups, and each group's summed counts
  are one channel of the context and of the target raster, most active group first. Silent units add nothing to any
  sum and are left out of the grouping, so the channels of the `Data` export (every unit listed) and of the `Data2`
  export (active units only) of the same trial are identical — the three duplicate recordings are a direct check,
  which `cache` runs in this mode (fraction of matched trials with identical channels, `duplicate_channel_agreement.csv`).
  `data.population_groups` sets the number of channels; finer groups pool fewer units, so less of the single-unit
  selectivity cancels within a channel and the top-ranked channels approach unit identity (8 groups cost 0.75 vs
  0.96 Left/Right accuracy on the `Data` sessions against their unit-level runs; 32 groups gave 0.74, no better
  than 8: the pooling, not the channel count, is the ceiling). `cache` also identifies, on the twin recordings,
  every `Data2` row by its spike train and tests whether the rows keep the `Data` order and whether they keep one
  fixed order across trials (`order_consistency`) — if so, unit identity is recoverable for the `Data2`-only
  sessions by sequence alignment; on the real data the rows are the same active units but not in `Data` order.
  The channels are not neurons: no selection is run, every arm takes all channels (the `rate` / `random` arms and
  the `linonly` / `noskip` ablations are skipped as they would train on identical inputs) and the report marks P1a,
  P1b, P2, P4 and P6 as not applicable, while P0, P3, P5a/b, P7 and P8 are tested on the full corpus (`Data` +
  `Data2`, 11 sessions). Cache key suffix `_pop8`; the cache JSON keeps, per trial, how many units were pooled.
* **Spike time base**: the epoch scalars are absolute session seconds in both datasets, but the `Data2` trial files
  store their spikes on another base (binned against absolute windows they were silent: 0 units pooled, 0.0 channel
  agreement with the `Data` twins). The loader decides the reference per trial from where the spikes lie relative to
  the trial bounds — absolute, relative to `start_time`, or milliseconds — rescales, and records it
  (`spike_time_reference` in the QC log and the `spike_ref` column of the cache table); an unresolvable base drops the
  trial with the numbers in the reason, and a session with no spike in the delay window of any kept trial is excluded
  (`no_spikes_in_window`).
* **NPZ-level QC** (both datasets): licks before the go cue → drop; folder label contradicted by the lick record
  (NPZ arrays, else log row) or licks on both sides → drop; a trial with **no lick record anywhere** keeps its folder
  label (it cannot be verified, `lick_source = none` in the metadata); delay length deviating from 1.2 s by more than
  `data.qc.max_delay_dev_ms` → drop. `cache/<key>/qc_log.csv` lists every trial with its reason, lick source and log
  outcome; `cache` prints the discovered / kept / drop-reason table and warns about any session that lost most of its
  trials.
* **Session minimums**: a session with fewer than `data.min_trials_per_session` (30) trials after QC, or fewer than
  `data.min_trials_per_lick_class` (5) Left or Right trials, is excluded from every command with a warning — such a
  session is a loading problem, not a small recording — and training refuses a degenerate corpus (< 20 training
  trials, one class, or no validation trials) instead of producing a NaN validation curve.
* **Duplicate recordings**: `Data/Session2-4` and three `Data2` sessions are the same recordings extracted twice
  (identical trial counts, class counts and absolute delay-onset timestamps). `cache` detects such pairs by their
  epoch-timestamp fingerprint and every later command uses only one copy (`data.duplicate_keep`, default the `Data`
  copy: complete unit table with IDs, NPZ lick times, identical class labels; the table lists the unit counts of both
  exports), so no trial can sit in the training set of one copy and the test set of the other, and cross-dataset
  transfer is a real transfer.
* Spikes are binned once into **uint8** count tensors (context 10 ms, right-aligned at the go cue; target 50 ms,
  left-aligned at go) and cached per session; ≈ 100 MB per real session in RAM.

Class imbalance is handled by capped class-weighted cross-entropy, stratified splits, and **balanced accuracy /
macro-F1 / Left-vs-Right balanced accuracy** with a within-session label-permutation chance level.

## 3. Stage 1 — model-free neuron selection with stability (per session)

Statistics are computed on the **fit trials only** — the training + validation split of a run (never the test trials;
`selection.fit_on_train_only`) — and vectorised over all units (`features/stats.py`; identical to scipy).

| tag | criterion | statistic | why it matters |
|---|---|---|---|
| floor | activity floor | mean delay rate ≥ 1 Hz and spikes on ≥ 20 % of trials | silent / unstable units cannot carry single-trial information |
| **S** | choice selectivity | Mann-Whitney U, delay spike count **Left vs Right**; effect size AUROC<sub>LR</sub> | the unit's delay rate differs by upcoming lick direction |
| **C** | delay → response coupling | rank correlation between the unit's late-delay (last 400 ms) rate and its own response-epoch rate, **within class and after removing slow drift** (running-mean detrending of the ranks over trial order); p calibrated on circular-shift permutations of the trial order | the unit's own past predicts its own future beyond choice and drift — what the forecaster must exploit |
| **W** | spectro-temporal selectivity | Mann-Whitney U Left vs Right of complex-Morlet CWT band power (slow / theta / beta) **after regressing out spike count**; Bonferroni over bands | rhythmic / transient structure that mean rate cannot see, and that is not a rate test in disguise |
| **R** | ramping | Wilcoxon signed-rank across trials of (late − early delay rate) **within each class** (Bonferroni over classes); slope in Hz/s | trial-level, choice-specific preparatory build-up (the ALM ramp) |
| T | temporal locus *(descriptive)* | AUROC<sub>LR</sub> in sliding 200 ms windows (50 ms step); cluster-mass permutation test → **information onset**, peak window, late-window AUROC, sustained-to-go | which part of the past context the unit's information lives in |
| I | no-lick selectivity *(descriptive)* | Mann-Whitney U Ignore vs lick trials, only with ≥ 8 Ignore trials | Ignore is too rare to enter the eligibility rule |

All p-values are Benjamini-Hochberg corrected across the floor-passing units of the session (q < 0.05). A unit is
**eligible** if it passes the floor and satisfies ≥ 2 of {S, C, W, R} **including at least one of {S, W}**; T and I
never count. The direction requirement (`selection.require_label_criterion`) was added after the first real-data run:
C and R are label-free (a unit that ramps and couples on every trial passes both under any label permutation), so
without it the label-permuted stability null reached 0.88 — the "criteria" set could be assembled from units that carry
no direction information, which is what a rate-matched control set is. Held-out (label-free) selection cannot apply it.

**Stability selection** (Meinshausen & Bühlmann 2010): the criteria and the top-K ranking are recomputed on 50
stratified half-subsamples (without replacement) of the fit trials; a unit's *stability* is its selection frequency.
The final selection takes the eligible units with stability ≥ 0.6, ranked by (stability, score), top-K = 32 per
region; regions with fewer stable units keep K<sub>eff</sub> < K (zero-padded). The expected number of false
selections is bounded by E[V] ≤ K² / ((2·0.6 − 1)·n<sub>eligible</sub>) (written to `selection_funnel.csv`; informative
only when K<sub>eff</sub> ≪ n<sub>eligible</sub>, which real sessions violate), and `delaycast select` also reports what
label-permuted data produce: the median stability of the top-K units and the number of units the full rule would select
— the empirical false-selection estimate (on the four `Data` sessions: null stability 0.14, 0 units). Pairwise φ coefficients between
S/C/W/R flags say how independent the evidence is.

Every unit receives a `reasons` sentence (fit-trial counts per class, class-conditional rates, direction and effect
size, onset, coupling, wavelet band, ramp slope, stability with its denominator, and — for unselected units — why not)
and a fixed-field `reason_short` used in the figure.

Held-out sessions (cross-session / cross-dataset evaluation) are selected **label-free** (floor + C + a net ramp test,
never reading a label) **on their adapt trials only** (`selection.holdout_mode`); the test trials of a held-out
session are never touched by selection, adapter fitting or any statistic, so transfer claims cannot leak.

## 4. Stage 2 — DelayCAST-Net v2

```
selected neurons (K × T) ─► NeuronGate (session) ─► normalised read-in (session) ─┐
                    │                                                              ├─► dilated causal TCN (RF 125 bins)
                    └─► gated population rate ─► learned causal filterbank (band power) ┘   per-time-step channel norm
                                                                                      ─► N causal Transformer blocks
                                                                                         (+ fixed time-to-go encoding)
                                                                                      ─► attention pooling (which past bins?)
      × 4 regions (shared weights)  ─► cross-region attention (which region?) ─► fused vector
                                                        ├─► 3-class head (Ignore / Left / Right)
                                                        └─► forecast decoder ─► session read-out ─► response-epoch rates
                                                                                                    of the K neurons (Poisson)
```

Design choices that make the method testable:

1. **Strictly causal within the delay.** Dilated convolutions use left padding only, normalisation is per time step
   (a `GroupNorm` over time would let bin *t* see the future), attention is causally masked, and the spectral branch
   is a *causal* filterbank (left padding only) applied inside the model. The representation at bin *t* depends on
   bins ≤ *t* only (`tests/test_causality.py`), which is what makes the context sweep and the occlusion maps
   interpretable.
2. **Spectro-temporal branch on the gated population.** A causal filterbank (300 ms windows) of the *gated*
   population rate gives band-power channels that are appended to the read-in; it is recomputed under any mask or
   occlusion and cannot bypass the neuron gates. The bank is **learned** (`spectral_branch: learned`): its quadrature
   kernels are initialised at the fixed Gabor pairs of the slow / theta / beta bands (a causal STFT) and are free
   parameters afterwards, so the branch can move to whatever spectro-temporal feature of the population rate predicts
   the action — the fixed bank (`bands`) added nothing over its control on the four `Data` sessions (P7: −0.013). A
   matched **population-mean control** (`spectral_branch: popmean`) has the same window and gating but no spectral
   information; `none` removes the branch. If the learned bank fails P7 as well, the conclusion is that the spectral
   content of the population rate carries no information about the upcoming lick beyond the units' rates.
3. **Identifiable neuron gates.** Read-in columns are L2-normalised, there is no free scale, and sparsity is enforced
   with the scale-invariant Hoyer penalty — so gate *ranks* are a model-based importance that can be compared with the
   model-free criteria. (The persistence path is an explicit ungated self-term.)
4. **Fixed time-to-go encoding** (sinusoidal, not learned) so attention is not shaped by a learned position prior and
   the model transfers across context lengths.
5. **Interventions are in-distribution.** Training uses prefix-context, random-window and region dropout, so the
   test-time context sweep, window occlusion and region drop are patterns the network has seen.
6. **Wide-and-deep classifier.** The class logits are the sum of two paths: the *deep* path (TCN → Transformer →
   attention pooling → cross-region attention → linear head) and a *wide* path — a session-specific linear read-out
   of each selected unit's mean √count over the visible context, over its last 200 ms (the two features of the
   mean-rate linear baseline) and over `model.skip_windows` (4) equal windows of the context — a time-resolved
   linear decoder in which *when* in the delay a unit carries the side is a learned weight — standardised on the
   training trials, multiplied by the same neuron gates, and with any window that holds no visible bin zeroed after
   standardisation (context sweep, occlusion). The network therefore contains the tuned time-resolved linear
   decoder on the same units as a special case and the deep path only has to add what a linear read-out of
   windowed rates cannot express; `logreg_selected_units_windows` is the matching external baseline of P1b. The
   ablations `linonly` (wide path alone) and `noskip` (deep path alone) are trained in the same pipeline and
   reported next to P1b. Added after the first real-data run, in which the deep path alone was 0.8 points below
   logistic regression on the same units. The wide path is **warm-started** (`model.skip_init: logreg`) at the
   C-tuned, class-balanced multinomial logistic regression fitted by scikit-learn on the standardised features of
   the training (+ adaptation) trials — never the validation trials, which early stopping needs untouched, and never
   the test trials; a class with fewer than two training trials is left out of the fit and starts at its log-prior;
   the weights are divided by the gates' initial values so that the logits are identical — and the deep head starts
   at zero: at epoch 0 the network *is* the tuned linear decoder, and early stopping on the validation cross-entropy
   can only keep what the deep path adds. Trained jointly from zero on the 11-session population corpus the
   read-out had stopped 1.9 points short of the same decoder fitted directly.
7. **Standardised read-in.** The backbone input of every unit is the z-score of its √count with the mean and SD of
   the training trials (`model.standardize_input`), so units — or population channels with tens of spikes per bin —
   enter on the same scale; the gates and the spectral population trace act on the raw gated √counts.
8. **Multi-task objective** `CE(class) + λ · PoissonNLL(response counts) + μ · Hoyer(gates)` with a persistence path
   (each neuron's late-delay log-rate seeds its own forecast) so the decoder learns deviations from persistence.
   Each unit's Poisson term is divided by its training mean count (`train.forecast_norm: mean_count`, floor 0.1), so
   the gradient of the forecast term is O(1) per unit whatever the count scale — without it the population channels
   (≈ 30 spikes per bin) made the forecast term 10× the classification term and the loss unstable. Each unit's
   forecast base (`log_base`, a T<sub>tgt</sub> × K template) starts at its training PSTH — exactly the null the
   forecast is scored against — so the decoder learns deviations (from a zero start, channels with 30 counts per
   bin never reached their scale within early stopping, and a constant base still sat below the PSTH's time
   course), and the forecast log-rate is capped at log 255 (no cached bin holds more; one runaway channel otherwise
   dominates a region's deviance). The checkpoint is the epoch with the lowest class-weighted validation
   cross-entropy (`train.select_by: val_ce`); the multi-task loss is dominated by the Poisson term late in training
   and can select a checkpoint that trades accuracy for forecast likelihood. μ = 0.1 (0.5 pushed the gates of the
   already sparse selected set to ≈ 0.68).
9. **Session adapters + shared backbone**: read-in, gates and read-out are session-specific; TCN, Transformer,
   attention and heads are shared across all sessions of both datasets — joint training on `Data` and `Data2`, and
   adapter-only transfer to held-out sessions / datasets.

## 5. Stage 3 — which context, which neurons: evaluation

Everything is inference-only on the test trials; all quantities are stored per session (the unit of replication).

* **Context sweep** (only the last τ ms visible) → **context sufficiency index** τ<sub>95</sub>: the shortest context
  keeping ≥ 95 % of full-context accuracy, with a bootstrap + isotonic-regression CI (balanced-accuracy and log-loss
  versions), and the same statistic for a tuned linear decoder on *all* units (`tau95_linear_ms`, model-free).
* **Temporal occlusion map**: each 200 ms window is masked (zero occlusion, `evaluate.occlusion: zero`) — the same
  intervention as the training-time window dropout, so the occluded input is in-distribution; the permutation variant
  (window replaced by the same window of another test trial) is available as `permute`. Δ balanced accuracy, Δ log-loss
  and Δ forecast deviance (backbone-only variant with the persistence input held fixed).
* **Region ablation**: in-distribution region drop (primary for P5a: the network was trained with region dropout, so
  it measures whether a region is *needed*) and permutation of the whole region (reported: how much the trained
  model *relies* on it; larger, because a permuted region is out of distribution). Both are always computed.
* **Neuron importance**: permutation occlusion of every selected neuron → Δ log-loss, Δ balanced accuracy, Δ forecast
  deviance of the *other* neurons; joined with gates and criteria (`neuron_importance.csv`); agreement summarised
  within session × region (Spearman ρ, sign test across cells).
* **Forecast**: Poisson deviance explained vs the training-PSTH null, for the model, for persistence alone (late-delay
  rate carried forward; on real data far below the null, so it is reported, not used as the comparator) and for the
  class-conditional oracle (mean response PSTH of the true class; a model above it forecasts trial-specific structure
  beyond class identity).
* **Control arms of the same size**: `rate` and `random` take per session × region as many units as the criteria
  selection produced in that split (K<sub>eff</sub>), never K; a session with an empty criteria set is excluded from
  every criteria-arm comparison in the report and listed in its header.
* **Per-session replication** (supplementary): for arm comparisons with per-trial predictions the report counts the
  sessions whose own trial-bootstrap CI (matched test trials, averaged over seeds) excludes 0 in the predicted
  direction; it never changes a verdict but is the evidence a four-recording corpus can offer.
* **Baselines**: tuned logistic regression (inner CV over C) on all units (delay mean + late-delay mean per unit),
  PCA-50, L1 (its sparse set's Jaccard overlap with the criteria set), the train-selected K units, ALM-only and
  STR-only selected units, and a trial-index drift control.
* **Chance**: 1000 within-session label permutations of the fixed predictions (p95 / p99); trial-bootstrap CIs
  stratified by session × class.
* **Arms**: neuron sets `criteria | rate | random`; model variants `popmean | nospec`; evaluation modes
  `within_session | cross_session (LOSO) | cross_dataset`; the **negative control** (labels permuted within session
  before selection, training and adaptation) must be at chance; 3 seeds by default, splits shared across arms.
* **`delaycast report`** turns all of this into verdicts: every comparison is a per-session paired difference
  (Wilcoxon signed-rank across sessions + session-bootstrap CI); *supported* only when p < 0.05 and the CI excludes 0.

## 6. Scientific claim

> During the 1.2 s delay of the auditory delayed-response task, **(i)** a criterion-selected subset of at most 32 units
> per region — chosen on training trials only by model-free single-unit statistics that survive stability selection —
> supports decoding of the upcoming lick direction with balanced accuracy not lower than a tuned linear decoder on all
> recorded units (P1a, P1b) and above rate-matched and random subsets of the same size (P2); **(ii)** the last 500 ms
> before the go cue retain ≥ 95 % of full-delay accuracy for both the selected-unit model and the all-unit linear
> decoder, and removing the last 400 ms costs more than removing any earlier 400 ms (P3); **(iii)** the selected
> units' late-delay activity forecasts their own response-epoch activity beyond the units' mean response (P4);
> **(iv)** removing ALM input degrades Left/Right decoding more than removing striatal input (P5a), whereas striatal
> involvement in no-lick trials is exploratory because Ignore trials are rare and confounded with engagement (P5b);
> **(v)** model-based neuron importance agrees with the model-free criteria (P6); **(vi)** the causal
> spectro-temporal population branch adds accuracy beyond a matched population-rate control (P7); and **(vii)** a
> backbone trained on one dataset decodes the other above the random-K and label-permuted nulls after fitting only
> session adapters, with held-out units chosen label-free (P8).

Each clause has a comparator and a failure condition (table in [`CLAIMS.md`](CLAIMS.md)); `REPORT.md` prints the
verdicts with the numbers.

## 7. Figures

* **Figure 1 `fig1_raster_selection_<session>.png/.pdf`** (the requested figure): **A** every recorded unit of the four
  regions on one real trial, rows sorted by selection status with a status strip (selected / eligible / floor /
  below floor) and the funnel counts; **B** the K selected units of that trial rank-ordered next to an *evidence
  strip* (AUROC L/R, AUROC Ignore, −log10 q for S/C/W/R, information onset, stability, learned gate and permutation
  importance when a run exists); **C** the rank-1 unit of every region: class-conditional PSTH delay → response with
  SEM, the coupling window and onset marker, and the per-trial late-delay vs response scatter; **D** criteria legend,
  a fixed-field reason table for the top units, and the per-region selection funnel. Sub-title states the trial set
  the statistics used. `python -m delaycast figure1 --npz <trial.npz>` draws it for any NPZ (QC-dropped trials are
  annotated).
* **Figure 2**: Morlet scalograms, STFT band-power time courses and wavelet band power by class.
* **Figure 3**: temporal attention per region and class, temporal occlusion map on the same axis, gate / importance /
  criteria agreement, criteria satisfied by selected units, onset histogram.
* **Figure 4**: confusion matrix, context sweep with τ<sub>95</sub> and the linear sweep, region ablation, forecast
  deviance explained, all arms and baselines with CIs, per-session dot plot (within / cross-session / cross-dataset /
  negative control), per-class ablation deltas.

## 8. Caveats

* Bundled example figures come from the synthetic generator; all numbers in a manuscript must come from the real
  `Data` / `Data2` runs.
* Ignore trials are rare: P5b is exploratory by design, sessions with < 3 Ignore test trials contribute Left/Right
  metrics only, and `logreg_trial_index` flags engagement confounds.
* `Data` has no CSV log; its QC relies on the lick times stored in each NPZ. `Data2` has no lick arrays in the NPZ;
  its QC relies on the audited log. A Data2 trial without a log row is kept with an unverified label.
* The population representation trades the neuron-level claim for coverage: with it the corpus is 11 sessions but
  the verdicts on P1, P2, P4 and P6 are not applicable; only the NWB re-export gives unit-level results on `Data2`.
* Small sessions (e.g. 153 trials with 109 lick trials) can select **no unit**: the criteria are met by some units but
  none is re-selected in ≥ 60 % of half-subsamples. `select` prints how far the eligible units are from the threshold
  (`max_stability_eligible`); such a session stays in the corpus with an empty criteria set and is reported as such.
  `--set selection.min_stability=0.5` relaxes the threshold for every session (the false-selection bound still holds).
* Unit identities are session-local; nothing is ever pooled across sessions by unit index.
