# Scientific claim and its testable predictions (DelayCAST v2)

`python -m delaycast report` evaluates every row of this table from the saved runs and writes `REPORT.md`.
Every comparison between arms is a **per-session paired difference** (the session is the unit of replication):
Wilcoxon signed-rank across sessions plus a session-level bootstrap CI of the mean difference. A prediction is

* **supported** when the CI excludes the failure threshold *and* p < 0.05,
* **inconclusive** when it does not (never "refuted" from a point comparison),
* **not run / not testable** when the required arms or trial counts are missing.

## The claim

> During the 1.2 s delay of the auditory delayed-response task, (i) a criterion-selected subset of at most 32 units per
> region — chosen on training trials only by model-free single-unit statistics that survive stability selection —
> supports decoding of the upcoming lick direction with balanced accuracy not lower than a tuned linear decoder on all
> recorded units and above rate-matched and random subsets of the same size; (ii) the last 500 ms before the go cue
> retain ≥ 95 % of full-delay accuracy for both the selected-unit model and the all-unit linear decoder, and removing
> the last 400 ms costs more than removing any earlier 400 ms; (iii) the selected units' late-delay activity forecasts
> their own response-epoch activity beyond a persistence baseline; (iv) removing ALM input degrades Left/Right decoding
> more than removing striatal input, whereas striatal involvement in no-lick trials is exploratory; (v) model-based
> neuron importance agrees with the model-free criteria; (vi) the causal spectro-temporal population branch adds
> accuracy beyond a matched population-rate control; and (vii) a backbone trained on one dataset decodes the other
> above the random-K and label-permuted nulls after fitting only session adapters, with held-out units chosen
> label-free.

## Predictions

| id | prediction | comparator | statistic (per session, then across sessions) | fails if |
|---|---|---|---|---|
| **P1a** sparsity | linear decoder on the train-selected K units ≥ tuned linear decoder on all units | `logreg_selected_units` vs `logreg_all_units` (both in every run's `baselines`) | paired Δ balanced accuracy (Left/Right and 3-class) | CI of Δ includes −0.02 |
| **P1b** temporal / non-linear gain | DelayCAST on the selected K ≥ linear decoder on the selected K | `criteria` run vs `logreg_selected_units` | paired Δ balanced accuracy | CI of Δ includes 0 |
| **P2** not just loud units | criteria > rate-matched K > random K | `criteria` vs `rate` vs `random` runs (same seeds/splits) | paired Δ balanced accuracy, both contrasts | either CI includes 0 |
| **P3** late-delay sufficiency | τ<sub>95</sub> ≤ 500 ms for the model (bootstrap CI upper bound, full-context accuracy above chance) and for the linear all-unit sweep; occluding the last window of the delay (200 ms) costs more than occluding any earlier window | context sweep, `csi`, `tau95_linear_ms`, `temporal_occlusion` | CI upper bound of τ<sub>95</sub>; paired Δ(last window) − min Δ(earlier windows) | model at chance, upper CI > 500 ms, linear τ<sub>95</sub> > 500 ms, or an earlier window costs as much |
| **P4** delay → response coupling | forecast deviance explained (model) − (persistence) > 0; units with C over-represented among selected vs eligible-unselected | `forecast` per session; selection tables | paired Δ deviance explained; Fisher exact per session + sign test | CI includes 0 |
| **P5a** ALM dominance for direction | removing ALM input costs more Left/Right accuracy than removing striatal input | `region_ablation` (in-distribution region drop, and permutation) | paired Δ(ALM) − Δ(STR) per session | CI includes 0 |
| **P5b** striatum and no-lick trials *(exploratory)* | Ignore recall and its Wilson CI; striatal vs ALM Ignore-recall loss under ablation | `confusion`, `region_ablation.recall` | reported, no verdict below 30 Ignore test trials; flagged *confounded* when `logreg_trial_index` decodes Ignore above chance | — |
| **P6** model importance agrees with criteria | permutation importance and gates correlate with the criteria score within session × region | `importance_agreement` | mean Spearman ρ across cells, sign test | sign test p ≥ 0.05 or mean ρ ≤ 0 |
| **P7** spectro-temporal information | full model > matched population-mean control (`popmean`); `nospec` reported | `criteria` vs `criteria_popmean` (and `criteria_nospec`) | paired Δ balanced accuracy | CI includes 0 |
| **P8** transfer across recordings | cross-dataset transfer (adapter-only, label-free unit selection) above chance p95 and above the random-K arm under the same protocol | `cross_dataset/criteria` vs its `chance` and vs `cross_dataset/random` | pooled balanced accuracy vs p95; paired Δ vs random | below p95, or Δ CI includes 0 |
| **NC** negative control | labels permuted within session before selection, training and adaptation → chance | `negative_control/criteria` | pooled balanced accuracy vs its own chance p95 | above p95 (indicates leakage) |

Also reported without a verdict: stability of the selected units vs the label-permuted stability null, φ coefficients
between criteria (independence of the evidence), the L1-decoder / criteria-set Jaccard overlap, per-class recall under
region ablation, attention centre-of-mass (descriptive only), and seed-to-seed SD of every arm.

## Anchors

* Preparatory activity and ramping in ALM: Li, Chen, Guo, Svoboda (2015, 2016); Inagaki et al. (2019).
* Cortico-striatal contribution to delayed-response choice: ALM → dorsomedial striatum projections.
* Stability selection: Meinshausen & Bühlmann (2010).
* Permutation-based feature importance and in-distribution occlusion: Breiman (2001); Hooker & Mentch (2021).
