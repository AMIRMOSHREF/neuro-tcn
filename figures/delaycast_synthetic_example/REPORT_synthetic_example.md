# DelayCAST claims report

generated from `/tmp/claude-0/-home-user-neuro-tcn/45bbd087-4c58-57cb-936a-9f32df8042b3/scratchpad/out_e2e2`

## 1. Data and runs found

* sessions: **4** (dataset A: 2, dataset B: 2)
* animals: dataset A 2 (Session1, Session2); dataset B 2 (sub-990000, sub-990001)
* seeds present: 0, 1
* test trials per class (criteria, first seed): Ignore = 8, Left = 24, Right = 24

| run | seeds | n_sessions | balanced acc. | bal. acc. L/R | chance p95 | n test trials |
| --- | --- | --- | --- | --- | --- | --- |
| criteria | 0, 1 | 4 | 0.674 | 0.729 | 0.438 | 56 |
| criteria_popmean | 0, 1 | 4 | 0.708 | 0.750 | 0.438 | 56 |
| cross_dataset/criteria | 0 | 4 | 0.424 | 0.433 | 0.374 | 256 |
| cross_dataset/random | 0 | 4 | 0.408 | 0.393 | 0.365 | 256 |
| negative_control/criteria | 0 | 4 | 0.319 | 0.354 | 0.417 | 56 |
| random | 0, 1 | 4 | 0.417 | 0.562 | 0.382 | 56 |
| rate | 0, 1 | 4 | 0.528 | 0.667 | 0.424 | 56 |


## 2. The claim and the verdicts

> During the 1.2 s delay of the auditory delayed-response task, (i) a criterion-selected subset of at most K units per region, chosen on training trials only by model-free single-unit statistics that survive stability selection, supports decoding of the upcoming lick direction with balanced accuracy not lower than a tuned linear decoder on all recorded units (P1a: linear on selected K >= linear on all units; P1b: DelayCAST on selected K >= linear on selected K) and above rate-matched and random subsets of the same size (P2); (ii) the last 500 ms before the go cue retain >= 95 % of full-delay accuracy for both the selected-unit model and the all-unit linear decoder, and removing the last 400 ms costs more than removing any earlier 400 ms (P3); (iii) the selected units' late-delay activity forecasts their own response-epoch activity beyond a persistence baseline (P4); (iv) removing ALM input degrades Left/Right decoding more than removing striatal input (P5a); striatal involvement in no-lick (Ignore) trials is exploratory (P5b); (v) model-based importance agrees with the model-free criteria (P6); (vi) the causal spectro-temporal population branch adds accuracy beyond a matched population-rate control (P7); (vii) a backbone trained on one dataset decodes the other above the random-K and label-permuted nulls after fitting only session adapters (P8). Negative control (labels permuted before selection/training) must be at chance (P0).

Statistical rule: per-session paired differences (mean over seeds per session first), Wilcoxon signed-rank across sessions (exact for n <= 25), 1000-resample session bootstrap CI of the mean difference. 'supported' needs p < 0.05 AND a CI excluding 0 in the predicted direction (non-inferiority claims: CI lower bound > -0.02); 'inconclusive' when >= 5 sessions but the rule fails; 'not testable' when < 5 sessions; 'not run' when a run is missing.

| prediction | comparator | statistic | failure condition | result numbers | verdict |
| --- | --- | --- | --- | --- | --- |
| P0: Negative control at chance | negative_control/criteria vs its own label-permutation chance | pooled balanced accuracy <= chance p95 (per seed) | any seed with balanced accuracy > chance p95 | seed0: bacc=0.319 p95=0.417 pass | **supported** |
| P1a: Linear decoder on selected K not lower than linear on all units | baseline logreg_selected_units vs logreg_all_units (criteria run) | paired balanced_accuracy_lr difference, non-inferiority margin 0.02 | p >= 0.05 or CI lower bound <= -0.02 | A=0.979 B=1.000 diff=-0.021 CI=[-0.042, 0.000] p=0.812 n_sessions=4 n_seeds=2 | **not testable** |
| P1b: DelayCAST on selected K not lower than linear on selected K | run criteria vs baseline logreg_selected_units | paired balanced_accuracy_lr difference, non-inferiority margin 0.02 | p >= 0.05 or CI lower bound <= -0.02 | A=0.729 B=0.979 diff=-0.250 CI=[-0.292, -0.208] p=1.000 n_sessions=4 n_seeds=2 | **not testable** |
| P2: Criteria subset above rate-matched and random subsets | criteria vs rate; criteria vs random | paired balanced_accuracy difference > 0 (both) | either comparison with p >= 0.05 or CI lower bound <= 0 | vs rate: A=0.674 B=0.528 diff=0.146 CI=[-0.007, 0.250] p=0.125 n_sessions=4 n_seeds=2 -> not testable || vs random: A=0.674 B=0.417 diff=0.257 CI=[0.153, 0.382] p=0.062 n_sessions=4 n_seeds=2 -> not testable | **not testable** |
| P3: Last 500 ms retain >= 95 % of accuracy; last window costs most | criteria run: CSI tau95 CI (full-context accuracy above chance p95); temporal occlusion last window vs worst earlier window | tau95 CI upper <= 500 ms (all seeds, full-context accuracy > chance p95) AND paired delta(last) - min delta(earlier) < 0 | any seed at chance or with tau95 CI upper > 500 ms, or occlusion test p >= 0.05 / CI upper >= 0 | tau95 CI upper per seed: 700, 600 ms (FAIL); occlusion A=-1.4e-17 B=-0.257 diff=0.257 CI=[0.139, 0.368] p=1.000 n_sessions=4 n_seeds=2 -> not testable; linear tau95: 200, 100 ms | **inconclusive** |
| P4: Late-delay activity forecasts response-epoch activity beyond persistence | criteria forecast deviance explained: model vs persistence (mean over regions) | paired per-session difference > 0 | p >= 0.05 or CI lower bound <= 0 | A=0.021 B=-0.794 diff=0.815 CI=[0.760, 0.870] p=0.062 n_sessions=4 n_seeds=2; coupled fraction selected=0.428 vs unselected=0.218 sign-test p=0.062 | **not testable** |
| P5a: Removing ALM hurts Left/Right decoding more than removing striatum | criteria region_ablation (drop; fallback permute): mean delta ALM vs mean delta STR | paired per-session difference (ALM - STR) < 0 | p >= 0.05 or CI upper bound >= 0 | A=-0.094 B=-0.108 diff=0.014 CI=[-0.132, 0.118] p=0.688 n_sessions=4 n_seeds=2 (method=drop) | **not testable** |
| P5b: Ignore-trial decodability (exploratory) | criteria pooled Ignore recall vs 1/3 (uniform guess); trial-index drift control | Wilson 95% CI of Ignore recall | n_Ignore < 30 (not testable); CI lower bound <= 1/3 | Ignore recall=0.562 Wilson CI=[0.215, 0.785] n_Ignore=8 n_seeds=2; trial-index bacc=0.250 vs chance p95=0.438 | **not testable** |
| P6: Model-based importance agrees with the model-free criteria | importance_agreement.importance_vs_score (gate_vs_score reported) | mean_rho > 0 and sign_test_p < 0.05 with n_cells >= 8 (every seed) | any seed with mean_rho <= 0 or sign_test_p >= 0.05; n_cells < 8 in all seeds -> not testable | importance_vs_score: seed0 rho=0.500 p=1.000 n_cells=1, seed1 rho=-0.419 p=0.250 n_cells=3 -> not testable; gate_vs_score: seed0 rho=0.000 p=1.000 n_cells=1, seed1 rho=-0.557 p=0.250 n_cells=3 -> not testable | **not testable** |
| P7: Spectro-temporal population branch adds accuracy beyond the population-rate control | criteria vs criteria_popmean | paired balanced_accuracy difference > 0 | p >= 0.05 or CI lower bound <= 0 | A=0.674 B=0.708 diff=-0.035 CI=[-0.118, 0.028] p=0.688 n_sessions=4 n_seeds=2 | **not testable** |
| P8: Cross-dataset transfer above chance and above random-K after adapter fitting | cross_dataset/criteria vs chance p95 (per seed) and vs cross_dataset/random (paired per session) | pooled balanced accuracy > chance p95 (all seeds) AND paired difference > 0 | any seed at/below chance p95, or paired test p >= 0.05 / CI lower bound <= 0 | seed0: bacc=0.424 p95=0.374 pass || vs random: A=0.424 B=0.408 diff=0.016 CI=[-0.071, 0.125] p=0.562 n_sessions=4 n_seeds=1 -> not testable | **not testable** |


## 3. Predictions in detail

### P0. Negative control at chance

comparator: negative_control/criteria vs its own label-permutation chance; statistic: pooled balanced accuracy <= chance p95 (per seed); fails if: any seed with balanced accuracy > chance p95

| seed | balanced accuracy | chance mean | chance p95 | pass |
| --- | --- | --- | --- | --- |
| 0 | 0.319 | 0.334 | 0.417 | True |

**Verdict P0: supported.**

### P1a. Linear decoder on selected K not lower than linear on all units

comparator: baseline logreg_selected_units vs logreg_all_units (criteria run); statistic: paired balanced_accuracy_lr difference, non-inferiority margin 0.02; fails if: p >= 0.05 or CI lower bound <= -0.02

| session | logreg_selected_units | logreg_all_units | difference (A - B) |
| --- | --- | --- | --- |
| A/Session1 | 0.958 | 1.000 | -0.042 |
| A/Session2 | 1.000 | 1.000 | 0.000 |
| B/sub-990000_ses-20190301T120000 | 1.000 | 1.000 | 0.000 |
| B/sub-990001_ses-20190302T120000 | 0.958 | 1.000 | -0.042 |

metric: `balanced_accuracy_lr` | mean A = 0.979, mean B = 1.000, mean difference = -0.021 | n_sessions = 4 | n_seeds = 2 (A 2, B 2) | Wilcoxon p = 0.812 | bootstrap 95% CI = [-0.042, 0.000] | direction `not_lower`

**Verdict P1a: not testable** (fewer than 5 sessions or too few trials/cells). Mean paired difference -0.021, 95% CI [-0.042, 0.000], Wilcoxon p = 0.812, n_sessions = 4, n_seeds = 2.

### P1b. DelayCAST on selected K not lower than linear on selected K

comparator: run criteria vs baseline logreg_selected_units; statistic: paired balanced_accuracy_lr difference, non-inferiority margin 0.02; fails if: p >= 0.05 or CI lower bound <= -0.02

| session | criteria (DelayCAST) | logreg_selected_units | difference (A - B) |
| --- | --- | --- | --- |
| A/Session1 | 0.667 | 0.958 | -0.292 |
| A/Session2 | 0.792 | 1.000 | -0.208 |
| B/sub-990000_ses-20190301T120000 | 0.708 | 1.000 | -0.292 |
| B/sub-990001_ses-20190302T120000 | 0.750 | 0.958 | -0.208 |

metric: `balanced_accuracy_lr` | mean A = 0.729, mean B = 0.979, mean difference = -0.250 | n_sessions = 4 | n_seeds = 2 (A 2, B 2) | Wilcoxon p = 1.000 | bootstrap 95% CI = [-0.292, -0.208] | direction `not_lower`

**Verdict P1b: not testable** (fewer than 5 sessions or too few trials/cells). Mean paired difference -0.250, 95% CI [-0.292, -0.208], Wilcoxon p = 1.000, n_sessions = 4, n_seeds = 2.

### P2. Criteria subset above rate-matched and random subsets

comparator: criteria vs rate; criteria vs random; statistic: paired balanced_accuracy difference > 0 (both); fails if: either comparison with p >= 0.05 or CI lower bound <= 0

**criteria vs rate**

| session | criteria | rate | difference (A - B) |
| --- | --- | --- | --- |
| A/Session1 | 0.611 | 0.444 | 0.167 |
| A/Session2 | 0.778 | 0.500 | 0.278 |
| B/sub-990000_ses-20190301T120000 | 0.472 | 0.556 | -0.083 |
| B/sub-990001_ses-20190302T120000 | 0.833 | 0.611 | 0.222 |

metric: `balanced_accuracy` | mean A = 0.674, mean B = 0.528, mean difference = 0.146 | n_sessions = 4 | n_seeds = 2 (A 2, B 2) | Wilcoxon p = 0.125 | bootstrap 95% CI = [-0.007, 0.250] | direction `>`

sub-verdict: not testable

**criteria vs random**

| session | criteria | random | difference (A - B) |
| --- | --- | --- | --- |
| A/Session1 | 0.611 | 0.333 | 0.278 |
| A/Session2 | 0.778 | 0.583 | 0.194 |
| B/sub-990000_ses-20190301T120000 | 0.472 | 0.361 | 0.111 |
| B/sub-990001_ses-20190302T120000 | 0.833 | 0.389 | 0.444 |

metric: `balanced_accuracy` | mean A = 0.674, mean B = 0.417, mean difference = 0.257 | n_sessions = 4 | n_seeds = 2 (A 2, B 2) | Wilcoxon p = 0.062 | bootstrap 95% CI = [0.153, 0.382] | direction `>`

sub-verdict: not testable

**Verdict P2: not testable** (fewer than 5 sessions or too few trials/cells).

### P3. Last 500 ms retain >= 95 % of accuracy; last window costs most

comparator: criteria run: CSI tau95 CI (full-context accuracy above chance p95); temporal occlusion last window vs worst earlier window; statistic: tau95 CI upper <= 500 ms (all seeds, full-context accuracy > chance p95) AND paired delta(last) - min delta(earlier) < 0; fails if: any seed at chance or with tau95 CI upper > 500 ms, or occlusion test p >= 0.05 / CI upper >= 0

**(i) context sufficiency index (criteria run)**

| seed | tau95 (ms) | tau95 CI (ms) | CI upper <= 500 |
| --- | --- | --- | --- |
| 0 | 500 | [400, 700] | False |
| 1 | 500 | [400, 600] | False |


**(ii) temporal occlusion: last window vs mean of earlier windows** (pooled worst window per seed: seed0 end=800 ms, seed1 end=1000 ms)

| session | delta last window | mean delta earlier windows | difference (A - B) |
| --- | --- | --- | --- |
| A/Session1 | -0.028 | -0.250 | 0.222 |
| A/Session2 | 0.056 | -0.361 | 0.417 |
| B/sub-990000_ses-20190301T120000 | -0.028 | -0.111 | 0.083 |
| B/sub-990001_ses-20190302T120000 | 0.000 | -0.306 | 0.306 |

metric: `delta_balanced_accuracy` | mean A = -1.4e-17, mean B = -0.257, mean difference = 0.257 | n_sessions = 4 | n_seeds = 2 (A 2, B 2) | Wilcoxon p = 1.000 | bootstrap 95% CI = [0.139, 0.368] | direction `<`


**(iii) linear decoder tau95 (reported)**: seed0 = 200 ms, seed1 = 100 ms

**Verdict P3: inconclusive** (test possible but the rule was not met). Mean paired difference 0.257, 95% CI [0.139, 0.368], Wilcoxon p = 1.000, n_sessions = 4, n_seeds = 2.

### P4. Late-delay activity forecasts response-epoch activity beyond persistence

comparator: criteria forecast deviance explained: model vs persistence (mean over regions); statistic: paired per-session difference > 0; fails if: p >= 0.05 or CI lower bound <= 0

| session | model dev. expl. | persistence dev. expl. | difference (A - B) |
| --- | --- | --- | --- |
| A/Session1 | 0.043 | -0.847 | 0.890 |
| A/Session2 | 0.005 | -0.845 | 0.851 |
| B/sub-990000_ses-20190301T120000 | 0.022 | -0.764 | 0.786 |
| B/sub-990001_ses-20190302T120000 | 0.014 | -0.721 | 0.735 |

metric: `deviance_explained (mean over regions)` | mean A = 0.021, mean B = -0.794, mean difference = 0.815 | n_sessions = 4 | n_seeds = 2 (A 2, B 2) | Wilcoxon p = 0.062 | bootstrap 95% CI = [0.760, 0.870] | direction `>`


**Coupling (C) enrichment among selected units (reported)**

| session | frac C selected | frac C eligible-unselected | difference | Fisher p (seed 0) |
| --- | --- | --- | --- | --- |
| A/Session1 | 0.357 | 0.167 | 0.190 | 0.783 |
| A/Session2 | 0.516 | 0.196 | 0.320 | 0.310 |
| B/sub-990000_ses-20190301T120000 | 0.410 | 0.343 | 0.067 | 0.372 |
| B/sub-990001_ses-20190302T120000 | 0.431 | 0.167 | 0.264 | 0.264 |


sign test across sessions: 4/4 positive, p = 0.062

**Verdict P4: not testable** (fewer than 5 sessions or too few trials/cells). Mean paired difference 0.815, 95% CI [0.760, 0.870], Wilcoxon p = 0.062, n_sessions = 4, n_seeds = 2.

### P5a. Removing ALM hurts Left/Right decoding more than removing striatum

comparator: criteria region_ablation (drop; fallback permute): mean delta ALM vs mean delta STR; statistic: paired per-session difference (ALM - STR) < 0; fails if: p >= 0.05 or CI upper bound >= 0

| session | mean delta ALM removed | mean delta STR removed | difference (A - B) |
| --- | --- | --- | --- |
| A/Session1 | -0.042 | -0.069 | 0.028 |
| A/Session2 | -0.306 | -0.097 | -0.208 |
| B/sub-990000_ses-20190301T120000 | 0.153 | 0.056 | 0.097 |
| B/sub-990001_ses-20190302T120000 | -0.181 | -0.319 | 0.139 |

metric: `delta_balanced_accuracy (drop)` | mean A = -0.094, mean B = -0.108, mean difference = 0.014 | n_sessions = 4 | n_seeds = 2 (A 2, B 2) | Wilcoxon p = 0.688 | bootstrap 95% CI = [-0.132, 0.118] | direction `<`

**Verdict P5a: not testable** (fewer than 5 sessions or too few trials/cells). Mean paired difference 0.014, 95% CI [-0.132, 0.118], Wilcoxon p = 0.688, n_sessions = 4, n_seeds = 2.

### P5b. Ignore-trial decodability (exploratory)

comparator: criteria pooled Ignore recall vs 1/3 (uniform guess); trial-index drift control; statistic: Wilson 95% CI of Ignore recall; fails if: n_Ignore < 30 (not testable); CI lower bound <= 1/3

Ignore recall = 0.562, Wilson 95% CI = [0.215, 0.785], n_Ignore = 8, n_seeds = 2; trial-index baseline balanced accuracy = 0.250 vs chance p95 = 0.438

**Verdict P5b: not testable** (fewer than 5 sessions or too few trials/cells).

### P6. Model-based importance agrees with the model-free criteria

comparator: importance_agreement.importance_vs_score (gate_vs_score reported); statistic: mean_rho > 0 and sign_test_p < 0.05 with n_cells >= 8 (every seed); fails if: any seed with mean_rho <= 0 or sign_test_p >= 0.05; n_cells < 8 in all seeds -> not testable

**importance_vs_score** (sub-verdict not testable)

| seed | mean rho | median rho | n_cells | n_positive | sign-test p | pass |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.500 | 0.500 | 1 | 1 | 1.000 | False |
| 1 | -0.419 | -0.500 | 3 | 0 | 0.250 | False |

**gate_vs_score** (sub-verdict not testable)

| seed | mean rho | median rho | n_cells | n_positive | sign-test p | pass |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.000 | 1 | 0 | 1.000 | False |
| 1 | -0.557 | -0.600 | 3 | 0 | 0.250 | False |

**Verdict P6: not testable** (fewer than 5 sessions or too few trials/cells).

### P7. Spectro-temporal population branch adds accuracy beyond the population-rate control

comparator: criteria vs criteria_popmean; statistic: paired balanced_accuracy difference > 0; fails if: p >= 0.05 or CI lower bound <= 0

| session | criteria | criteria_popmean | difference (A - B) |
| --- | --- | --- | --- |
| A/Session1 | 0.611 | 0.583 | 0.028 |
| A/Session2 | 0.778 | 0.806 | -0.028 |
| B/sub-990000_ses-20190301T120000 | 0.472 | 0.639 | -0.167 |
| B/sub-990001_ses-20190302T120000 | 0.833 | 0.806 | 0.028 |

metric: `balanced_accuracy` | mean A = 0.674, mean B = 0.708, mean difference = -0.035 | n_sessions = 4 | n_seeds = 2 (A 2, B 2) | Wilcoxon p = 0.688 | bootstrap 95% CI = [-0.118, 0.028] | direction `>`

**Verdict P7: not testable** (fewer than 5 sessions or too few trials/cells). Mean paired difference -0.035, 95% CI [-0.118, 0.028], Wilcoxon p = 0.688, n_sessions = 4, n_seeds = 2.

### P8. Cross-dataset transfer above chance and above random-K after adapter fitting

comparator: cross_dataset/criteria vs chance p95 (per seed) and vs cross_dataset/random (paired per session); statistic: pooled balanced accuracy > chance p95 (all seeds) AND paired difference > 0; fails if: any seed at/below chance p95, or paired test p >= 0.05 / CI lower bound <= 0

| seed | balanced accuracy | chance p95 | pass |
| --- | --- | --- | --- |
| 0 | 0.424 | 0.374 | True |


**vs cross_dataset/random**

| session | cross_dataset/criteria | cross_dataset/random | difference (A - B) |
| --- | --- | --- | --- |
| A/Session1 | 0.542 | 0.357 | 0.185 |
| A/Session2 | 0.476 | 0.530 | -0.054 |
| B/sub-990000_ses-20190301T120000 | 0.345 | 0.321 | 0.024 |
| B/sub-990001_ses-20190302T120000 | 0.333 | 0.423 | -0.089 |

metric: `balanced_accuracy` | mean A = 0.424, mean B = 0.408, mean difference = 0.016 | n_sessions = 4 | n_seeds = 1 (A 1, B 1) | Wilcoxon p = 0.562 | bootstrap 95% CI = [-0.071, 0.125] | direction `>`

**Verdict P8: not testable** (fewer than 5 sessions or too few trials/cells). Mean paired difference 0.016, 95% CI [-0.071, 0.125], Wilcoxon p = 0.562, n_sessions = 4, n_seeds = 1.


## 4. Selection summary

from `/tmp/claude-0/-home-user-neuro-tcn/45bbd087-4c58-57cb-936a-9f32df8042b3/scratchpad/out_e2e2/selection/summary.csv` (4 sessions, all trials): units recorded 289, pass floor 278, eligible 72, selected 64

| region | recorded | selected |
| --- | --- | --- |
| ALM_L | 72 | 21 |
| ALM_R | 73 | 15 |
| STR_L | 74 | 15 |
| STR_R | 70 | 13 |


* median stability of selected units: 0.950 (median over sessions) vs null median-stability max 0.050
* median onset of selected units: n/a ms; fraction sustained to go: 0.000
* criterion fractions among units (mean over sessions): selectivity 0.256, coupling 0.291, spectral 0.090, ramp 0.317, locus 0.000, ignore 0.229

train-split funnel of the criteria run (per region, summed over sessions, mean over 2 seed(s); K = 16):

| region | recorded | pass_floor | eligible | stable | selected |
| --- | --- | --- | --- | --- | --- |
| ALM_L | 72.0 | 70.0 | 23.0 | 15.5 | 15.5 |
| ALM_R | 73.0 | 68.0 | 17.0 | 10.0 | 10.0 |
| STR_L | 74.0 | 71.0 | 16.0 | 11.0 | 11.0 |
| STR_R | 70.0 | 68.0 | 14.0 | 10.5 | 10.5 |


* expected false-selection bound: mean per region-session 3.032, max 6.000

* phi coefficients between criteria (mean over sessions): phi_SC = 0.128, phi_SW = 0.532, phi_SR = 0.838, phi_CW = 0.087, phi_CR = 0.078, phi_WR = 0.454

**W-independence**: phi_SW mean = 0.532 (threshold 0.7) -> W is independent enough of S
