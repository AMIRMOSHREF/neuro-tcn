# SCOPE-DCC methodology

SCOPE-DCC (Sparse Context-Optimized Population Encoding with Dilated Causal
Convolutions) is a proposed analysis, not an established neuroscience result.
Its central test is whether delay-period population activity predicts both the
future response-period raster and the eventual Ignore/Left/Right action in
sessions not used to fit the model.

## Trial definition and harmonization

Each NPZ is split by its per-unit `brain_region` field into bilateral ALM and
striatum. Spikes are binned separately in:

- **input:** `delay_start_times` through `delay_stop_times`;
- **forecast target:** `go_start_times` through `go_stop_times`.

The target is response-epoch activity, not “activity after first lick.” Aligning
to the first lick would be undefined for Ignore trials and would leak class
information. Every epoch is resampled to a fixed number of bins while retaining
its true start and stop timestamps. The Data2 audited CSV takes precedence over
folder names. Audited misses, early licks, photostimulation, and water-assisted
trials should remain excluded. A hit is labelled Left or Right from
`trial_instruction`; an ignored trial is labelled Ignore.

Unit IDs are meaningful only inside a recording session. The model shares a
per-neuron encoder across sessions but never treats row 12 in two sessions as
the same biological neuron.

## Model

1. A shared per-neuron DCC encoder uses depthwise causal convolutions with
   exponentially increasing dilations. Its receptive field spans the delay
   without access to future bins.
2. A learned temporal attention distribution identifies delay subwindows used
   by each neuron.
3. A log-STFT branch summarizes transient and sustained firing-rate structure.
   This is a time-frequency description of binned point-process counts—not an
   LFP oscillation analysis.
4. A sparse sigmoid gate scores neurons. Masked pooling forms one token per
   region, followed by cross-region multi-head attention.
5. A shared point-process decoder forecasts every observed neuron's
   response-epoch rate. A second head classifies Ignore/Left/Right.

The objective is

`L = L_CE + λf L_Poisson + λs mean(gate) + λtv TV(temporal_attention)`.

The forecast auxiliary task discourages the classifier from selecting neurons
that are discriminative only through an unstable shortcut. Sparsity produces a
compact candidate ensemble; it does not establish causal necessity.

## Selection rule

Selection is computed using training-fold predictions only:

- a neuron is in the top configured gate fraction within a trial;
- its top-fraction membership is stable in at least 70% of trials in which it
  was observed;
- it has measurable delay firing;
- class modulation (one-way eta-squared) and preferred context bin are reported.

The CSV preserves session, region, and unit ID. For a paper, repeat model fitting
across seeds and bootstrap whole sessions. Call a unit stable only if it passes
in at least 70% of bootstrap fits, not merely trials.

## Validation required for a scientific claim

- Use nested grouped validation. The outer split holds out entire sessions (and,
  where possible, entire animals). Hyperparameters and neuron thresholds are
  chosen only in inner training-session splits.
- Primary classification endpoint: macro F1 and balanced accuracy with
  animal/session-clustered 95% bootstrap confidence intervals.
- Primary forecast endpoint: held-out Poisson deviance or pseudo-R² relative to
  a training-set mean-rate model.
- Build the null by permuting labels and response targets within session and
  trial type, refitting the complete selection procedure.
- Compare against majority class, multinomial logistic regression on mean delay
  rates, noncausal TCN, DCC without STFT, DCC without forecast loss, and the
  four leave-one-region-out models.
- Report calibration, per-class confusion matrices, performance by dataset,
  and cross-dataset transfer (train Data/test Data2 and the reverse).
- Correct region/window comparisons using a hierarchical model or false
  discovery rate control. The experimental unit for population claims is the
  animal/session, not each neuron or trial.

## Defensible claim template

Only after the tests above succeed:

> “Delay-period activity in a sparse bilateral ALM–striatal ensemble carried
> prospective information about the subsequent response. A model constrained
> to delay activity predicted held-out response-epoch firing and decoded
> Ignore/Left/Right choices above within-session permutation nulls in
> held-out sessions.”

Do not write “these neurons cause the action,” “the model predicts licking”
without a behavioral baseline, or “STFT reveals oscillations.” Attention and
selection identify predictive associations. Causal claims require perturbation
or an appropriate causal design.
