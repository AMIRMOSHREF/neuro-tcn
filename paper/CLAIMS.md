# Scientific claims

These are the statements the SPEC-TCNN analysis is designed to support. Each claim has a test that can be reported as a figure panel or table.

## C1. A sparse delay-period ensemble forecasts lick-period population activity

A minority of ALM and striatal units (~15–20% per region) carry a causal predictive code: their delay-epoch spike trains are sufficient to reconstruct lick-epoch rasters in all four regions. Dilated causal convolutions plus neuron attention recover this ensemble without leaking go-cue or lick-time information into the encoder.

**Testable prediction.** Occluding the selected set raises lick-raster Poisson NLL far more than occluding a size-matched random set. Classification accuracy should remain near the full-population ceiling when only selected units are kept.

**Anchors.** Li, Chen, Svoboda (preparatory ALM); Inagaki et al. (ramping / attractor dynamics).

## C2. Contralateral ALM is preferentially selected for upcoming lick direction

Neuron attention and d′ concentrate on left ALM before right licks and on right ALM before left licks. Ignore trials flatten this laterality. The result supports a hemispheric motor-plan code that is already resolved during the delay, not only at movement onset.

**Testable prediction.** A laterality index (selected contralateral minus ipsilateral ALM weight) is positive on Left/Right hits and near zero on Ignore. The same index computed on rejected units is weak.

**Anchors.** Li et al., Nature 2015/2016; Guo et al., ALM laterality and interhemispheric inhibition.

## C3. ALM delay context predicts striatal lick-period firing better than the reverse

Cross-region prediction (delay ALM → lick STR) outperforms delay STR → lick ALM. This is the expected cortico-striatal hierarchy: ALM holds the plan; striatum expresses the selected action once the go cue arrives.

**Testable prediction.** In a 2×2 transfer matrix of {ALM, STR} delay → {ALM, STR} lick, the ALM→STR off-diagonal exceeds STR→ALM. Attention mass in late delay sits more on ALM than on STR.

**Anchors.** ALM–dorsomedial striatum projections; same-task DMS choice signals.

## C4. The last ~300–400 ms of delay is the dominant causal context

Temporal attention, constrained to be causal, peaks in the pre-go ramp — not uniformly across the 1.2 s delay. That window is when preparatory activity becomes choice-specific enough to forecast both the lick-period burst and the 3-class action.

**Testable prediction.** Truncating the encoder to the first 800 ms of delay drops accuracy more than dropping the first 800 ms and keeping the last 400 ms. Attention center-of-mass lies after 0.8 s.

**Anchors.** Inagaki, Chen, Svoboda — discrete attractor + ramping into the go cue.

## C5. Delay β / low-γ wavelet structure marks the same units attention selects

Selected neurons show class-modulated Morlet CWT power in 12–45 Hz during delay. Fusing STFT/CWT with the spike TCNN improves both lick-raster prediction and Ignore/Left/Right decoding relative to a spike-only ablation.

**Testable prediction.** Δaccuracy of the TF branch is largest for ALM delay-choice cells and near zero for tonic unselective cells. Beta energy d′ correlates with neuron-attention rank.

**Anchors.** Motor beta as holding / status-quo; gamma as readout / movement preparation.

## C6. The selected set is shared across Data and Data2 after CSV-consistent labeling

When Data2 trials are labeled from audited master logs (exclude photostim, early lick, bilateral licks) and Data trials from class folders, the same functional types (delay-choice, delay-ramp) dominate selection in both corpora. That is evidence for a task-general cortico-striatal predictive subspace, not a session artifact.

**Testable prediction.** Jaccard overlap of selected functional types (not raw unit IDs) between datasets exceeds a label-shuffled null. Cross-dataset training (train Data, test Data2 and reverse) degrades less for the selected subset than for the full population.

**Anchors.** Multi-animal generalization of choice circuits (IBL-style).
