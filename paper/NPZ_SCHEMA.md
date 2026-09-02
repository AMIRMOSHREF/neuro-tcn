# NPZ Analysis Guide — Multi-Region Raster Extraction for GAN Training

## Overview

Each NPZ file in this pipeline contains the full spiking population for one trial, along with
per-unit brain region labels and all behavioral epoch timestamps. This guide explains how to:

1. Load and inspect the NPZ contents
2. Split the combined raster into four brain-region-specific rasters
3. Structure the four region rasters as separate GAN inputs, paired with one video as the target output

---

## 1. NPZ File Contents

Each trial NPZ contains the following keys:

| Key | Shape | Description |
|---|---|---|
| `unit_ids` | (n_units,) | Neuron/unit IDs recorded in this session |
| `brain_region` | (n_units,) | Brain region label per unit (e.g. "left ALM", "right ALM", "left Striatum", "right Striatum") |
| `spike_times` | (n_units,) object array | Spike times per unit, cropped to this trial's window (absolute time) |
| `trial_start` / `trial_stop` | scalar | Trial boundary times |
| `presample_start_times` / `presample_stop_times` | scalar | Presample epoch boundaries |
| `sample_start_times` / `sample_stop_times` | scalar | Sample epoch boundaries |
| `delay_start_times` / `delay_stop_times` | scalar | Delay epoch boundaries |
| `go_start_times` / `go_stop_times` | scalar | Go/response epoch boundaries |
| `left_lick_times` / `right_lick_times` | (n_licks,) | Lick timestamps, empty if no licks occurred (e.g. ignore trials) |

### Basic inspection
```python
import numpy as np

data = np.load("trial_XXX.npz", allow_pickle=True)
for key in data.files:
    arr = data[key]
    print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
```

### Checking which brain regions are present
```python
unique_regions, counts = np.unique(data["brain_region"], return_counts=True)
for region, count in zip(unique_regions, counts):
    print(f"{region}: {count} units")
```
Expected output should show four region groups, e.g.:
```
left ALM: ~480 units
right ALM: ~460 units
left Striatum: ~510 units
right Striatum: ~502 units
```
If you see an "unknown" region, that means some units did not match the region-parsing regex
during extraction and should be reviewed before training.

---

## 2. Splitting One Trial Into Four Region-Specific Rasters

The core idea: instead of one combined raster (all units stacked together), we create four
independent rasters — one per brain region — each containing only the units belonging to that region.

```python
import numpy as np

def split_by_region(npz_path, bin_size=0.01):
    data = np.load(npz_path, allow_pickle=True)

    trial_start = float(data["trial_start"])
    trial_stop  = float(data["trial_stop"])
    duration    = trial_stop - trial_start

    n_bins = int(np.ceil(duration / bin_size))
    bin_edges = np.linspace(trial_start, trial_stop, n_bins + 1)

    regions = data["brain_region"]
    spike_times = data["spike_times"]
    unique_regions = np.unique(regions)

    region_rasters = {}

    for region in unique_regions:
        unit_mask = (regions == region)
        unit_indices = np.where(unit_mask)[0]

        raster = np.zeros((len(unit_indices), n_bins), dtype=np.float32)

        for row_idx, unit_idx in enumerate(unit_indices):
            spikes = np.asarray(spike_times[unit_idx])
            spikes = spikes[(spikes >= trial_start) & (spikes <= trial_stop)]
            counts, _ = np.histogram(spikes, bins=bin_edges)
            raster[row_idx, :] = counts

        region_rasters[region] = raster

    return region_rasters, bin_edges

rasters, bin_edges = split_by_region("trial_010.npz", bin_size=0.01)

for region, raster in rasters.items():
    print(f"{region}: raster shape = {raster.shape}  (units x time_bins)")
```

This produces a dictionary like:
```python
{
    "left ALM":       array of shape (n_units_leftALM, n_bins),
    "right ALM":      array of shape (n_units_rightALM, n_bins),
    "left Striatum":  array of shape (n_units_leftStriatum, n_bins),
    "right Striatum": array of shape (n_units_rightStriatum, n_bins),
}
```

### Handling variable unit counts across trials/sessions
Since different sessions may have different numbers of units per region, pad or truncate to a
fixed size per region before batching for the GAN (standard practice for CNN/RNN inputs):

```python
def pad_or_crop(raster, target_units):
    n_units = raster.shape[0]
    if n_units == target_units:
        return raster
    if n_units > target_units:
        return raster[:target_units, :]
    pad_amount = target_units - n_units
    return np.pad(raster, ((0, pad_amount), (0, 0)), mode="constant")

FIXED_UNITS_PER_REGION = {
    "left ALM": 128,
    "right ALM": 128,
    "left Striatum": 128,
    "right Striatum": 128,
}

for region in rasters:
    rasters[region] = pad_or_crop(rasters[region], FIXED_UNITS_PER_REGION[region])
```

---

## 3. Pairing Four Region Rasters With One Video (GAN Input Structure)

For your GAN, each training sample becomes:

- **4 inputs**: `left_ALM_raster`, `right_ALM_raster`, `left_Striatum_raster`, `right_Striatum_raster`
  (each shaped `[fixed_units, n_time_bins]`)
- **1 target output**: the corresponding trial's video clip (`trial_XXX.avi` from your video
  extraction pipeline, matched by trial number and class folder)

### Building one combined training sample
```python
import os
import cv2

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)

def build_training_sample(npz_path, video_path, bin_size=0.01):
    rasters, bin_edges = split_by_region(npz_path, bin_size=bin_size)
    for region in rasters:
        rasters[region] = pad_or_crop(rasters[region], FIXED_UNITS_PER_REGION[region])

    video_frames = load_video_frames(video_path)

    sample = {
        "left_ALM":       rasters["left ALM"],
        "right_ALM":      rasters["right ALM"],
        "left_Striatum":  rasters["left Striatum"],
        "right_Striatum": rasters["right Striatum"],
        "video":          video_frames,
    }
    return sample
```

### Suggested GAN generator structure
Since you have four distinct input streams feeding one video output, a multi-branch
generator is the natural architecture:

```
left_ALM_raster ----> Encoder_A --\
right_ALM_raster ---> Encoder_B ---\
left_Striatum_raster -> Encoder_C ---> Fusion Layer (concat/attention) --> Video Decoder --> Generated video
right_Striatum_raster -> Encoder_D --/
```

Each `Encoder_X` can be a shared architecture (e.g. Conv1D + LSTM/Transformer over the time
axis) applied independently per region, so the model learns region-specific temporal features
before fusing them.

---

## 4. Batch Dataset Loader Template

```python
import glob

class RegionRasterVideoDataset:
    def __init__(self, npz_dir, video_dir, bin_size=0.01):
        self.npz_files = sorted(glob.glob(os.path.join(npz_dir, "trial_*.npz")))
        self.video_dir = video_dir
        self.bin_size = bin_size

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_path = self.npz_files[idx]
        trial_num = os.path.basename(npz_path).replace("trial_", "").replace(".npz", "")
        video_path = os.path.join(self.video_dir, f"trial_{trial_num}.avi")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"No matching video for trial {trial_num}")

        return build_training_sample(npz_path, video_path, bin_size=self.bin_size)
```

---

## 5. Quick Sanity Checks Before Training

Run these on a handful of trials before committing to a full training run:

```python
sample = build_training_sample("trial_010.npz", "trial_010.avi")

for key in ["left_ALM", "right_ALM", "left_Striatum", "right_Striatum"]:
    print(f"{key}: shape={sample[key].shape}, "
          f"total_spikes={sample[key].sum():.0f}")

print(f"video: shape={sample['video'].shape}")
```

Confirm:
- All four raster arrays have the same time-bin dimension (n_bins) so they align temporally
- `total_spikes` is non-zero for each region (a region with all zeros likely means the
  region label didn't match any units — check `brain_region` values again)
- Video frame count roughly matches `n_bins` in temporal resolution, or note the ratio if
  the raster is sampled at a different rate than the video FPS

---

## Notes and Known Pitfalls

- **"unknown" region label**: if the regex parser in the extraction script failed to match a
  unit's electrode metadata, that unit gets labeled "unknown" — decide whether to drop these
  units or investigate the source NWB electrodes table before training.
- **Empty regions in some trials**: not every session guarantees units in all four regions;
  verify region coverage across your full trial set before building a fixed-size GAN input.
- **FPS mismatch between raster bins and video frames**: your video clips were extracted using
  real (non-rounded) FPS per trial, while raster bins use a fixed `bin_size` — these are
  different temporal resolutions and should be resampled to a common time base before feeding
  both into the GAN.
- **left_lick_times / right_lick_times empty arrays**: this simply reflects ignore trials
  with no lick response — not an error.


---

## 6. Visualizing Four Region Rasters Side-by-Side

Before committing to a full training run, it helps to visually QA a batch of samples —
confirming each region raster looks reasonable (not empty, not all noise) and that all four
panels align in time with each other and with the epoch markers.

```python
import numpy as np
import matplotlib.pyplot as plt

REGION_COLORS = {
    "left ALM": "#1f77b4",
    "right ALM": "#ff7f0e",
    "left Striatum": "#2ca02c",
    "right Striatum": "#d62728",
}

def plot_four_region_rasters(npz_path, bin_size=0.01, save_path=None):
    data = np.load(npz_path, allow_pickle=True)

    trial_start = float(data["trial_start"])
    trial_stop  = float(data["trial_stop"])

    regions = data["brain_region"]
    spike_times = data["spike_times"]
    unique_regions = [r for r in REGION_COLORS.keys() if r in np.unique(regions)]

    epoch_markers = {
        "presample_start_times": "presample start",
        "sample_start_times": "sample start",
        "delay_start_times": "delay start",
        "go_start_times": "go start",
        "go_stop_times": "go stop",
    }

    fig, axes = plt.subplots(len(unique_regions), 1, figsize=(12, 10), sharex=True)
    if len(unique_regions) == 1:
        axes = [axes]

    for ax, region in zip(axes, unique_regions):
        unit_mask = (regions == region)
        unit_indices = np.where(unit_mask)[0]
        color = REGION_COLORS.get(region, "gray")

        for row_idx, unit_idx in enumerate(unit_indices):
            spikes = np.asarray(spike_times[unit_idx])
            spikes = spikes[(spikes >= trial_start) & (spikes <= trial_stop)]
            ax.vlines(spikes, row_idx, row_idx + 1, color=color, linewidth=0.5)

        for col, label in epoch_markers.items():
            if col in data.files:
                val = float(data[col])
                if not np.isnan(val):
                    ax.axvline(val, color="black", linestyle=":", linewidth=1.2, zorder=5)

        ax.set_ylabel(region, fontsize=9, color=color, fontweight="bold")
        ax.set_ylim(0, max(len(unit_indices), 1))
        ax.set_xlim(trial_start, trial_stop)

    for col, label in epoch_markers.items():
        if col in data.files:
            val = float(data[col])
            if not np.isnan(val):
                axes[0].text(val, max(len(np.where(regions == unique_regions[0])[0]), 1) * 1.05,
                             label, rotation=90, va="bottom", ha="center", fontsize=7)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Four-Region Raster — {npz_path}", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved -> {save_path}")

    plt.show()

plot_four_region_rasters("trial_010.npz", bin_size=0.01, save_path="trial_010_four_regions.png")
```

### What to look for when QA-ing a batch
- **Non-empty rows in every panel**: an entirely blank region panel usually means the region
  label didn't match any units for that session — revisit the `brain_region` extraction regex.
- **Epoch lines aligned across all four panels**: since all regions share the same trial
  timeline, the dotted epoch markers (presample/sample/delay/go) should land at the exact same
  x-position in every subplot — if they don't, there's a per-region time-shift bug.
- **Consistent firing density across trials of the same class**: spot-check a few `left` vs
  `right` vs `ignore` trials side-by-side — firing patterns should visibly differ by outcome
  class, which is the signal your GAN needs to learn from.

### Batch QA over multiple trials
```python
import glob

npz_files = sorted(glob.glob("NPZ_OUTPUT_DIR/trial_*.npz"))[:5]  # spot-check first 5

for npz_path in npz_files:
    trial_id = os.path.basename(npz_path).replace(".npz", "")
    plot_four_region_rasters(npz_path, save_path=f"qa_{trial_id}.png")
```
This generates one PNG per trial so you can flip through a handful of samples quickly and
confirm the extraction pipeline is producing clean, well-aligned four-region rasters before
scaling up to full GAN training.
