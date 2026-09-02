"""DelayCAST: Delay-epoch Causal Attention with Spectro-Temporal neuron selection.

Predict response-epoch (lick) population activity of four regions (left/right ALM,
left/right striatum) from the delay epoch, and classify the upcoming action
(Ignore / Lick-Left / Lick-Right) using selectively attended past context.
"""

__version__ = "0.1.0"

REGIONS = ("ALM_L", "ALM_R", "STR_L", "STR_R")
REGION_LABELS = {
    "ALM_L": "left ALM",
    "ALM_R": "right ALM",
    "STR_L": "left Striatum",
    "STR_R": "right Striatum",
}
REGION_COLORS = {
    "ALM_L": "#1f77b4",
    "ALM_R": "#ff7f0e",
    "STR_L": "#2ca02c",
    "STR_R": "#d62728",
}
CLASSES = ("Ignore", "Left", "Right")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
