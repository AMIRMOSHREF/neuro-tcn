from pathlib import Path

import torch

from neurocontext.config import load_config
from neurocontext.data import discover_trials, load_trial
from neurocontext.demo import generate_demo
from neurocontext.model import ContextForecaster, multitask_loss


def test_discovery_loading_and_forward(tmp_path: Path):
    base = load_config(Path(__file__).parents[1] / "config" / "default.yaml")
    paths = generate_demo(tmp_path, base, seed=4)
    config = load_config(paths["config"])
    records = discover_trials(
        paths["data_root"], paths["data2_root"], [paths["metadata"]]
    )
    assert len(records) == 108
    assert {record.dataset for record in records} == {"Data", "Data2"}
    assert {record.label for record in records} == {"Ignore", "Left", "Right"}

    sample = load_trial(records[0], config)
    assert sample["delay"].shape == (4, 8, 32)
    assert sample["response"].shape == (4, 8, 36)
    assert sample["unit_mask"].all()

    model = ContextForecaster(config)
    output = model(sample["delay"].unsqueeze(0), sample["unit_mask"].unsqueeze(0))
    assert output["logits"].shape == (1, 3)
    assert output["rates"].shape == (1, 4, 8, 36)
    assert torch.isfinite(output["rates"]).all()
    loss, parts = multitask_loss(
        output,
        sample["response"].unsqueeze(0),
        sample["label"].unsqueeze(0),
        sample["unit_mask"].unsqueeze(0),
        config,
    )
    assert torch.isfinite(loss)
    assert parts["forecast_loss"] > 0
