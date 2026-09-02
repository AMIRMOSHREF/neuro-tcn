from rodent_tcnn.config import load_config
from rodent_tcnn.constants import CLASSES, REGIONS
from rodent_tcnn.data.io_npz import load_trial_npz, split_by_region
from rodent_tcnn.data.synthetic import generate_trial
from rodent_tcnn.models.spec_tcnn import SPECTCNN


def test_synthetic_npz_schema():
    import numpy as np

    rng = np.random.default_rng(0)
    payload = generate_trial("Right", 8, rng, 1)
    assert set(REGIONS).issuperset(set(payload["brain_region"]))
    assert payload["delay_stop_times"] - payload["delay_start_times"] == 1.2
    assert len(payload["right_lick_times"]) > 0


def test_split_regions_and_forward(tmp_path):
    import numpy as np
    import torch

    rng = np.random.default_rng(1)
    payload = generate_trial("Left", 8, rng, 2)
    path = tmp_path / "t.npz"
    np.savez_compressed(path, **payload)
    data = load_trial_npz(path)
    rasters = split_by_region(data, float(data["delay_start_times"]), float(data["delay_stop_times"]), 40)
    assert rasters["left_ALM"]["raster"].shape[0] == 8
    cfg = load_config()
    cfg.model.units_per_region = 8
    cfg.epochs.lick_bins = 20
    model = SPECTCNN(cfg)
    x = torch.rand(2, 4, 8, 40)
    out = model(x, None)
    assert out["logits"].shape == (2, 3)
    assert out["y_lick"].shape[0] == 2
    assert out["neuron_attn"].shape == (2, 4, 8)


def test_class_names():
    assert CLASSES == ("Ignore", "Left", "Right")
