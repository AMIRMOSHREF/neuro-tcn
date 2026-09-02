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
    assert np.isclose(
        payload["delay_stop_times"] - payload["delay_start_times"],
        1.2,
    )
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


def test_csv_epochs_fill_missing_npz_timestamps(tmp_path):
    import numpy as np

    rng = np.random.default_rng(3)
    payload = generate_trial("Left", 8, rng, 3)
    metadata = {
        key: float(np.asarray(payload[key]))
        for key in (
            "trial_start",
            "trial_stop",
            "presample_start_times",
            "presample_stop_times",
            "sample_start_times",
            "sample_stop_times",
            "delay_start_times",
            "delay_stop_times",
            "go_start_times",
            "go_stop_times",
        )
    }
    payload["delay_start_times"] = np.asarray(np.nan)
    payload["go_start_times"] = np.asarray(np.nan)
    path = tmp_path / "missing_epochs.npz"
    np.savez_compressed(path, **payload)

    from delaycast.config import load_config as load_delaycast_config
    trial = load_trial_npz(path)
    from delaycast.data.rasters import load_trial_rasters

    cfg = load_delaycast_config(None)
    rasters = load_trial_rasters(path, cfg, metadata=metadata)
    assert np.isfinite(trial["spike_times"][0]).all()
    assert rasters.context["ALM_L"].shape[1] == 120
    assert rasters.target["ALM_L"].shape[1] == 30


def test_class_names():
    assert CLASSES == ("Ignore", "Left", "Right")
