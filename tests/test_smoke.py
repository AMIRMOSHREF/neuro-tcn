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


def test_data2_split_region_and_singular_epoch_schema(tmp_path):
    import numpy as np
    from delaycast.config import load_config as load_delaycast_config
    from delaycast.data.rasters import load_trial_rasters

    payload = generate_trial("Right", 8, np.random.default_rng(5), 4)
    regions = np.asarray(payload.pop("brain_region")).astype(str)
    spikes = np.asarray(payload.pop("spike_times"), dtype=object)
    payload.pop("unit_ids")
    for label, key in {
        "left ALM": "left_ALM_spikes",
        "right ALM": "right_ALM_spikes",
        "left Striatum": "left_Striatum_spikes",
        "right Striatum": "right_Striatum_spikes",
    }.items():
        payload[key] = spikes[regions == label]
    for plural, singular in {
        "trial_start": "start_time",
        "trial_stop": "stop_time",
        "presample_start_times": "presample_start_time",
        "presample_stop_times": "presample_stop_time",
        "sample_start_times": "sample_start_time",
        "sample_stop_times": "sample_stop_time",
        "delay_start_times": "delay_start_time",
        "delay_stop_times": "delay_stop_time",
        "go_start_times": "go_start_time",
        "go_stop_times": "go_stop_time",
    }.items():
        payload[singular] = payload.pop(plural)
    path = tmp_path / "data2_trial.npz"
    np.savez_compressed(path, **payload)

    rasters = load_trial_rasters(path, load_delaycast_config(None))
    assert all(rasters.context[region].shape == (8, 120) for region in ("ALM_L", "ALM_R", "STR_L", "STR_R"))
    assert all(rasters.target[region].shape == (8, 30) for region in ("ALM_L", "ALM_R", "STR_L", "STR_R"))


def test_class_names():
    assert CLASSES == ("Ignore", "Left", "Right")


def test_delaycast_default_config_contract():
    """The defaults the verification tests (causality / selection / pipeline) and the report rely on."""
    from delaycast import CLASSES as DC_CLASSES, REGIONS as DC_REGIONS
    from delaycast.config import load_config as load_delaycast_config

    cfg = load_delaycast_config(None)
    assert DC_REGIONS == ("ALM_L", "ALM_R", "STR_L", "STR_R") and DC_CLASSES == ("Ignore", "Left", "Right")
    assert int(cfg.selection.min_criteria) == 2 and float(cfg.selection.min_stability) == 0.6
    assert int(cfg.selection.top_k_per_region) == 32 and float(cfg.selection.fdr_q) == 0.05
    assert set(cfg.selection.bands_hz) == {"slow", "theta", "beta"}
    assert cfg.model.spectral_branch == "bands" and cfg.model.neuron_gate_penalty == "hoyer"
    assert int(cfg.data.bin_ms) == 10 and int(cfg.data.target_bin_ms) == 50
    assert int(cfg.data.context.delay_ms) // int(cfg.data.bin_ms) == 120
    assert int(cfg.data.target.response_ms) // int(cfg.data.target_bin_ms) == 30
    # dotted overrides parse scalars and create nested paths
    cfg2 = load_delaycast_config(None, ["model.d_model=32", "train.epochs=4", "evaluate.new.flag=true"])
    assert cfg2.model.d_model == 32 and cfg2.train.epochs == 4 and cfg2.get_path("evaluate.new.flag") is True
