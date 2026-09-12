from labsim import ExperimentConfig
from labsim.config_io import load_config, save_config


def test_config_round_trip(tmp_path):
    original = ExperimentConfig(t0=2.0, duration=4.0, dt=0.02, method="euler")
    path = tmp_path / "experiment.json"

    save_config(original, path)
    loaded = load_config(path)

    assert loaded == original


def test_saved_config_is_human_readable(tmp_path):
    path = tmp_path / "experiment.json"
    save_config(ExperimentConfig(duration=1.0, dt=0.1), path)
    text = path.read_text(encoding="utf-8")
    assert '"duration": 1.0' in text
    assert '"method": "rk4"' in text
