import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sarcomp.config import load_config  # noqa: E402
from sarcomp.smoke import Report, check_layers, check_transform, run_smoke_test  # noqa: E402


def test_transform_and_complex_layers():
    rep = Report(verbose=False)
    check_transform(rep)
    check_layers(rep)
    assert rep.ok, rep.summary()


def test_tiny_end_to_end():
    """All 4 stages through the real Trainer + checkpoints + pipelines + metrics, both architectures."""
    rep = run_smoke_test(device="cpu", size="tiny", verbose=False)
    assert rep.ok, rep.summary()


def test_config_inheritance():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
    cfg = load_config(os.path.join(root, "exp2_complex_ae.yaml"), ["train.lr=3e-4", "data.source=synthetic"])
    assert cfg["model"]["arch"] == "complex" and cfg["model"]["image_size"] == 256     # merged with base
    assert cfg["train"]["lr"] == 3e-4 and cfg["data"]["source"] == "synthetic"          # overrides applied
    fin = load_config(os.path.join(root, "exp4_complex_3_finetune.yaml"))
    assert fin["stage"] == "finetune" and fin["ckpt"]["ae"].endswith("exp2_complex_ae/last.pt")
