"""python scripts/train.py --config configs/exp1_baseline_ae.yaml [--set train.max_steps=100 data.source=synthetic]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402

from sarcomp.config import load_config  # noqa: E402
from sarcomp.data import build_data  # noqa: E402
from sarcomp.training import STAGES, Trainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.lr=3e-4")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    torch.manual_seed(cfg.get("seed", 0))
    loaders, _ = build_data(cfg["data"])
    stage = STAGES.build(cfg["stage"], cfg, device)
    run_dir = os.path.join(cfg.get("run_root", "runs"), cfg["name"])
    print("stage=%s  run_dir=%s  device=%s" % (cfg["stage"], run_dir, device))
    Trainer(stage, loaders, cfg, run_dir, device).fit()


if __name__ == "__main__":
    main()
