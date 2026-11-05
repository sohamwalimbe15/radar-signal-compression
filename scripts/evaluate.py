"""Evaluate the FINAL model of an experiment (config of its last stage: ae or finetune).

python scripts/evaluate.py --config configs/exp2_complex_ae.yaml --split test
python scripts/evaluate.py --config configs/exp3_baseline_3_finetune.yaml
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402

from sarcomp.config import load_config  # noqa: E402
from sarcomp.data import build_data  # noqa: E402
from sarcomp.metrics import DEFAULT_METRICS, build_metrics, evaluate  # noqa: E402
from sarcomp.pipelines import build_pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--split", default=None)
    ap.add_argument("--ckpt", default=None, help="default: runs/<name>/last.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sam", action="store_true", help="also run the SAM agreement metrics (needs `transformers`)")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    ev = cfg.get("eval", {})
    split = a.split or ev.get("split", "test")
    loaders, tf = build_data(cfg["data"])
    pipe = build_pipeline(cfg, device, a.ckpt)
    metrics = build_metrics(ev.get("metrics") or DEFAULT_METRICS, tf)
    res = evaluate(pipe, loaders[split], metrics, device, ev.get("max_batches"), ev.get("seed", 0))
    if a.sam:
        from sarcomp.sam import build_backend, evaluate_sam
        ds = loaders[split].dataset
        assert hasattr(ds, "nx"), "SAM evaluation needs the TIFF grid dataset (data.source: tif)"
        res.update(evaluate_sam(pipe, lambda iy, ix: ds[iy * ds.nx + ix]["x"], (ds.ny, ds.nx), tf,
                                build_backend(cfg["sam"], device), cfg["sam"], cfg["data"]["pixel_spacing"],
                                device, ev.get("seed", 0)))
    x = next(iter(loaders[split]))["x"][:1].to(device)
    with torch.no_grad():
        code_values = pipe.code(x).numel()
    res.update({"code_values": code_values, "input_values": x.numel(), "compression_ratio_values": x.numel() / code_values,
                "params_total": sum(p.numel() for p in pipe.parameters())})
    print(json.dumps(res, indent=2))
    out = os.path.join(cfg.get("run_root", "runs"), cfg["name"], "metrics_%s.json" % split)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
