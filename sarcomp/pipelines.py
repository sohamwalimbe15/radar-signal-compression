"""Inference pipelines: x (B,2,H,W) -> reconstruction (B,2,H,W). The compressed code is latent2."""
import os

import torch
import torch.nn as nn

from .training.stages import FinetuneStage, load_ae


class AEPipeline(nn.Module):
    """exp 1/2:  E1 -> E2 -> D1 -> D2"""

    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def code(self, x):
        return self.ae.encode(x)[1]

    def forward(self, x, generator=None):
        return self.ae(x)["recon"]


class FMPipeline(nn.Module):
    """exp 3/4:  E1 -> E2 -> (flow matching: noise -> latent1 | latent2) -> D2"""

    def __init__(self, e1, e2, fm, d2, steps=20, solver="euler"):
        super().__init__()
        self.e1, self.e2, self.fm, self.d2, self.steps, self.solver = e1, e2, fm, d2, steps, solver

    def code(self, x):
        return self.e2(self.e1(x))

    def forward(self, x, generator=None):
        l1_hat = self.fm.sample(self.code(x), self.steps, self.solver, generator)
        return self.d2(l1_hat)


def build_pipeline(cfg, device, ckpt=None):
    """Build the final model of an experiment from its LAST stage config (`ae` or `finetune`).

    ckpt: trained checkpoint of that last stage (default: runs/<name>/last.pt).
    """
    ckpt = ckpt or os.path.join(cfg.get("run_root", "runs"), cfg["name"], "last.pt")
    if cfg["stage"] == "ae":
        ae, _ = load_ae(ckpt, device)
        return AEPipeline(ae.eval()).to(device)
    if cfg["stage"] == "finetune":
        cfg = {**cfg, "ckpt": {**cfg["ckpt"], "d2": ckpt}}
        st = FinetuneStage(cfg, device)
        return FMPipeline(st.e1, st.e2, st.fm, st.d2.eval(), st.steps, st.solver).to(device)
    raise ValueError("final-stage config must have stage: ae | finetune (got %s)" % cfg["stage"])
