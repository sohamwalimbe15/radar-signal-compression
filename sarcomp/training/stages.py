"""Training stages = what is trained, what is frozen, and what the loss is.

  ae        exp 1/2          train E1 E2 D1 D2 end to end (reconstruction)
  fm        exp 3/4 step 1   frozen E1,E2 (from `ckpt.ae`); train flow matching  p(latent1 | latent2)
  e1d2      exp 3/4 step 2   frozen E1 (from `ckpt.ae`); train D2 to reconstruct from the TRUE latent1
  finetune  exp 3/4 step 3   frozen E1,E2,FM; train D2 (init `ckpt.d2`) to reconstruct from FM-GENERATED latent1

Add a stage: subclass Stage, @STAGES.register("name"), implement modules()/loss()/state().
"""
import itertools

import torch
import torch.nn.functional as F

from ..fm import build_fm
from ..models import build_autoencoder, build_d2, latent_info, make_ops
from ..utils import Registry
from .ema import EMA

STAGES = Registry("stage")
LOSSES = {"mse": F.mse_loss, "l1": F.l1_loss}


def freeze(m):
    return m.eval().requires_grad_(False)


def load_ae(path, device):
    ck = torch.load(path, map_location=device)
    ae = build_autoencoder(ck["model_cfg"]).to(device)
    ae.load_state_dict(ck["ae"])
    return ae, ck["model_cfg"]


def load_fm(path, device, use_ema=True):
    ck = torch.load(path, map_location=device)
    fm = build_fm(ck["fm_cfg"], latent_info(ck["model_cfg"])).to(device)
    fm.load_state_dict(ck["ema"] if (use_ema and ck.get("ema") is not None) else ck["fm"])
    return fm


class Stage:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device

    def modules(self):            # name -> trainable module
        raise NotImplementedError

    def setup(self, loaders):     # optional, e.g. fit normalisation stats
        pass

    def loss(self, batch):
        raise NotImplementedError

    def after_step(self):
        pass

    def state(self):
        raise NotImplementedError

    def load_state(self, s):
        raise NotImplementedError

    def trainable_parameters(self):
        return [p for m in self.modules().values() for p in m.parameters() if p.requires_grad]

    def set_train(self, train):
        for m in self.modules().values():
            m.train(train)


@STAGES.register("ae")
class AEStage(Stage):
    def __init__(self, cfg, device, ae=None):
        super().__init__(cfg, device)
        self.model_cfg = cfg["model"]
        self.ae = (ae if ae is not None else build_autoencoder(self.model_cfg)).to(device)
        self.loss_fn = LOSSES[cfg["train"].get("loss", "mse")]

    def modules(self):
        return {"ae": self.ae}

    def loss(self, batch):
        x = batch["x"].to(self.device)
        return self.loss_fn(self.ae(x)["recon"].float(), x.float())

    def state(self):
        return {"ae": self.ae.state_dict(), "model_cfg": self.model_cfg}

    def load_state(self, s):
        self.ae.load_state_dict(s["ae"])


@STAGES.register("fm")
class FMStage(Stage):
    def __init__(self, cfg, device, ae=None, model_cfg=None):
        super().__init__(cfg, device)
        if ae is None:
            ae, model_cfg = load_ae(cfg["ckpt"]["ae"], device)
        self.ae, self.model_cfg = freeze(ae.to(device)), model_cfg
        self.fm_cfg = cfg["fm"]
        self.fm = build_fm(self.fm_cfg, latent_info(model_cfg)).to(device)
        self.ema = EMA(self.fm, cfg["train"].get("ema", 0.999))

    def modules(self):
        return {"fm": self.fm}

    @torch.no_grad()
    def setup(self, loaders):
        """Fit per-channel mean/std of latent1 and of the latent2 condition on training data."""
        n, s1, q1, s2, q2, cnt1, cnt2 = self.fm_cfg.get("stats_batches", 20), 0, 0, 0, 0, 0, 0
        for batch in itertools.islice(iter(loaders["train"]), n):
            l1, l2 = self.ae.encode(batch["x"].to(self.device))
            l1, l2 = l1.double(), l2.flatten(1).double()
            s1, q1, cnt1 = s1 + l1.sum((0, 2, 3)), q1 + (l1 ** 2).sum((0, 2, 3)), cnt1 + l1.shape[0] * l1.shape[2] * l1.shape[3]
            s2, q2, cnt2 = s2 + l2.sum(0), q2 + (l2 ** 2).sum(0), cnt2 + l2.shape[0]
        m1, m2 = s1 / cnt1, s2 / cnt2
        std = lambda m, q, c: (q / c - m ** 2).clamp_min(0).sqrt()
        self.fm.set_stats(m1.float(), std(m1, q1, cnt1).float(), m2.float(), std(m2, q2, cnt2).float())
        self.ema.module.load_state_dict(self.fm.state_dict())

    def loss(self, batch):
        with torch.no_grad():
            l1, l2 = self.ae.encode(batch["x"].to(self.device))
        return self.fm.loss(l1, l2)

    def after_step(self):
        self.ema.update(self.fm)

    def state(self):
        return {"fm": self.fm.state_dict(), "ema": self.ema.module.state_dict(),
                "fm_cfg": self.fm_cfg, "model_cfg": self.model_cfg}

    def load_state(self, s):
        self.fm.load_state_dict(s["fm"])
        self.ema.module.load_state_dict(s["ema"])


@STAGES.register("e1d2")
class E1D2Stage(Stage):
    def __init__(self, cfg, device, ae=None, model_cfg=None):
        super().__init__(cfg, device)
        if ae is None:
            ae, model_cfg = load_ae(cfg["ckpt"]["ae"], device)
        self.model_cfg = model_cfg
        self.e1 = freeze(ae.e1.to(device))
        self.d2 = build_d2(make_ops(model_cfg), model_cfg["e1_widths"]).to(device)
        if cfg.get("stage_opts", {}).get("init_d2_from_ae", False):
            self.d2.load_state_dict(ae.d2.state_dict())
        self.loss_fn = LOSSES[cfg["train"].get("loss", "mse")]

    def modules(self):
        return {"d2": self.d2}

    def loss(self, batch):
        x = batch["x"].to(self.device)
        with torch.no_grad():
            l1 = self.e1(x)
        return self.loss_fn(self.d2(l1).float(), x.float())

    def state(self):
        return {"d2": self.d2.state_dict(), "model_cfg": self.model_cfg}

    def load_state(self, s):
        self.d2.load_state_dict(s["d2"])


@STAGES.register("finetune")
class FinetuneStage(Stage):
    def __init__(self, cfg, device, ae=None, model_cfg=None, fm=None, d2=None):
        super().__init__(cfg, device)
        if ae is None:
            ae, model_cfg = load_ae(cfg["ckpt"]["ae"], device)
        self.model_cfg = model_cfg
        self.e1, self.e2 = freeze(ae.e1.to(device)), freeze(ae.e2.to(device))
        self.fm = freeze((fm if fm is not None else load_fm(cfg["ckpt"]["fm"], device,
                          cfg.get("sampling", {}).get("use_ema", True))).to(device))
        if d2 is None:
            d2 = build_d2(make_ops(model_cfg), model_cfg["e1_widths"])
            if cfg.get("ckpt", {}).get("d2"):
                d2.load_state_dict(torch.load(cfg["ckpt"]["d2"], map_location=device)["d2"])
        self.d2 = d2.to(device)
        s = cfg.get("sampling", {})
        self.steps, self.solver = s.get("steps", 20), s.get("solver", "euler")
        self.loss_fn = LOSSES[cfg["train"].get("loss", "mse")]

    def modules(self):
        return {"d2": self.d2}

    def loss(self, batch):
        x = batch["x"].to(self.device)
        with torch.no_grad():
            l2 = self.e2(self.e1(x))
            l1_hat = self.fm.sample(l2, self.steps, self.solver)   # generated latent1 (noise ~ global RNG)
        return self.loss_fn(self.d2(l1_hat).float(), x.float())

    def state(self):
        return {"d2": self.d2.state_dict(), "model_cfg": self.model_cfg}

    def load_state(self, s):
        self.d2.load_state_dict(s["d2"])
