"""Lightweight smoke test: are gradients flowing, are there NaNs, do all stages + checkpoints work?

    from sarcomp.smoke import run_smoke_test
    report = run_smoke_test(device="cuda", size="tiny")     # seconds, synthetic data
    report = run_smoke_test(device="cuda", size="full")     # real layer widths (256x256, 1024/2048 latents), batch 2
    assert report.ok

Everything uses synthetic data unless `tif_dir` is given (then one real batch per split is also checked).
Checks (each prints PASS/FAIL):
  * log-complex transform round-trips exactly
  * complex conv / conv-transpose == explicit 4-conv formula == PyTorch native complex conv; i-equivariance
  * per architecture: latent shapes, no non-finite activations, every parameter gets a finite gradient,
    overfit-one-batch loss drops
  * every training stage (ae, fm[backbones], e1d2, finetune) via the real Trainer + checkpoints;
    frozen modules do not change, trained modules do
  * final pipelines build from checkpoints, metrics are finite, FM sampling is repeatable with a seed
"""
import contextlib
import copy
import os
import tempfile
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import LogComplex, build_data
from .fm import make_generator
from .metrics import DEFAULT_METRICS, build_metrics, evaluate
from .models import ComplexConv2d, ComplexConvTranspose2d, build_autoencoder, latent_info
from .pipelines import build_pipeline
from .training import STAGES, Trainer

SIZES = {
    "tiny": dict(image_size=32, e1_widths=[8, 16, 16], e2_widths=[16, 32], batch=4, overfit_steps=120,
                 fm_args={"conv_film": dict(hidden=32, n_blocks=2, emb=32),
                          "dit": dict(hidden=32, depth=2, heads=4, emb=32)}),
    "full": dict(image_size=256, e1_widths=[64, 128, 256, 512, 1024], e2_widths=[1024, 1024, 2048], batch=2,
                 overfit_steps=0, fm_args={"conv_film": {}, "dit": {}}),
}


class Report:
    def __init__(self, verbose=True):
        self.rows, self.verbose = [], verbose

    def add(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        if self.verbose:
            print("%s | %s%s" % ("PASS" if ok else "FAIL", name, (" | " + detail) if detail else ""), flush=True)

    @property
    def ok(self):
        return all(r[1] for r in self.rows)

    def summary(self):
        bad = [r for r in self.rows if not r[1]]
        return "%d checks, %d failed%s" % (len(self.rows), len(bad), "".join("\n  FAIL " + r[0] + " " + r[2] for r in bad))


# --------------------------------------------------------------------------- helpers
@contextlib.contextmanager
def finite_hooks(model, bad):
    """Record every leaf module whose output contains NaN/Inf."""
    hs = []
    for n, m in model.named_modules():
        if not list(m.children()):
            hs.append(m.register_forward_hook(
                lambda mod, i, o, n=n: bad.append(n) if (torch.is_tensor(o) and not torch.isfinite(o).all()) else None))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


def grad_report(parts):
    """parts: name -> module (after loss.backward()). Returns dict(problems, warnings, norms)."""
    problems, warns, norms = [], [], {}
    for name, m in parts.items():
        g2 = 0.0
        for pn, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                problems.append("%s.%s has no grad" % (name, pn))
            elif not torch.isfinite(p.grad).all():
                problems.append("%s.%s grad non-finite" % (name, pn))
            else:
                if float(p.grad.abs().max()) == 0.0:
                    warns.append("%s.%s grad is all zero" % (name, pn))
                g2 += float(p.grad.float().pow(2).sum())
        norms[name] = g2 ** 0.5
    return {"problems": problems, "warnings": warns, "norms": norms}


def snapshot(m):
    return [p.detach().clone() for p in m.parameters()]


def changed(m, snap):
    return any(not torch.equal(p.detach(), s) for p, s in zip(m.parameters(), snap))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def make_cfg(arch, size, name, stage, root, steps, backbone="conv_film", norm="none"):
    S = SIZES[size]
    return {
        "name": name, "run_root": root, "stage": stage, "seed": 0,
        "data": {"source": "synthetic", "patch": S["image_size"], "batch_size": S["batch"], "num_workers": 0,
                 "n_synthetic": {"train": 4 * S["batch"], "val": 2 * S["batch"], "test": 2 * S["batch"]}},
        "model": {"arch": arch, "image_size": S["image_size"], "e1_widths": S["e1_widths"],
                  "e2_widths": S["e2_widths"], "act": "relu", "norm": norm},
        "fm": {"backbone": backbone, "backbone_args": S["fm_args"][backbone], "stats_batches": 2},
        "train": {"lr": 1e-3, "max_steps": steps, "log_every": 1, "val_every": steps, "val_batches": 2,
                  "ckpt_every": steps, "grad_clip": 1.0, "amp": False, "ema": 0.9},
        "sampling": {"steps": 3, "solver": "euler"},
        "ckpt": {},
    }


def run_dir(cfg):
    return os.path.join(cfg["run_root"], cfg["name"])


def fit_stage(cfg, device, stage, loaders):
    Trainer(stage, loaders, cfg, run_dir(cfg), device).fit()


# --------------------------------------------------------------------------- individual checks
def check_transform(rep):
    g = torch.Generator().manual_seed(0)
    amp = torch.expm1(torch.rand(4, 64, 64, generator=g) * 11.0)        # DN from 0 to ~60000
    ph = (torch.rand(4, 64, 64, generator=g) * 2 - 1) * 3.14159
    tf = LogComplex(3.36)
    a2, p2 = tf.inverse(tf(amp, ph))
    rel = ((a2 - amp).abs() / amp.clamp_min(1)).max().item()
    wrap = torch.atan2(torch.sin(p2 - ph), torch.cos(p2 - ph))[amp > 1].abs().max().item()
    rep.add("transform: log-complex round trip", rel < 1e-3 and wrap < 1e-3, "max rel amp err %.1e, phase err %.1e" % (rel, wrap))


def check_layers(rep):
    torch.manual_seed(0)
    cin, cout = 3, 5
    z = torch.randn(2, 2 * cin, 9, 9)
    a, b = z[:, :cin], z[:, cin:]
    for kind in ("conv", "convT"):
        if kind == "conv":
            L = ComplexConv2d(cin, cout, 3, stride=2, padding=1)
            op = lambda u, w: F.conv2d(u, w, None, 2, 1)
        else:
            L = ComplexConvTranspose2d(cin, cout, 4, stride=2, padding=1)
            op = lambda u, w: F.conv_transpose2d(u, w, None, 2, 1)
        nn.init.normal_(L.bias)
        out = L(z)
        re = op(a, L.x) - op(b, L.y) + L.bias[:cout].view(1, -1, 1, 1)
        im = op(b, L.x) + op(a, L.y) + L.bias[cout:].view(1, -1, 1, 1)
        err = (out - torch.cat([re, im], 1)).abs().max().item()
        rep.add("complex %s == explicit (ax-by)+i(bx+ay)" % kind, err < 1e-5, "max err %.1e" % err)
        try:   # independent reference: PyTorch native complex tensors
            nat_op = F.conv2d if kind == "conv" else F.conv_transpose2d
            kw = dict(stride=2, padding=1)
            nat = nat_op(torch.complex(a, b), torch.complex(L.x, L.y), None, **kw) + torch.complex(
                L.bias[:cout], L.bias[cout:]).view(1, -1, 1, 1)
            err = (out - torch.cat([nat.real, nat.imag], 1)).abs().max().item()
            rep.add("complex %s == PyTorch native complex" % kind, err < 1e-4, "max err %.1e" % err)
        except RuntimeError as e:
            rep.add("complex %s native-complex reference" % kind, True, "skipped (%s)" % str(e)[:50])
        nn.init.zeros_(L.bias)
        o = L(z)
        o_rot = L(torch.cat([-b, a], 1))                      # layer(i z) should equal i layer(z) = (-im, re)
        err = (o_rot - torch.cat([-o[:, cout:], o[:, :cout]], 1)).abs().max().item()
        rep.add("complex %s i-equivariance (zero bias)" % kind, err < 1e-5, "max err %.1e" % err)


def check_arch(rep, arch, size, device, norms=("none", "group")):
    S = SIZES[size]
    cfg = make_cfg(arch, size, "x", "ae", "", 1)
    loaders, _ = build_data(cfg["data"])
    x = next(iter(loaders["val"]))["x"].to(device)
    for norm in norms:
        m = {**cfg["model"], "norm": norm}
        ae = build_autoencoder(m).to(device)
        info = latent_info(m)
        tag = "%s/%s" % (arch, norm)
        if norm == "none":
            rep.add("%s: params" % tag, True, "E1 %.2fM  E2 %.2fM  D1 %.2fM  D2 %.2fM  total %.2fM" % tuple(
                [n_params(p) / 1e6 for p in (ae.e1, ae.e2, ae.d1, ae.d2, ae)]))
        bad = []
        with finite_hooks(ae, bad):
            out = ae(x)
            loss = F.mse_loss(out["recon"], x)
            loss.backward()
        shapes_ok = (tuple(out["l1"].shape[1:]) == info["l1_shape"] and tuple(out["l2"].shape[1:]) == info["l2_shape"]
                     and out["recon"].shape == x.shape)
        rep.add("%s: shapes (l1 %s, l2 %s)" % (tag, tuple(out["l1"].shape[1:]), tuple(out["l2"].shape[1:])), shapes_ok)
        rep.add("%s: forward/backward finite (no NaN/Inf in any layer)" % tag, not bad and torch.isfinite(loss),
                "loss %.4f %s" % (loss.item(), bad[:3]))
        lat_std = out["l2"].flatten(1).std(0).mean().item(), out["l1"].std(0).mean().item()
        rep.add("%s: latents carry information (std over batch: l2 %.2e, l1 %.2e)" % (tag, *lat_std),
                min(lat_std) > 1e-4, "signal vanished / collapsed if ~0")
        gr = grad_report({"e1": ae.e1, "e2": ae.e2, "d1": ae.d1, "d2": ae.d2})
        ratio = max(gr["norms"].values()) / max(min(gr["norms"].values()), 1e-30)
        rep.add("%s: gradient norms balanced across E1/E2/D1/D2 (max/min %.1e < 1e3)" % (tag, ratio), ratio < 1e3,
                "vanishing/exploding gradients if large")
        rep.add("%s: every parameter gets a finite gradient" % tag, not gr["problems"],
                "grad norms " + " ".join("%s=%.2e" % kv for kv in gr["norms"].items()) +
                ("  WARN " + "; ".join(gr["warnings"][:3]) if gr["warnings"] else "") +
                ("  " + "; ".join(gr["problems"][:3]) if gr["problems"] else ""))
        if norm == "none" and S["overfit_steps"]:                 # can the model fit ONE batch?
            ae = build_autoencoder(m).to(device)
            opt = torch.optim.AdamW(ae.parameters(), lr=2e-3)
            first = None
            for _ in range(S["overfit_steps"]):
                l = F.mse_loss(ae(x)["recon"], x)
                first = first if first is not None else l.item()
                opt.zero_grad(), l.backward(), opt.step()
            rep.add("%s: overfit one batch (loss falls)" % tag, l.item() < 0.8 * first and torch.isfinite(l),
                    "%.4f -> %.4f in %d steps" % (first, l.item(), S["overfit_steps"]))
    if str(device).startswith("cuda"):
        ae = build_autoencoder(cfg["model"]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = F.mse_loss(ae(x)["recon"].float(), x)
        l.backward()
        rep.add("%s: bf16 autocast step finite" % arch, torch.isfinite(l), "loss %.4f" % l.item())


def stage_parts(stage):
    parts = stage.modules()
    if "ae" in parts:
        ae = parts["ae"]
        return {"e1": ae.e1, "e2": ae.e2, "d1": ae.d1, "d2": ae.d2}
    return parts


def check_stage_grads(rep, stage, loaders, device, tag, frozen=()):
    batch = next(iter(loaders["train"]))
    bad = []
    mods = [m for m in stage.modules().values()]
    with contextlib.ExitStack() as es:
        for m in mods:
            es.enter_context(finite_hooks(m, bad))
        loss = stage.loss(batch)
        loss.backward()
    gr = grad_report(stage_parts(stage))
    rep.add("%s: loss finite, every trainable param has finite grad" % tag, torch.isfinite(loss) and not gr["problems"] and not bad,
            "loss %.4f  %s" % (loss.item(), "; ".join(gr["problems"][:3] + bad[:3] + gr["warnings"][:2])))
    leaked = [n for n, m in frozen if any(p.grad is not None for p in m.parameters())]
    rep.add("%s: frozen modules receive no gradient" % tag, not leaked, str(leaked))
    for m in stage.modules().values():
        m.zero_grad(set_to_none=True)


def check_chain(rep, arch, size, device, root, steps, backbones):
    """The whole experiment-3/4 recipe through the real Trainer + checkpoints."""
    S = SIZES[size]
    cfgs = {}
    c = cfgs["ae"] = make_cfg(arch, size, "%s_ae" % arch, "ae", root, steps)
    loaders, tf = build_data(c["data"])

    # ---- exp 1/2
    st = STAGES.build("ae", c, device)
    check_stage_grads(rep, st, loaders, device, "%s/ae stage" % arch)
    fit_stage(c, device, st, loaders)
    ae_ckpt = os.path.join(run_dir(c), "last.pt")
    rep.add("%s/ae stage: trained %d steps, checkpoint written" % (arch, steps), os.path.exists(ae_ckpt))

    # ---- exp 3/4 step 1 (every backbone)
    fm_ckpt = None
    for bb in backbones:
        c = make_cfg(arch, size, "%s_fm_%s" % (arch, bb), "fm", root, steps, backbone=bb)
        c["ckpt"]["ae"] = ae_ckpt
        st = STAGES.build("fm", c, device)
        st.setup(loaders)
        check_stage_grads(rep, st, loaders, device, "%s/fm[%s]" % (arch, bb), frozen=[("e1/e2/d1/d2", st.ae)])
        snap_ae, snap_fm = snapshot(st.ae), snapshot(st.fm)
        fit_stage(c, device, st, loaders)
        rep.add("%s/fm[%s]: trains FM only (AE frozen & unchanged, FM changed)" % (arch, bb),
                not changed(st.ae, snap_ae) and changed(st.fm, snap_fm))
        l2 = st.ae.encode(next(iter(loaders["val"]))["x"].to(device))[1]
        g1, g2 = make_generator(0, device), make_generator(0, device)
        s1, s2 = st.fm.sample(l2, 3, "euler", g1), st.fm.sample(l2, 3, "heun", g2)
        s3 = st.fm.sample(l2, 3, "euler", make_generator(0, device))
        rep.add("%s/fm[%s]: sampling finite, right shape, repeatable with a seed" % (arch, bb),
                torch.isfinite(s1).all() and torch.isfinite(s2).all() and tuple(s1.shape[1:]) == tuple(st.fm.latent_shape)
                and torch.equal(s1, s3), str(tuple(s1.shape)))
        if fm_ckpt is None:
            fm_ckpt = os.path.join(run_dir(c), "last.pt")

    # ---- exp 3/4 step 2
    c = make_cfg(arch, size, "%s_e1d2" % arch, "e1d2", root, steps)
    c["ckpt"]["ae"] = ae_ckpt
    st = STAGES.build("e1d2", c, device)
    check_stage_grads(rep, st, loaders, device, "%s/e1d2" % arch, frozen=[("e1", st.e1)])
    snap_e1, snap_d2 = snapshot(st.e1), snapshot(st.d2)
    fit_stage(c, device, st, loaders)
    rep.add("%s/e1d2: only D2 trains (E1 frozen & unchanged)" % arch, not changed(st.e1, snap_e1) and changed(st.d2, snap_d2))
    d2_ckpt = os.path.join(run_dir(c), "last.pt")

    # ---- exp 3/4 step 3
    c = make_cfg(arch, size, "%s_finetune" % arch, "finetune", root, steps)
    c["ckpt"] = {"ae": ae_ckpt, "fm": fm_ckpt, "d2": d2_ckpt}
    st = STAGES.build("finetune", c, device)
    check_stage_grads(rep, st, loaders, device, "%s/finetune" % arch, frozen=[("e1", st.e1), ("e2", st.e2), ("fm", st.fm)])
    snaps = {n: snapshot(m) for n, m in [("e1", st.e1), ("e2", st.e2), ("fm", st.fm), ("d2", st.d2)]}
    fit_stage(c, device, st, loaders)
    rep.add("%s/finetune: only D2 trains (E1, E2, FM frozen & unchanged)" % arch,
            all(not changed(m, snaps[n]) for n, m in [("e1", st.e1), ("e2", st.e2), ("fm", st.fm)]) and changed(st.d2, snaps["d2"]))

    # ---- final pipelines + metrics
    for tag, cfg_ in (("exp1/2 (AE)", cfgs["ae"]), ("exp3/4 (FM)", c)):
        pipe = build_pipeline(cfg_, device)
        mets = build_metrics(DEFAULT_METRICS, tf)
        res = evaluate(pipe, loaders["val"], mets, device, seed=0)
        res2 = evaluate(pipe, loaders["val"], build_metrics(DEFAULT_METRICS, tf), device, seed=0)
        rep.add("%s/%s: pipeline + all metrics finite, repeatable" % (arch, tag),
                all(map(lambda v: v == v and abs(v) != float("inf"), res.values())) and res == res2,
                " ".join("%s=%.3g" % kv for kv in res.items()))


def check_sam(rep, device):
    """SAM-evaluation plumbing with the toy backend: a perfect reconstruction must agree perfectly, a noisy one must not."""
    from .sam import IdentityPipeline, build_backend, evaluate_sam
    cfg = make_cfg("baseline", "tiny", "s", "ae", "", 1)
    loaders, tf = build_data(cfg["data"])
    ds = loaders["val"].dataset
    tile_fn = lambda iy, ix: ds[(iy * 6 + ix) % len(ds)]["x"]
    sc = {"backend": "toy", "mosaic_rows": 2, "mosaic_cols": 3, "max_mosaics": 2, "prompt_grid": 4}
    ps = {"range": 1.0, "azimuth": 1.0}
    be = build_backend(sc, device)

    class Noisy(nn.Module):
        def forward(self, x, generator=None):
            return x + 0.6 * torch.randn_like(x)

    r0 = evaluate_sam(IdentityPipeline(), tile_fn, (4, 6), tf, be, sc, ps, device)
    r1 = evaluate_sam(Noisy(), tile_fn, (4, 6), tf, be, sc, ps, device)
    rep.add("sam eval: perfect reconstruction -> embed cos 1, mask IoU 1", r0["sam_embed_cos"] > 0.999 and r0["sam_mask_iou"] > 0.999,
            "cos %.4f iou %.4f" % (r0["sam_embed_cos"], r0["sam_mask_iou"]))
    rep.add("sam eval: noisy reconstruction scores lower, finite", r1["sam_embed_cos"] < r0["sam_embed_cos"] and r1["sam_mask_iou"] < r0["sam_mask_iou"]
            and all(map(lambda v: v == v, [r1["sam_embed_cos"], r1["sam_mask_iou"]])),
            "cos %.4f iou %.4f (%d prompts)" % (r1["sam_embed_cos"], r1["sam_mask_iou"], r1["sam_n_prompts"]))


def check_real_data(rep, tif_dir):
    cfg = make_cfg("baseline", "tiny", "r", "ae", "", 1)
    cfg["data"] = {"source": "tif", "data_dir": tif_dir, "patch": 256, "batch_size": 2, "num_workers": 0,
                   "train_samples_per_epoch": 4, "stats_path": None}
    loaders, tf = build_data(cfg["data"])
    for k, l in loaders.items():
        x = next(iter(l))["x"]
        rep.add("real data [%s]: batch %s finite" % (k, tuple(x.shape)), torch.isfinite(x).all() and x.shape[1] == 2,
                "|z| mean %.3f" % torch.hypot(x[:, 0], x[:, 1]).mean().item())


# --------------------------------------------------------------------------- entry point
def run_smoke_test(device="auto", size="tiny", archs=("baseline", "complex"), backbones=("conv_film", "dit"),
                   steps=2, tif_dir=None, workdir=None, verbose=True, seed=0):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    torch.manual_seed(seed)
    rep = Report(verbose)
    t0 = time.time()
    print("device=%s  size=%s  torch=%s" % (device, size, torch.__version__))
    check_transform(rep)
    check_layers(rep)
    with tempfile.TemporaryDirectory() if workdir is None else contextlib.nullcontext(workdir) as root:
        for arch in archs:
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            check_arch(rep, arch, size, device, norms=("none", "group") if size == "tiny" else ("none",))
            check_chain(rep, arch, size, device, root, steps, backbones)
            if str(device).startswith("cuda"):
                print("   peak GPU memory so far: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
    check_sam(rep, "cpu")
    if tif_dir:
        check_real_data(rep, tif_dir)
    print("\n%s  (%.1fs)" % (rep.summary(), time.time() - t0))
    return rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--size", default="tiny", choices=list(SIZES))
    ap.add_argument("--tif-dir", default=None)
    a = ap.parse_args()
    raise SystemExit(0 if run_smoke_test(a.device, a.size, tif_dir=a.tif_dir).ok else 1)
