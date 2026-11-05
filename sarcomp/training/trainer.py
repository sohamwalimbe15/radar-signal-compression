"""Generic step-based trainer. All the experiment-specific logic lives in the Stage."""
import json
import math
import os
import time

import torch
import yaml


def _infinite(loader):
    while True:
        for b in loader:
            yield b


class Trainer:
    def __init__(self, stage, loaders, cfg, run_dir, device):
        self.stage, self.loaders, self.cfg, self.run_dir, self.device = stage, loaders, cfg, run_dir, device
        self.t = cfg["train"]
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f)
        self.log_path = os.path.join(run_dir, "log.jsonl")

    def _log(self, rec):
        print(" ".join("%s=%s" % (k, ("%.5g" % v) if isinstance(v, float) else v) for k, v in rec.items()), flush=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    @torch.no_grad()
    def validate(self):
        self.stage.set_train(False)
        devs = [self.device] if str(self.device).startswith("cuda") else []
        losses = []
        with torch.random.fork_rng(devices=devs):      # deterministic val, doesn't disturb training RNG
            torch.manual_seed(1234)
            for i, b in enumerate(self.loaders["val"]):
                if i >= self.t.get("val_batches", 20):
                    break
                losses.append(float(self.stage.loss(b)))
        self.stage.set_train(True)
        return sum(losses) / max(len(losses), 1)

    def save(self, name="last.pt", extra=None):
        d = self.stage.state()
        d.update(extra or {})
        torch.save(d, os.path.join(self.run_dir, name))

    def fit(self):
        t, stage = self.t, self.stage
        stage.setup(self.loaders)
        params = stage.trainable_parameters()
        opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t.get("weight_decay", 0.0),
                                betas=tuple(t.get("betas", (0.9, 0.999))))
        warm, total = t.get("warmup_steps", 0), t["max_steps"]
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * (
            0.5 * (1 + math.cos(math.pi * min(s, total) / total)) if t.get("cosine", True) else 1.0))
        use_amp = bool(t.get("amp", False)) and str(self.device).startswith("cuda")
        data, t0, best = _infinite(self.loaders["train"]), time.time(), float("inf")
        stage.set_train(True)
        for step in range(1, total + 1):
            batch = next(data)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = stage.loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss at step %d: %s" % (step, loss.item()))
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, t.get("grad_clip", 1.0))
            opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
            stage.after_step()
            if step % t.get("log_every", 50) == 0 or step == 1:
                self._log({"step": step, "loss": float(loss), "grad_norm": float(gn),
                           "lr": sched.get_last_lr()[0], "sec": time.time() - t0})
            if step % t.get("val_every", 500) == 0 or step == total:
                v = self.validate()
                self._log({"step": step, "val_loss": v})
                if v < best:
                    best = v
                    self.save("best.pt")
            if step % t.get("ckpt_every", 1000) == 0 or step == total:
                self.save("last.pt")
        return best
