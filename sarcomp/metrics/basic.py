"""Starter metrics (placeholders until the project's own evaluation proposal is added)."""
import math

import torch

from .base import METRICS, Metric

PEAK_LOG_AMP = math.log1p(65535.0)


@METRICS.register("complex_mse")
class ComplexMSE(Metric):
    """MSE over the (Re, Im) model-space channels (what the training loss measures)."""
    name = "complex_mse"

    def reset(self):
        self.s, self.n = 0.0, 0

    def update(self, p, t):
        self.s += float(((p - t) ** 2).sum())
        self.n += t.numel()

    def compute(self):
        return self.s / self.n


@METRICS.register("complex_nmse_db")
class ComplexNMSEdB(Metric):
    """10 log10( sum|z_pred - z_true|^2 / sum|z_true|^2 )   (lower / more negative is better)."""
    name = "complex_nmse_db"

    def reset(self):
        self.e, self.t = 0.0, 0.0

    def update(self, p, t):
        self.e += float(((p - t) ** 2).sum())
        self.t += float((t ** 2).sum())

    def compute(self):
        return 10 * math.log10(max(self.e, 1e-30) / max(self.t, 1e-30))


@METRICS.register("complex_corr")
class ComplexCorr(Metric):
    """|sum z_p conj(z_t)| / sqrt(sum|z_p|^2 sum|z_t|^2)   in [0,1]; 1 = perfect up to a complex scale."""
    name = "complex_corr"

    def reset(self):
        self.re = self.im = self.pp = self.tt = 0.0

    def update(self, p, t):
        pr, pi, tr, ti = p[:, 0].double(), p[:, 1].double(), t[:, 0].double(), t[:, 1].double()
        self.re += float((pr * tr + pi * ti).sum())
        self.im += float((pi * tr - pr * ti).sum())
        self.pp += float((pr ** 2 + pi ** 2).sum())
        self.tt += float((tr ** 2 + ti ** 2).sum())

    def compute(self):
        return math.hypot(self.re, self.im) / math.sqrt(max(self.pp * self.tt, 1e-30))


@METRICS.register("log_amp_mae")
class LogAmpMAE(Metric):
    """Mean |log1p(A_pred) - log1p(A_true)|  (natural-log amplitude units)."""
    name = "log_amp_mae"

    def reset(self):
        self.s, self.n = 0.0, 0

    def update(self, p, t):
        d = (self.tf.log_amp(p) - self.tf.log_amp(t)).abs()
        self.s += float(d.sum())
        self.n += d.numel()

    def compute(self):
        return self.s / self.n


@METRICS.register("log_amp_psnr_db")
class LogAmpPSNR(Metric):
    """PSNR of log1p-amplitude, peak = log1p(65535)."""
    name = "log_amp_psnr_db"

    def reset(self):
        self.s, self.n = 0.0, 0

    def update(self, p, t):
        d = (self.tf.log_amp(p) - self.tf.log_amp(t)) ** 2
        self.s += float(d.sum())
        self.n += d.numel()

    def compute(self):
        return 10 * math.log10(PEAK_LOG_AMP ** 2 / max(self.s / self.n, 1e-30))


def _wrapped_phase_err(p, t, tf):
    """|wrap(phi_pred - phi_true)| in radians, (B,H,W); plus target log-amp for masking/weighting."""
    d = torch.atan2(p[:, 1], p[:, 0]) - torch.atan2(t[:, 1], t[:, 0])
    return torch.atan2(torch.sin(d), torch.cos(d)).abs(), tf.log_amp(t)


@METRICS.register("phase_mae_deg")
class PhaseMAEDeg(Metric):
    """Mean wrapped phase error (degrees) over pixels with target amplitude > 0.
    NOTE: speckle phase is ~uniform, so unweighted error is dominated by unlearnable pixels;
    see phase_mae_deg_ampw."""
    name = "phase_mae_deg"

    def reset(self):
        self.s, self.n = 0.0, 0

    def update(self, p, t):
        e, m = _wrapped_phase_err(p, t, self.tf)
        valid = m > 0
        self.s += float(e[valid].sum())
        self.n += int(valid.sum())

    def compute(self):
        return math.degrees(self.s / max(self.n, 1))


@METRICS.register("phase_mae_deg_ampw")
class PhaseMAEDegAmpW(Metric):
    """Wrapped phase error (degrees) weighted by target log-amplitude (bright pixels count more)."""
    name = "phase_mae_deg_ampw"

    def reset(self):
        self.s, self.w = 0.0, 0.0

    def update(self, p, t):
        e, m = _wrapped_phase_err(p, t, self.tf)
        self.s += float((e * m).sum())
        self.w += float(m.sum())

    def compute(self):
        return math.degrees(self.s / max(self.w, 1e-30))


DEFAULT_METRICS = ["complex_mse", "complex_nmse_db", "complex_corr", "log_amp_mae", "log_amp_psnr_db",
                   "phase_mae_deg", "phase_mae_deg_ampw"]
