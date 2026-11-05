"""Parameter counts per part for both architectures (use to pick widths for a fair comparison).
python scripts/count_params.py [--set model.e1_widths=[32,64,128,256,512]]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sarcomp.config import load_config  # noqa: E402
from sarcomp.fm import build_fm  # noqa: E402
from sarcomp.models import build_autoencoder, latent_info  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "base.yaml"))
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
cfg = load_config(a.config, a.set)
n = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
for arch in ("baseline", "complex"):
    m = {**cfg["model"], "arch": arch}
    ae = build_autoencoder(m)
    info = latent_info(m)
    print("%-9s E1 %7.2fM  E2 %7.2fM  D1 %7.2fM  D2 %7.2fM  | AE total %7.2fM  | latent1 %s latent2 %s" % (
        arch, n(ae.e1), n(ae.e2), n(ae.d1), n(ae.d2), n(ae), info["l1_shape"], info["l2_shape"]))
print("FM (%s): %.2fM params" % (cfg["fm"]["backbone"], n(build_fm(cfg["fm"], info))))
