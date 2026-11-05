"""Collect runs/*/metrics_<split>.json into one markdown table.   python scripts/compare.py [--split test]"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--split", default="test")
a = ap.parse_args()
rows = {os.path.basename(os.path.dirname(p)): json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.runs, "*", "metrics_%s.json" % a.split)))}
if not rows:
    raise SystemExit("no metrics_%s.json found under %s/" % (a.split, a.runs))
cols = sorted({k for r in rows.values() for k in r})
print("| experiment | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1))
for n, r in rows.items():
    print("| %s | " % n + " | ".join((("%.4g" % r[c]) if isinstance(r[c], (int, float)) else str(r[c])) if c in r else "" for c in cols) + " |")
