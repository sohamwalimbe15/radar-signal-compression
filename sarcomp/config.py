"""YAML configs with `base:` inheritance and dotted CLI overrides (`--set train.lr=1e-4`)."""
import copy
import os

import yaml


def deep_merge(base, new):
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load(path):
    path = os.path.abspath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(_load(os.path.join(os.path.dirname(path), base)), cfg)
    return cfg


def _parse(val):
    v = yaml.safe_load(val)
    if isinstance(v, str):          # YAML 1.1 reads "3e-4" as a string; make it a float
        try:
            return float(v)
        except ValueError:
            pass
    return v


def apply_overrides(cfg, overrides):
    for o in overrides or []:
        key, val = o.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse(val)
    return cfg


def load_config(path, overrides=()):
    return apply_overrides(_load(path), list(overrides))
