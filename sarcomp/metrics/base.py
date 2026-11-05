"""Metric framework.

A metric accumulates sums over batches (exact dataset-level numbers, not mean-of-batch-means):
    m.update(pred, target)   # both (B,2,H,W) model-space tensors ([Re, Im], log-compressed)
    m.compute() -> float
Metrics that need physical amplitude/phase use `self.tf` (the LogComplex transform) to invert.

Add a metric: subclass Metric, @METRICS.register("name"), implement reset/update/compute.
"""
import torch

from ..utils import Registry

METRICS = Registry("metric")


class Metric:
    name = "metric"

    def __init__(self, transform=None, **kw):
        self.tf, self.kw = transform, kw
        self.reset()

    def reset(self):
        raise NotImplementedError

    def update(self, pred, target):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


def build_metrics(names, transform):
    return [METRICS.build(n, transform=transform) for n in names]


@torch.no_grad()
def evaluate(pipeline, loader, metrics, device, max_batches=None, seed=0):
    """Run `pipeline` on `loader`; returns {metric_name: value}. Fixed noise seed => repeatable for FM pipelines."""
    from ..fm import make_generator
    pipeline.eval()
    g = make_generator(seed, device)
    for m in metrics:
        m.reset()
    for i, b in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = b["x"].to(device)
        pred = pipeline(x, generator=g)
        for m in metrics:
            m.update(pred.float(), x.float())
    return {m.name: m.compute() for m in metrics}
