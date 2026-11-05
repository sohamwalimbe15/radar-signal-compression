import copy

import torch


class EMA:
    """Exponential moving average of a module's parameters (buffers are copied)."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.module = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for pe, p in zip(self.module.parameters(), model.parameters()):
            pe.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for be, b in zip(self.module.buffers(), model.buffers()):
            be.copy_(b)
