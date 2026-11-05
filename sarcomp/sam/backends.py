"""Segmentation-model backends for the downstream (SAM) evaluation. Swap the model from the config:

    sam:
      backend: hf_sam                    # which adapter
      model: facebook/sam-vit-base       # which weights (HF hub id or local folder)

Backend interface (3 methods; images are uint8 RGB numpy arrays (H, W, 3)):
    set_image(rgb)        compute + cache the image embedding
    embedding()           -> Tensor (C, h, w) cached embedding (used for the embedding-similarity metric)
    predict(points)       -> bool Tensor (N, H, W): one mask per prompt point; `points` is (N, 2) float (x, y) in pixels

To add another model (SAM 2, SAM-HQ, MedSAM, a local fine-tune, ...): subclass SegBackend and
@SAM_BACKENDS.register("name"). Nothing else changes.
"""
import numpy as np
import torch
import torch.nn.functional as F

from ..utils import Registry

SAM_BACKENDS = Registry("sam_backend")


class SegBackend:
    def set_image(self, rgb):
        raise NotImplementedError

    def embedding(self):
        raise NotImplementedError

    def predict(self, points):
        raise NotImplementedError


def build_backend(cfg, device):
    """cfg = config["sam"]."""
    return SAM_BACKENDS.build(cfg.get("backend", "hf_sam"), device=device, model=cfg.get("model"))


@SAM_BACKENDS.register("hf_sam")
class HFSam(SegBackend):
    """Original SAM (ViT-B/L/H) through Hugging Face `transformers`.

    model: "facebook/sam-vit-base" | "facebook/sam-vit-large" | "facebook/sam-vit-huge" | local path.
    Pass `model_obj` / `processor` to inject already-built objects (used by the tests).
    """

    def __init__(self, device="cpu", model="facebook/sam-vit-base", model_obj=None, processor=None, **_):
        from transformers import SamModel, SamProcessor   # optional dependency, imported lazily
        self.device = device
        self.model = (model_obj if model_obj is not None else SamModel.from_pretrained(model)).to(device).eval()
        self.processor = processor if processor is not None else SamProcessor.from_pretrained(model)
        self._emb = None

    @torch.no_grad()
    def set_image(self, rgb):
        inp = self.processor(images=rgb, return_tensors="pt")
        self._rgb = rgb
        self._emb = self.model.get_image_embeddings(inp["pixel_values"].to(self.device))

    def embedding(self):
        return self._emb[0].float().cpu()

    @torch.no_grad()
    def predict(self, points, chunk=64):
        out = []
        for i in range(0, len(points), chunk):
            pts = [[[[float(x), float(y)]] for x, y in points[i:i + chunk]]]    # 1 image, N prompts, 1 point each
            inp = self.processor(images=self._rgb, input_points=pts, return_tensors="pt")
            o = self.model(image_embeddings=self._emb, input_points=inp["input_points"].to(self.device),
                           multimask_output=False)
            m = self.processor.image_processor.post_process_masks(
                o.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
            out.append(m.reshape(m.shape[0], m.shape[-2], m.shape[-1]).bool())
        return torch.cat(out)


@SAM_BACKENDS.register("toy")
class ToySegmenter(SegBackend):
    """NOT a real segmenter. Deterministic stand-in so tests / smoke checks run without model weights:
    embedding = 16x16 average-pooled (mean, local contrast); mask(point) = pixels within a tolerance of the
    (smoothed) grey value at the point."""

    def __init__(self, device="cpu", tol=0.12, **_):
        self.tol = tol

    def set_image(self, rgb):
        g = torch.from_numpy(np.ascontiguousarray(rgb[..., 0])).float()[None, None] / 255.0
        self._g = F.avg_pool2d(g, 5, 1, 2)[0, 0]
        mean = F.avg_pool2d(g, 16)
        contrast = F.avg_pool2d((g - F.interpolate(mean, size=g.shape[-2:], mode="nearest")).abs(), 16)
        self._e = torch.cat([mean, contrast], 1)[0]

    def embedding(self):
        return self._e

    def predict(self, points):
        H, W = self._g.shape
        masks = []
        for x, y in points:
            v = self._g[min(max(int(y), 0), H - 1), min(max(int(x), 0), W - 1)]
            masks.append((self._g - v).abs() < self.tol)
        return torch.stack(masks)
