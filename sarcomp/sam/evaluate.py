"""Downstream evaluation: does a segmentation model (SAM) see the same thing in the reconstruction as in the original?

AMPLITUDE ONLY (SAM cannot see phase; phase quality is covered by the phase metrics in sarcomp.metrics).
No ground-truth masks are used: everything is *agreement between original and reconstruction*.

Pipeline per mosaic
  1. cut a region of `rows x cols` tiles (each tile = the model's 256x256 input) from the split. rows x cols is
     chosen to be roughly square on the GROUND (default 4 x 19 tiles ~ 330 m x 330 m), because pixels are ~4.7x
     finer in azimuth than in range.
  2. reconstruct every tile independently with the pipeline and paste them back ("stitching"). Tile seams are NOT
     blended: that is what a tiled codec would produce, so seam artefacts count against the model.
  3. log-amplitude -> resample azimuth to square pixels -> grey-level window taken from the ORIGINAL (same window for
     both images) -> uint8 -> 3 channels.
  4. run SAM on both and compute
        sam_embed_cos      mean cosine similarity of the SAM image embeddings (deterministic, no prompts)
        sam_mask_iou       mean IoU between the masks SAM gives for the SAME point prompts (a regular grid)
        sam_mask_iou_ge50  fraction of prompts whose masks have IoU >= 0.5
"""
import numpy as np
import torch
import torch.nn.functional as F

from ..fm import make_generator


# ------------------------------------------------------------------ geometry / tiling
def auto_cols(rows, pixel_spacing):
    """Tiles across (azimuth) that cover the same ground distance as `rows` tiles down (range)."""
    return max(1, int(round(rows * pixel_spacing["range"] / pixel_spacing["azimuth"])))


def mosaic_positions(ny, nx, rows, cols, max_n):
    """Top-left (iy, ix) tile indices of non-overlapping mosaics, spread evenly over the split."""
    pos = [(r * rows, c * cols) for r in range(ny // rows) for c in range(nx // cols)]
    if not pos:
        raise ValueError("split has %d x %d tiles, too small for a %d x %d mosaic" % (ny, nx, rows, cols))
    if max_n and len(pos) > max_n:
        pos = [pos[i] for i in np.linspace(0, len(pos) - 1, max_n).round().astype(int)]
    return pos


def build_mosaic(tile_fn, iy0, ix0, rows, cols):
    """tile_fn(iy, ix) -> (2,P,P) model-space tile. Returns (2, rows*P, cols*P)."""
    return torch.cat([torch.cat([tile_fn(iy0 + r, ix0 + c) for c in range(cols)], dim=-1) for r in range(rows)], dim=-2)


@torch.no_grad()
def reconstruct_mosaic(pipeline, x, P, device, batch_tiles=16, generator=None):
    """Reconstruct every PxP tile of the mosaic x (2,H,W) independently, paste the results back."""
    _, H, W = x.shape
    rows, cols = H // P, W // P
    tiles = x.unfold(1, P, P).unfold(2, P, P).permute(1, 2, 0, 3, 4).reshape(rows * cols, 2, P, P)
    out = torch.cat([pipeline(tiles[i:i + batch_tiles].to(device), generator=generator).float().cpu()
                     for i in range(0, len(tiles), batch_tiles)])
    return out.reshape(rows, cols, 2, P, P).permute(2, 0, 3, 1, 4).reshape(2, H, W)


# ------------------------------------------------------------------ image preparation (amplitude only)
def isotropic(logamp, pixel_spacing):
    """(H,W) log-amplitude, rows = range, cols = azimuth -> square ground pixels (area-average the azimuth axis)."""
    H, W = logamp.shape
    new_w = max(1, int(round(W * pixel_spacing["azimuth"] / pixel_spacing["range"])))
    return F.interpolate(logamp[None, None], size=(H, new_w), mode="area")[0, 0]


def to_rgb(logamp, lo, hi):
    g = ((logamp - lo) / max(hi - lo, 1e-6)).clamp(0, 1).mul(255).round().byte().numpy()
    return np.repeat(g[..., None], 3, axis=-1)


def prompt_points(H, W, grid):
    ys, xs = (np.arange(grid) + 0.5) * H / grid, (np.arange(grid) + 0.5) * W / grid
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float32)


# ------------------------------------------------------------------ metrics
def embedding_cosine(e1, e2):
    return F.cosine_similarity(e1.flatten(1), e2.flatten(1), dim=0).mean().item()


def mask_ious(m1, m2):
    """(N,H,W) bool x2 -> (N,) IoU; two empty masks count as IoU 1."""
    inter = (m1 & m2).flatten(1).sum(1).float()
    union = (m1 | m2).flatten(1).sum(1).float()
    return torch.where(union > 0, inter / union.clamp_min(1), torch.ones_like(union))


# ------------------------------------------------------------------ main entry
@torch.no_grad()
def evaluate_sam(pipeline, tile_fn, grid_shape, tf, backend, cfg, pixel_spacing, device, seed=0):
    """cfg = config["sam"]. grid_shape = (ny, nx) tiles in the split; tf = LogComplex."""
    pipeline.eval()
    ny, nx = grid_shape
    rows = cfg.get("mosaic_rows", 4)
    cols = cfg.get("mosaic_cols", "auto")
    cols = auto_cols(rows, pixel_spacing) if cols == "auto" else int(cols)
    P = tile_fn(0, 0).shape[-1]
    g = make_generator(seed, device)
    pct = cfg.get("clip_percentiles", [2, 98])
    cos, ious = [], []
    for iy0, ix0 in mosaic_positions(ny, nx, rows, cols, cfg.get("max_mosaics", 8)):
        x = build_mosaic(tile_fn, iy0, ix0, rows, cols)
        xr = reconstruct_mosaic(pipeline, x, P, device, cfg.get("batch_tiles", 16), g)
        la, lr = isotropic(tf.log_amp(x), pixel_spacing), isotropic(tf.log_amp(xr), pixel_spacing)
        lo, hi = [float(v) for v in np.percentile(la.numpy(), pct)]       # window from the ORIGINAL, used for both
        rgb_o, rgb_r = to_rgb(la, lo, hi), to_rgb(lr, lo, hi)
        pts = prompt_points(*rgb_o.shape[:2], cfg.get("prompt_grid", 8))
        backend.set_image(rgb_o)
        e_o, m_o = backend.embedding(), backend.predict(pts)
        backend.set_image(rgb_r)
        e_r, m_r = backend.embedding(), backend.predict(pts)
        cos.append(embedding_cosine(e_o, e_r))
        ious.append(mask_ious(m_o.cpu(), m_r.cpu()))
    iou = torch.cat(ious)
    return {"sam_embed_cos": float(np.mean(cos)), "sam_mask_iou": iou.mean().item(),
            "sam_mask_iou_ge50": (iou >= 0.5).float().mean().item(),
            "sam_n_mosaics": len(cos), "sam_n_prompts": int(iou.numel()),
            "sam_mosaic_tiles": "%dx%d" % (rows, cols)}


class IdentityPipeline(torch.nn.Module):
    """Returns its input: the 'perfect reconstruction' reference (agreement metrics must be 1.0)."""

    def forward(self, x, generator=None):
        return x
