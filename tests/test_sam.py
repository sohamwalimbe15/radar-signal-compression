import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sarcomp.sam.evaluate import auto_cols, mask_ious, mosaic_positions, build_mosaic, reconstruct_mosaic, IdentityPipeline  # noqa: E402
from sarcomp.smoke import Report, check_sam  # noqa: E402


def test_sam_plumbing_with_toy_backend():
    rep = Report(verbose=False)
    check_sam(rep, "cpu")
    assert rep.ok, rep.summary()


def test_mosaic_geometry_and_stitching():
    assert auto_cols(4, {"range": 0.323197, "azimuth": 0.068595}) == 19       # ~square on the ground
    assert mosaic_positions(35, 56, 4, 19, None) == [(r * 4, c * 19) for r in range(8) for c in range(2)]
    tile = lambda iy, ix: torch.full((2, 8, 8), float(iy * 10 + ix))
    x = build_mosaic(tile, 1, 2, 2, 3)                                         # 2x3 tiles of 8x8
    assert x.shape == (2, 16, 24) and x[0, 0, 0] == 12 and x[0, 15, 23] == 24
    assert torch.equal(reconstruct_mosaic(IdentityPipeline(), x, 8, "cpu", batch_tiles=4), x)   # split + paste is lossless


def test_mask_iou():
    a = torch.zeros(2, 4, 4, dtype=torch.bool); a[0, :2] = True
    b = torch.zeros(2, 4, 4, dtype=torch.bool); b[0, :1] = True
    assert torch.allclose(mask_ious(a, b), torch.tensor([0.5, 1.0]))           # [0.5, both empty -> 1]


def test_hf_backend_plumbing_with_tiny_random_sam():
    pytest.importorskip("transformers")
    from transformers import SamConfig, SamImageProcessor, SamModel, SamProcessor, SamVisionConfig
    from sarcomp.sam.backends import HFSam
    cfg = SamConfig(vision_config=SamVisionConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, mlp_dim=64,
                                                 global_attn_indexes=[1], output_channels=256))
    b = HFSam(device="cpu", model_obj=SamModel(cfg), processor=SamProcessor(image_processor=SamImageProcessor()))
    b.set_image((np.random.rand(120, 100, 3) * 255).astype("uint8"))
    m = b.predict(np.array([[10, 10], [50, 60], [90, 100]], dtype=np.float32))
    assert b.embedding().shape == (256, 64, 64) and m.shape == (3, 120, 100) and m.dtype == torch.bool
