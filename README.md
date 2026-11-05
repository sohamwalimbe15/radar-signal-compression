# sarcomp: complex-valued SAR autoencoders with flow-matching decoders

Modular code for learned compression of an ICEYE X-band SLC image (amplitude + phase). Every component
(architecture, flow-matching backbone, training stage, metric, dataset) is a plug-in selected from a YAML config.

* Data description (what the TIFF / GeoJSON contain, encodings, statistics): [`docs/data.md`](docs/data.md)
* Design notes: `baseline_model.md`, `complex_model.md`, `idea.md`

## The idea in one picture

```
image (2,256,256) --E1--> latent1 (1024,8,8) --E2--> latent2 (2048,1,1)   <- the compressed code
latent2 --D1--> latent1_hat --D2--> reconstruction               (exp 1, 2: end-to-end AE)
latent2 --[flow matching: noise -> latent1 | latent2]--> --D2--> reconstruction   (exp 3, 4)
```

| exp | architecture | decoder path | how it is trained |
|---|---|---|---|
| 1 | baseline (real conv; separate conv on Re and Im, concat) | D1, D2 | end-to-end reconstruction (`stage: ae`) |
| 2 | complex (complex conv, split ReLU) | D1, D2 | end-to-end reconstruction (`stage: ae`) |
| 3 | baseline | FM, D2 | (1) `fm`: FM on frozen E1,E2 (2) `e1d2`: D2 on frozen E1 (3) `finetune`: D2 on FM-generated latents |
| 4 | complex | FM, D2 | same three steps |

Input representation: `z = log1p(A)/scale * exp(j*phase)` as 2 real channels `[Re, Im]`; exactly invertible to amplitude and phase.
Complex features are kept as a *stacked real* tensor `[Re | Im]`, so complex and real networks have the same tensor interface and the
flow-matching model treats both latents identically (a real-valued network).

## Install

```bash
pip install -r requirements.txt          # torch, numpy, pyyaml, (rasterio for the real TIFF)
pip install -e .                          # optional: makes `import sarcomp` work from anywhere
```
Put the ICEYE `.tif` in `data/` (or set `data.tif_path`). Python >= 3.9.

## 1. Smoke test first (no data needed)

```bash
python -m sarcomp.smoke --size tiny      # seconds, CPU is fine
python -m sarcomp.smoke --size full      # real widths (256x256, 1024/2048 latents); use a GPU
pytest tests                             # same checks as a test suite
```
On Colab: open `notebooks/smoke_test.ipynb`. It checks (all PASS/FAIL): transform round-trip; complex layers vs PyTorch native
complex conv; latent shapes; NaN/Inf on every layer; finite gradients on **every** parameter and balanced gradient norms across
E1/E2/D1/D2; overfit-one-batch; all four stages through the real `Trainer` with checkpoints; frozen modules stay frozen; FM sampling
repeatable with a seed; final pipelines + metrics.

## 2. Run the experiments

Each command trains one stage; checkpoints go to `runs/<name>/{last,best}.pt`, logs to `runs/<name>/log.jsonl`.

```bash
# experiment 1 / 2: end-to-end autoencoders
python scripts/train.py --config configs/exp1_baseline_ae.yaml
python scripts/train.py --config configs/exp2_complex_ae.yaml

# experiment 3 (baseline) - needs exp1; the three steps in order
python scripts/train.py --config configs/exp3_baseline_1_fm.yaml
python scripts/train.py --config configs/exp3_baseline_2_e1d2.yaml
python scripts/train.py --config configs/exp3_baseline_3_finetune.yaml

# experiment 4 (complex) - needs exp2
python scripts/train.py --config configs/exp4_complex_1_fm.yaml
python scripts/train.py --config configs/exp4_complex_2_e1d2.yaml
python scripts/train.py --config configs/exp4_complex_3_finetune.yaml

# evaluate the FINAL model of each experiment (same metrics for all), then compare
python scripts/evaluate.py --config configs/exp1_baseline_ae.yaml
python scripts/evaluate.py --config configs/exp3_baseline_3_finetune.yaml     # ... etc.
python scripts/compare.py --split test                                         # markdown table
```
Override anything from the command line: `--set train.lr=3e-4 train.max_steps=50000 data.batch_size=32 fm.backbone=dit`.
Steps 1 and 2 of experiments 3/4 are independent of each other (can run in parallel); step 3 needs both.
`python scripts/count_params.py` prints parameter counts per part for both architectures.

## 3. Downstream evaluation with SAM (amplitude only)

Question answered: *does a segmentation model see the same thing in the reconstruction as in the original?* No ground-truth masks:
everything is agreement between original and reconstruction. SAM cannot see phase, so phase quality is covered by the phase metrics instead.

```bash
pip install transformers                                     # optional dependency
python scripts/evaluate.py --config configs/exp2_complex_ae.yaml --sam
python scripts/evaluate.py --config configs/exp3_baseline_3_finetune.yaml --sam --set sam.model=facebook/sam-vit-large
```
How it works (`sarcomp/sam/evaluate.py`):
1. **Mosaic / stitching.** The models only take 256x256 tiles but SAM needs a larger scene, so a block of tiles (default 4 down x 19 across,
   about 330 m x 330 m on the ground because azimuth pixels are ~4.7x finer than range pixels) is cut from the test split, every tile is
   reconstructed *independently* and the results are pasted back into one image. Seams between tiles are not blended, so seam artefacts count against the model.
2. **Image prep:** log-amplitude, azimuth area-averaged to square ground pixels, grey window (2-98 percentile) taken from the *original* and applied to both, uint8, 3 channels.
3. **Metrics** (same prompts on both images): `sam_embed_cos` (cosine similarity of SAM image embeddings), `sam_mask_iou` (mean IoU of the masks for an 8x8 grid
   of point prompts), `sam_mask_iou_ge50` (fraction of prompts with IoU >= 0.5). A perfect reconstruction scores exactly 1 (checked by the smoke test).

**Swapping the SAM model:** change the config, no code. `sam.model` = `facebook/sam-vit-base | -large | -huge` or a local folder; `sam.backend` selects the adapter.
For another model family (SAM 2, SAM-HQ, a fine-tune) write a small `SegBackend` subclass with `set_image / embedding / predict` and `@SAM_BACKENDS.register("name")`
(`sarcomp/sam/backends.py`). The `toy` backend is a deterministic stand-in used by tests so they need no weights.

Caveats: SAM is trained on natural RGB images, so SAR is out of distribution; treat the numbers as relative between models, not absolute.
Flow-matching reconstructions are stochastic (noise seed fixed by `eval.seed`) and may score well here while differing in fine texture, so read these together with the pixel/phase metrics.

## 4. Plug and play

| to change | do this | where |
|---|---|---|
| architecture (baseline vs complex, or a new one) | `model.arch: ...`; new one: subclass `Ops`, `@ARCHS.register("name")` | `sarcomp/models/arch.py` |
| latent sizes / depth | `model.e1_widths`, `model.e2_widths` (REAL channel counts; one stride-2 stage per entry; `2^len(e2_widths)` must equal the latent1 grid, `image_size/2^len(e1_widths)`) | config |
| activation / normalisation | `model.act`, `model.norm` (`none`/`group`) | config |
| flow-matching network | `fm.backbone: conv_film \| dit` + `fm.backbone_args`; new one: `@BACKBONES.register` | `sarcomp/fm/backbones.py` |
| FM sampler | `sampling.solver: euler \| midpoint \| heun`, `sampling.steps` | config |
| what is trained / frozen / loss | new `Stage` subclass, `@STAGES.register("name")` | `sarcomp/training/stages.py` |
| metrics | `@METRICS.register("name")` on a `Metric` subclass; list in `eval.metrics` | `sarcomp/metrics/` |
| SAM / segmentation model for the downstream eval | `sam.backend`, `sam.model`; new family: `@SAM_BACKENDS.register` | `sarcomp/sam/` |
| data | `data.source: tif \| synthetic`; patch size, split fractions, crops | `sarcomp/data/` |

The four parts `E1, E2, D1, D2` are separate `nn.Module`s inside `HierarchicalAE` (`sarcomp/models/autoencoder.py`), so any of them can be
swapped, frozen, or re-initialised independently.

## Layout

```
sarcomp/
  data/        transforms.py (log-complex <-> amp/phase), dataset.py (TIFF patches, synthetic stand-in)
  models/      ops.py (complex conv / conv-transpose / norm), arch.py (baseline & complex "ops"), autoencoder.py (E1 E2 D1 D2)
  fm/          backbones.py (conv_film, dit), flow.py (conditional OT flow matching, solvers, latent normaliser)
  training/    stages.py (ae, fm, e1d2, finetune), trainer.py, ema.py
  metrics/     base.py (framework + evaluate loop), basic.py (starter metrics)
  sam/         backends.py (swappable SAM adapters), evaluate.py (mosaics, agreement metrics)
  pipelines.py inference pipelines for exp 1/2 (AE) and exp 3/4 (FM)
  smoke.py     the smoke test
configs/       base.yaml + one small file per experiment stage (inherit with `base:`)
scripts/       train.py  evaluate.py  compare.py  count_params.py
notebooks/     smoke_test.ipynb (Colab)
tests/         pytest wrappers
docs/data.md   data guide
```

## Notes and assumptions

* **Parameter counts differ between architectures by construction.** Widths are REAL channel counts so latents hold the same number of real
  values; a complex layer at the same real width has half the parameters (default: baseline ~223M vs complex ~112M for the AE). Use
  `scripts/count_params.py` and the width settings if you want parameter-matched comparisons.
* **Compression ratio** is reported in *values* (131,072 inputs vs 2,048 latent2 values = 64x). No quantisation or entropy coding is
  implemented, so bits-per-pixel is not reported yet.
* FM evaluation is stochastic; `evaluate` fixes the noise seed (`eval.seed`) so runs are comparable.
* He initialisation is used for all convs (PyTorch's default makes this 17-layer no-norm network's gradients vanish; found by the smoke test).
* Phase DN -> radians mapping is assumed to be [-pi, pi) (offset only; see `docs/data.md`).
* Default training hyperparameters in `configs/base.yaml` are untuned placeholders.
