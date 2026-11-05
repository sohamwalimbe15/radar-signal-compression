# ICEYE SLC dataset – data guide for radar signal compression

This document explains what the two data files are, what is inside them, how the
pixels are encoded, and how `sarcomp/data/dataset.py` turns them into train/val/test loaders.

Legend: **[verified]** = measured/read from the files in this folder,
**[from metadata]** = stated in the geojson, **[assumption]** = not confirmed anywhere in the files, check before relying on it.

---

## 1. Files

| file | size | what it is |
|---|---|---|
| `ICEYE_DR72HB_20260817T165443Z_10330165_X35_SLED_SLC.tif` | ~2.9 GB | The image: a Single Look Complex (SLC) SAR product in Cloud-Optimized GeoTIFF layout |
| `ICEYE_DR72HB_20260817T165443Z_10330165_X35_SLED_SLC.geojson` | 19 KB | The metadata: a STAC *Item* describing the acquisition, geometry, processing and calibration |
| `sarcomp/data/dataset.py` | | Dataset / split / DataLoader code |
| `python -m sarcomp.smoke` | | Sanity checks for the three loaders |

Filename decoded: `ICEYE_<scene DR72HB>_<UTC start 2026-08-17 16:54:43>_<image id 10330165>_<satellite X35>_<mode SLED>_SLC`.

### What "SLC" means
A *Single Look Complex* image is the focused radar image **before** any speckle
averaging (looks = 1 in both directions, **[from metadata]**) and **before** any
conversion to a map projection. Every pixel is one complex number
`z = A · e^{jφ}`: an amplitude `A` (how strongly the ground cell scattered) and a phase `φ`
(related to sub-wavelength distance to the scatterer plus the scatterer's own phase). Because the phase is kept, an SLC contains
roughly twice the information of the familiar "grey-scale radar picture" (which is amplitude only),
and it is what makes interferometry, coherent change detection, and sub-aperture/speckle analysis possible.

---

## 2. The GeoJSON (STAC Item)

It is one STAC 1.0.0 `Feature` with the SAR, SAT, view, processing, projection and raster extensions plus
ICEYE's own `iceye:*` product-spec fields. The top-level keys are
`stac_version, stac_extensions, id, type, geometry, bbox, properties, links, assets`.

- `geometry` / `bbox`: footprint polygon in lon/lat (WGS84). Approx. lon −74.001…−73.929, lat 40.757…40.812, i.e. the Upper-West-Side-to-Upper-East-Side strip of Manhattan including Central Park. **[verified]** (`links` is empty.)
- `assets`: points to `slc-cog` (the TIFF) and `slc-json` (a sidecar JSON with the same content, not present here).

### 2.1 `properties`, grouped by purpose

**Acquisition**
| field | value | meaning |
|---|---|---|
| `platform` | ICEYE-X35 | satellite |
| `datetime`, `start/end_datetime` | 2026-08-17 16:54:43.8 → 16:54:56.1Z | collection time (~12.3 s of data collected) |
| `sar:instrument_mode` | spotlight | steers the beam to dwell on the scene → very fine azimuth resolution |
| `sar:frequency_band`, `center_frequency` | X, 9.65 GHz | wavelength λ ≈ 3.1 cm |
| `sar:polarizations` | VV | transmit V, receive V; one polarisation only |
| `sar:observation_direction` | left | |
| `sat:orbit_state` | descending | |
| `view:off_nadir`, `incidence_angle` | 33.2°, 35.6° | viewing geometry (near 35.36° → far 35.88° across the swath) |
| `iceye:range_near/range/range_far` | 486.3 / 487.7 / 489.2 km | slant range from sensor to scene (satellite altitude ~402 km) |
| `iceye:squint_angle` | 0.07° | essentially zero-squint / zero-Doppler geometry |

**Resolution and sampling** (important for compression)
| field | value | meaning |
|---|---|---|
| `sar:resolution_range` | 0.4997 m | true range resolution = c / (2·B) with B = 300 MHz |
| `sar:resolution_azimuth` | 0.08 m | true azimuth resolution |
| `sar:pixel_spacing_range` | 0.3232 m | sample spacing along range |
| `sar:pixel_spacing_azimuth` | 0.0686 m | sample spacing along azimuth |
| `iceye:pulse_bandwidth` | 300 MHz | chirp bandwidth |
| `iceye:pulse_duration` | 31.8 µs | |
| `iceye:acquisition_prf` / `processing_prf` | 6279 Hz / 106,305 Hz | pulse repetition frequency as acquired, and the (up-sampled) rate used in processing |
| `iceye:acquisition_range_sampling_rate` | 463.8 MHz | ADC rate → c/(2·fs) = 0.3232 m spacing |
| `iceye:synthetic_aperture_angle` | 11.1° | |
| `iceye:processing_bandwidth_azimuth` | 91.1 kHz | Doppler bandwidth that was focused |
| `iceye:processing_mode` | SLED | ICEYE's spotlight "SLED" mode |

Oversampling factors (my computation from the above): range `0.4997/0.3232 ≈ 1.55×`, azimuth `0.08/0.0686 ≈ 1.17×`.
The data are therefore sampled **finer than the resolution cell**, so neighbouring pixels are not independent. That redundancy is
what a compression model can exploit (see §5).

**Orbit / Doppler / geometry** (needed only if you want to re-focus, geocode, or physically model the signal)
`iceye:orbit_states` (50 ECEF position/velocity state vectors, 2026-08-17 16:54:43–16:54:56),
`doppler_centroid_coeffs` + `doppler_centroid_datetimes` (10 time samples, −4667 Hz → +442 Hz),
`doppler_rate_coeffs`, `zero_doppler_start/end_datetime`, `incidence_angle_coeffs`, `grazing_angle*`,
`ground_to_slant_coeff` (polynomial range ground→slant), `focal_plane_normal`, `image_plane_normal`,
`layover_angle`, `shadow_angle`, `orientation` (= shadows-down), `coordinate_frame` (= ecef), `orbit_precision` (= precise).

**Geocoding**
- `iceye:rpc`: 20-term rational polynomial coefficients (line/sample ↔ lat/lon/height) → lets you map any pixel to a ground location (e.g. with rasterio/GDAL RPC warping).
- `proj:code` = EPSG:4326, `proj:shape` = [9010, 72892] (rows, cols), `proj:transform`, `proj:centroid`.
  Caution: the TIFF itself has **no CRS and an identity transform**; the image is in radar slant-range/azimuth coordinates and the `proj:*` transform is only an approximate affine description. For real geocoding use the RPC.

**Radiometry**
- `iceye:calibration_factor` = 2.6126e-5, `iceye:amplitude_mapping` = `{function: minmax, parameters: {scale: 1, exponent: 1}}`.
  The way these convert stored DN into physical backscatter (σ⁰/β⁰) is defined in ICEYE's product spec, which is not in these files. **[assumption]** Treat DN as relative amplitude unless you confirm the conversion in the ICEYE SLC product specification; the loaders do not apply it.

**Raster**
- `raster:bands`: two bands, both `uint16`.

---

## 3. The TIFF

**[verified]** unless stated.

| property | value |
|---|---|
| width × height | 72,892 × 9,010 pixels (≈ 657 M pixels per band) |
| bands | 2 |
| dtype | uint16 |
| layout | tiled, 512×512 blocks, **pixel-interleaved** (`[row, col, band]` on disk), no compression |
| overviews | decimation levels 4, 8, 16 |
| CRS / transform | none / identity (slant geometry, see above) |
| nodata flag | none set |
| band 1 name tag | `AMPLITUDE` |
| band 2 name tag | `PHASE` |
| dataset tags | `ID, PRODUCT_FILE, PRODUCT_NAME, PRODUCT_TYPE=SLC-COG, ACQUISITION_MODE=spotlight, ICEYE_PROPERTIES` (the full STAC properties as a JSON string, so the TIFF is self-describing) |

> **Not I/Q.** Despite being an "SLC", this file stores **amplitude and phase**, not real/imag. To get the complex
> value: `z = amp · exp(j·phase)` and then `I = Re z`, `Q = Im z`.

> **Overviews:** the overviews are made by averaging. Averaging amplitude is fine for previews, but **averaging phase is meaningless**; never use overviews as training data.

### 3.1 Axes
- **Width (columns, 72,892 px × 0.0686 m ≈ 5.0 km) = azimuth** (along-track, the direction the satellite flies). Matches `iceye:extent_azimuth = 5000`.
- **Height (rows, 9,010 px × 0.3232 m ≈ 2.9 km) = slant range** (across-track). Top = near range or far range, depends on orientation. Metadata says `shadows-down`, which implies far range is at the bottom.
- Pixels are therefore anisotropic: one range pixel is ≈ 4.7× larger on the ground than one azimuth pixel. Patches that look square in pixels are not square on the ground (≈ 83 m × 17.6 m for 256×256).
- The preview rendered during exploration looked like a heavily stretched Manhattan: the grid of streets and Central Park are recognisable, with the stretching due to this anisotropy and slant geometry.

### 3.2 Band 1 – Amplitude (uint16)
Measured on 300 random 512×512 tiles:

| statistic | value |
|---|---|
| min / median / 90% / 99% / 99.9% / 99.99% / max | 0 / 26 / 61 / 200 / 597 / 1570 / 59,652 |
| fraction exactly 0 | 0.11% |
| fraction = 65535 or ≥ 60000 | 0 (nothing saturates the 16-bit range) |

- It is **extremely heavy-tailed**: the bulk of pixels are < 100, but strong scatterers (building corners, metal, bridges, "bright points" due to double-bounce in the urban grid) reach thousands to tens of thousands. A 16-bit range is used in only the top ~0.01%.
- Mean (≈ 32–36 depending on region) is much larger than median (26): speckle makes pixel amplitudes Rayleigh-like in homogeneous areas, and the urban area adds bright outliers.
- Practical consequence: **do not train on raw DN with MSE** (a few bright pixels dominate). The loader uses `log1p(DN)` then z-score with train-region stats (mean 3.263, std 0.790); `amp_dn` (raw) is also returned for metrics and for inverting the transform: `DN = expm1(amplitude·std + mean)`.
- Zeros: probably areas without signal, in or near very dark regions (water, shadow) or the image margins. Not filtered.

### 3.3 Band 2 – Phase (uint16)
- Measured range 0…65534, **uniformly distributed** (each of 8 equal bins holds 12.5% ± 0.01%).
- Interpreted as a linear map of the **wrapped** phase onto the 16-bit range. The loader uses `φ = DN/65535·2π − π ∈ [−π, π)`. **[assumption]**: whether the file means [0, 2π) or [−π, π) is not stated in the files. This changes only a constant offset: correlations and phase differences are unaffected, but be consistent if you later compare against another tool.
- Phase has a **quantisation step of 2π/65535 ≈ 9.6e-5 rad**, effectively lossless as far as any model is concerned.
- Phase is **circular**: −π and +π are the same. Plain MSE penalises 3.14 vs −3.14 as a huge error when they are almost identical. Use a wrapped error `angle(exp(j(φ̂−φ)))`, or predict `(cos φ, sin φ)`, or predict the complex number directly.
- The marginal distribution of phase is uniform, as expected for fully-developed speckle, so there is no per-pixel prior to exploit. **The structure is in the relationships between neighbouring pixels** (below).

### 3.4 Inter-pixel structure (measured on 40 random 1024×1024 windows)
Normalised complex correlation `ρ = Σ z₁ z₂* / sqrt(Σ|z₁|² Σ|z₂|²)` between a pixel and its neighbour at lag 1:

| direction | \|ρ\| | mean ρ (complex) |
|---|---|---|
| range (rows) | 0.43 | 0.41 − 0.12j |
| azimuth (columns) | 0.18 | 0.17 − 0.01j |

Phase-only coherence `|mean(e^{j(φ₁−φ₂)})|`: range 0.30, azimuth 0.12.

Reading these: neighbouring complex samples are clearly correlated (more in range, consistent with its 1.55× oversampling) and the mean ρ has a
small phase angle (≈ −0.28 rad in range, ≈ −0.06 rad in azimuth), i.e. only a modest residual phase ramp, so the data is
close to baseband/demodulated. So there is real redundancy, but not a lot: this is **speckle-dominated, noise-like data that is hard to compress losslessly**; expect gains
mostly from (i) lossy/perceptual-style coding, (ii) exploiting range oversampling and amplitude/structure of man-made scenes, and (iii) the strong spatial
structure of bright scatterers.

---

## 4. How the data is split and loaded (`sarcomp/data/dataset.py`)

**Split:** along the width (azimuth), left to right, so train/val/test are spatially disjoint strips of the same scene.

| split | columns [x0, x1) | width | 256×256 patches (non-overlapping) |
|---|---|---|---|
| train (60%) | 0 – 43,735 | 43,735 | random crops (grid would be 5,950 = 170 × 35) |
| val (20%) | 43,735 – 58,313 | 14,578 | 1,960 (56 × 35) |
| test (20%) | 58,313 – 72,892 | 14,579 | 1,960 (56 × 35) |

- Patches never straddle a boundary, so there is no overlap/leakage between splits (asserted in `python -m sarcomp.smoke`).
  The leftover columns at the right edge of each split that do not fill a full patch are dropped.
- Train uses **random crops**: each item is a crop at a random (x, y) fully inside the train columns, and an epoch is `samples_per_epoch` crops (default 2000). Seed with `torch.manual_seed` for reproducibility. Val/test always use the fixed non-overlapping grid so evaluation is deterministic.
- Caveat: a left-to-right split also means the splits differ in content (e.g. the left part of the image is a denser, brighter area; mean amplitude 36.0 / 35.5 / 32.4 for train / val / test), so
  val/test scores are an honest out-of-region estimate but can differ systematically from train.
- Reading is lazy: each worker opens the TIFF on first use and reads only the 256×256 window (the 2.9 GB file never needs to be in RAM).
  Because the TIFF is **uncompressed and pixel-interleaved**, a window read touches both bands anyway.

**Each sample (dict):**

| key | shape / dtype | content |
|---|---|---|
| `x` | (2,H,W) float32 | log-compressed complex image, channels `[Re, Im]` of `log1p(A)/scale * exp(j*phase)` |
| `xy` | (2,) int64 | (x, y) top-left pixel in the full image |

`scale` is the RMS of `log1p(amplitude DN)` over the TRAIN columns (3.357 for this scene), cached in `data.stats_path`.
The transform is exactly invertible (`sarcomp.data.LogComplex.inverse`): `A = expm1(|z|*scale)`, `phase = atan2(Im, Re)`.
Why log-compress: raw I/Q is dominated by a few bright pixels (see 3.2), and pixels with tiny amplitude get a tiny `|z|`, so their
meaningless noise phase barely counts in an L2 loss. Round trip is tested in `python -m sarcomp.smoke`.

Usage:
```python
from sarcomp.config import load_config
from sarcomp.data import build_data

if __name__ == "__main__":                      # required on macOS (spawn)
    cfg = load_config("configs/base.yaml", ["data.data_dir=."])
    loaders, tf = build_data(cfg["data"])       # loaders["train" | "val" | "test"], tf = LogComplex
    x = next(iter(loaders["train"]))["x"]       # (B, 2, 256, 256)
    amp_dn, phase = tf.inverse(x)               # back to raw amplitude DN and radians
```
Synthetic stand-in for smoke tests (no TIFF needed): `data.source: synthetic`.

---|---|---|
| `amplitude` | (1,H,W) float32 | `(log1p(DN) − 3.263) / 0.790` |
| `phase` | (1,H,W) float32 | radians in [−π, π) |
| `amp_dn` | (1,H,W) float32 | raw amplitude DN, for metrics / inversion |
| `xy` | (2,) int64 | (x, y) top-left pixel in the full image |

Usage:
```python
from sar_data import make_loaders

if __name__ == "__main__":                      # required on macOS (spawn)
    ds, loaders, amp_stats = make_loaders(patch=256, batch_size=16, num_workers=4)
    for batch in loaders["train"]:
        amp, phase = batch["amplitude"], batch["phase"]
        ...
```
Verification: `python check_loaders.py` (shapes, finiteness, phase range, boundaries). Last run: all checks passed.

---

## 5. Using this for radar signal compression: practical guidance

1. **Choose the representation the model sees and outputs.** Options, in my order of recommendation:
   - *Complex as 2 channels* `(I, Q)` or `(amp-ish, cos φ, sin φ)`. Avoids the phase wrap problem and lets the network use the correlation structure.
   - *Amplitude + phase as separate heads*, with a wrapped phase loss. Matches the way you described the target (amplitude and phase) but needs care because phase is meaningless where amplitude ≈ 0.
   Build I/Q from the loader outputs: `amp_lin = amp_dn` (or `expm1(...)`), `I = amp_lin·cos φ`, `Q = amp_lin·sin φ`.
2. **Loss.**
   - Amplitude: L1/L2 on the normalised log-amplitude (matches the dynamic range), optionally plus relative error on `amp_dn`.
   - Phase: `mean(1 − cos(φ̂ − φ))` or the squared wrapped difference; weight by amplitude (or mask low-amplitude pixels) because phase of near-zero pixels is noise and unlearnable.
   - Complex: MSE on (I, Q) is dominated by bright pixels; consider a log-compressed complex loss, e.g. `log1p(|z|)·e^{j∠z}`.
3. **Metrics to report** (computed on the *test* split, on raw-amplitude and phase after inverse transform): bits per pixel (or compression ratio vs. 32 bit/pixel for the two uint16 bands), amplitude PSNR/SSIM in log domain, wrapped phase error (degrees), amplitude-weighted phase error, complex SNR / NMSE, and ideally a downstream-task check (e.g. coherence or point-target response preservation, since compressing speckle-phase badly destroys interferometric usefulness).
4. **Expectations.** Speckle is incompressible noise-like signal; most pixels' phase is random relative to a pure-amplitude description. At high compression the model will mostly preserve the amplitude structure (buildings, park, water) and phase only for strong, coherent scatterers. Use bits-per-pixel/rate–distortion curves rather than a single score.
5. **Patch shape.** Pixels are anisotropic (4.7:1). Square pixel patches are fine, but consider range-direction strides/kernels that account for the oversampling difference, and note that azimuth neighbours decorrelate faster.
6. **Data-hygiene reminders.**
   - Compute normalisation stats from train only (done).
   - Never use overviews as data.
   - Zero-amplitude pixels (0.11%) can create `log1p = 0` and meaningless phase; decide on a mask.
   - Only a single scene/acquisition is available. This is one image, so generalisation to other scenes, polarisations, or modes is untested. For generalisation claims, you'd need more scenes.

---

## 6. Summary of unverified assumptions
1. Phase DN → radians mapping ([−π, π) vs [0, 2π)).
2. Physical meaning of amplitude DN (`amplitude_mapping = minmax`, `calibration_factor`); see ICEYE SLC product specification.
3. Which end of the height axis is near range (inferred from `orientation = shadows-down`).
4. Zero-valued amplitude pixels: assumed to be no-signal areas; not inspected individually.
