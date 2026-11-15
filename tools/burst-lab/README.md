# Burst lab

`arco_lab.py` runs Arco's multi-frame pipeline on a computer, so tone, colour and noise can be
tuned on real bursts without rebuilding the app. It also holds prototypes of algorithms that are
**not in the app**, with a synthetic burst generator to measure them against a known truth.

Needs Python 3 with numpy, scipy and Pillow.

## Getting a burst from the phone

1. Settings → General → **Save bursts for tuning** (developer option, off by default).
2. Take an MF photo. The app writes into
   `Android/data/org.cityman.camera/files/bursts/burst_<date>/`:
   - `frame_NN.raw16`: every RAW frame, 16-bit little endian, tightly packed;
   - `meta.json`: size, CFA layout, black/white level, AWB gains, colour matrix, lens shading map,
     burst plan (ISO, exposure, gain, under-exposure, highlight frames) and the processing options;
   - `hal.jpg`: the camera's own JPEG of the same moment;
   - `arco.jpg`: what the app saved.
3. Copy it: `adb pull /sdcard/Android/data/org.cityman.camera/files/bursts/ .`

A 12-frame burst of the rear camera takes about 480 MB: turn the option off again when done.

## Commands

```bash
# develop as the app does, and put it next to the HAL and app JPEGs
python3 arco_lab.py develop bursts/burst_20261105_201500 -o lab.jpg --compare

# try other choices: merge, tone mapping, hot pixel filter, tone parameters
python3 arco_lab.py develop BURST -o lab.jpg --merge wiener --tone laplacian --hot cross \
    --compress 0.68 --lift 0.92 --contrast 0.3 --sat 1.1

# super-resolution prototype on a centred crop (raw pixels)
python3 arco_lab.py sr BURST -o crop.png --size 512

# synthetic burst with known truth, and the benchmark of every method on it
python3 arco_lab.py synth OUT_DIR --frames 8
python3 arco_lab.py bench OUT_DIR
```

The lab follows the shaders step by step but in floating point and with simpler resampling, so
its output is close to the app's, not bit-exact.

## Results on the synthetic burst (5 November 2026)

The synthetic burst is a 768×768 RGGB chart. It has a textured background, a zone plate and bars
up to 0.35 cycles per raw pixel, coloured patches and 0.02% hot pixels. It has 8 frames with
random hand shake up to ±3 px, shot noise and read noise. PSNR is measured against the truth on
the quad grid.

| What | Result | Verdict |
|---|---|---|
| Hot pixel filter of the app (PACK) | changes 2.5% of the samples, almost all on fine detail: 18.8 dB, against 31.0 dB with no filter at all (hot pixels included) | **problem**: flattens detail near the Nyquist limit (fine text, foliage, stars) |
| Cross-colour hot pixel test (prototype) | 27.0 dB, but lets more hot pixels through (71 of 97 visible ones left, the app filter leaves 43) | promising; needs a temporal test (hot pixels stay still on the sensor, the scene moves) before it goes in the app |
| Robust merge (app) | 32.2 → 40.2 dB on texture with 8 frames | works as expected (√8 ≈ 9 dB) |
| Wiener merge per tile, HDR+ style | same on texture, +1.3 dB where alignment fails (aliasing patterns) | small gain; an FFT per tile on GLES is a lot of work for it |
| Super-resolution by splatting samples | below the app's merge + bicubic in every region tried | **not ready**: needs anisotropic kernels and alignment on the full RAW grid |
| Local Laplacian tone mapping | runs; judged by eye only | to compare with the app's two-scale tone mapping on real bursts |
| Noise fit | correct on flat data; 10× too high on this harsh chart (residual of sub-pixel texture) | check on real bursts before changing the app |

Nothing from these prototypes is in the app yet: each one waits for a comparison on real bursts.

## Spatial denoise (15 November 2026)

`python3 arco_lab.py dnbench OUT_DIR` makes synthetic bursts at three noise levels (`low`, `mid`,
`high`) and 1, 3 and 8 frames, merges them and runs each spatial denoise variant on the merged quad
image. It reports PSNR against the truth over the whole chart, over the textured background and
over the fine detail (zone plate and bars), plus SSIM of the grey level, and writes crops side by
side. By default the variants get the true noise; `--fitted-noise` uses the noise fitted from the
burst, as the app does.

| Variant | Result |
|---|---|
| Old app denoise (5×5 bilateral on the grey level, up to 0.4.0) | best SSIM on the smooth background, but fine detail **3-5 dB below no denoise at all**: the grey guide (mean of a 2×2 block) does not see detail near the Nyquist limit, so each colour plane is smoothed over 10×10 raw pixels; false colour on the zone plate |
| Multi-scale (opponent space, Laplacian pyramid, local Wiener gain per band) | overall PSNR +2 to +5 dB against the old denoise at every level, fine detail kept (above no denoise), texture within ±0.7 dB of the old denoise, SSIM 0.003-0.03 lower at high noise |
| Threshold for grey 0.7 / 1.0 / 1.4 | within 0.1 dB; 1.4 smoother at high noise. The app uses 1.2 |
| Threshold for colour 3 instead of 2 | always worse (−0.2 to −0.9 dB) |
| Grain put back (0.15 of the removed fine grey noise) | same PSNR, slightly lower SSIM, by design; the app keeps 0.1 |
| Bilateral for the finest band, multi-scale for the coarse ones | as bad as the bilateral alone on fine detail |
| Colour step of the finish: guided filter instead of the blurred half image | worse colour error (up to −2 dB), so the blur stays |

The multi-scale denoise is in the app since this date (`DN_DOWN`, `DN_UP`). It has not been
compared on real bursts yet.

The noise fitted from a synthetic burst is 3-9 times too high (the residual of the texture between
two frames counts as noise), which makes any denoise smooth too much. On real bursts this is still
to be checked against flat areas.

## References

- Hasinoff et al., *Burst photography for high dynamic range and low-light imaging on mobile
  cameras* (SIGGRAPH Asia 2016): tile alignment and frequency-domain merge.
- Wronski et al., *Handheld multi-frame super-resolution* (SIGGRAPH 2019).
- Paris, Hasinoff, Kautz, *Local Laplacian filters* (SIGGRAPH 2011); Aubry et al., *Fast local
  Laplacian filters* (ACM TOG 2014).
- Malvar, He, Cutler, *High-quality linear interpolation for demosaicing of Bayer-patterned
  color images* (ICASSP 2004).

The code is written from these papers: no code from their implementations is used.
