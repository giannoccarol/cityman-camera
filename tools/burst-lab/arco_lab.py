#!/usr/bin/env python3
"""
Arco burst lab: the multi-frame pipeline of the app, on a computer.

It reads a burst saved by the app's developer option "Save bursts for tuning"
(Android/data/org.cityman.camera/files/bursts/<name>: frame_NN.raw16, meta.json,
hal.jpg, arco.jpg) and develops it with the same steps and parameters as the GPU
shaders, so tone, colour and noise can be tuned without the phone. It also holds
prototypes that are not in the app yet (frequency-domain merge, local Laplacian
tone mapping, multi-frame super-resolution), and a synthetic burst generator with a
known ground truth to measure them.

    python3 arco_lab.py develop BURST_DIR -o out.jpg [--merge robust|wiener|single]
                        [--tone app|laplacian] [--compare]
    python3 arco_lab.py sr BURST_DIR -o crop.png [--size 512]
    python3 arco_lab.py synth OUT_DIR [--frames 8] [--image some.png]
    python3 arco_lab.py bench OUT_DIR  # synthetic burst, PSNR of each method

GPL-3.0-or-later, like the app.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

# ---------------------------------------------------------------------------
# Loading

def load_burst(path):
    meta = json.load(open(os.path.join(path, "meta.json")))
    w, h = meta["width"], meta["height"]
    frames = []
    for i in range(meta.get("received", meta["frames"])):
        f = os.path.join(path, "frame_%02d.raw16" % i)
        if os.path.exists(f):
            frames.append(np.fromfile(f, dtype="<u2").reshape(h, w))
    if not frames:
        sys.exit("no frames in " + path)
    return meta, frames


def channel_positions(chan_at):
    """Raw (x, y) inside the 2x2 block of quad channels R, Ge, Go, B."""
    pos = [None] * 4
    for i, ch in enumerate(chan_at):
        pos[ch] = (i % 2, i // 2)
    return pos


HOT_CROSS = False  # False = the app's PACK shader; True = the cross-colour prototype (--hot cross)


def to_quad(raw, meta, scale=1.0, cross=None):
    """RAW16 -> (H/2, W/2, 4) black-subtracted, normalised, hot pixels fixed (as PACK)."""
    pos = channel_positions(meta["chanAt"])
    black = np.zeros(4)
    for i, ch in enumerate(meta["chanAt"]):
        black[ch] = meta["black"][i]
    white = meta["white"]
    rawf = raw.astype(np.float32)
    q = np.stack([rawf[y::2, x::2] for (x, y) in pos], axis=-1)
    q = fix_hot(rawf, q, pos, black, white, HOT_CROSS if cross is None else cross)
    q = (q - black) / (white - black)
    return np.clip(q, -0.1, 1.0) * scale


def fix_hot(raw, q, pos, black, white, cross=True):
    """
    A sample far above or below its 4 same-colour neighbours (2 px away) takes their median.
    With `cross`, a bright sample is only a hot pixel when it also stands well above its 4
    direct neighbours of the other colours: the lens spreads real detail (text, foliage, stars)
    over more than one pixel, a hot pixel is alone. Without it, fine detail near the Nyquist
    limit is flattened.
    """
    out = q.copy()
    rp = np.pad(raw, 2, mode="edge")
    H, W = raw.shape
    bmean = black.mean()
    for c in range(4):
        px, py = pos[c]
        sl = lambda dy, dx: rp[2 + py + dy: 2 + py + dy + H: 2, 2 + px + dx: 2 + px + dx + W: 2][: q.shape[0], : q.shape[1]]
        x = q[..., c]
        n = np.stack([sl(0, 2), sl(0, -2), sl(2, 0), sl(-2, 0)])
        hi, lo = n.max(0), n.min(0)
        rng = white - black[c]
        high = x > hi + 0.5 * (hi - black[c]) + 0.03 * rng
        low = x < lo - 0.5 * (lo - black[c]) - 0.03 * rng
        if cross:
            o = np.stack([sl(0, 1), sl(0, -1), sl(1, 0), sl(-1, 0)])
            ohi, olo = o.max(0) - bmean, o.min(0) - bmean
            high &= (x - black[c]) > 2.0 * ohi + 0.03 * rng
            low &= (x - black[c]) < 0.5 * olo - 0.03 * rng
        med = 0.5 * (n.sum(0) - hi - lo)
        out[..., c] = np.where(high | low, med, x)
    return out


def gray(q):
    return np.minimum(q, 1.0).mean(-1)

# ---------------------------------------------------------------------------
# Alignment (as ALIGN: pyramid ½, ¼, ¼ over the quad grey; tiles 16/8; L1 at the finest level)

LEVELS = [(1, 16, 1), (2, 8, 2), (4, 8, 4), (4, 8, 4)]   # (factor, tile, radius)


def pyramid(g):
    out = [g]
    for f, _, _ in LEVELS[1:]:
        p = out[-1]
        h, w = p.shape[0] // f * f, p.shape[1] // f * f
        out.append(p[:h, :w].reshape(h // f, f, w // f, f).mean((1, 3)))
    return out


def sample(img, yy, xx):
    """Nearest sample with clamping."""
    return img[np.clip(yy, 0, img.shape[0] - 1), np.clip(xx, 0, img.shape[1] - 1)]


def align(ref_pyr, alt_pyr):
    """Integer offsets per finest tile (dy, dx) plus a sub-pixel part from a parabola."""
    prev = None
    for li in range(len(LEVELS) - 1, -1, -1):
        f, t, r = LEVELS[li]
        ref, alt = ref_pyr[li], alt_pyr[li]
        nty, ntx = math.ceil(ref.shape[0] / t), math.ceil(ref.shape[1] / t)
        if prev is None:
            init = np.zeros((nty, ntx, 2), np.int64)
        else:
            pf, pt = LEVELS[li + 1][0], LEVELS[li + 1][1]
            ty, tx = np.meshgrid(np.arange(nty), np.arange(ntx), indexing="ij")
            cy = np.clip((ty * t + t // 2) // pf // pt, 0, prev.shape[0] - 1)
            cx = np.clip((tx * t + t // 2) // pf // pt, 0, prev.shape[1] - 1)
            init = prev[cy, cx] * pf
        yy, xx = np.mgrid[0:nty * t, 0:ntx * t]
        tyi, txi = yy // t, xx // t
        refp = sample(ref, yy, xx)
        costs = np.zeros((2 * r + 1, 2 * r + 1, nty, ntx))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                a = sample(alt, yy + init[tyi, txi, 0] + dy, xx + init[tyi, txi, 1] + dx)
                e = np.abs(refp - a) if li == 0 else (refp - a) ** 2
                costs[dy + r, dx + r] = e.reshape(nty, t, ntx, t).sum((1, 3))
        best = costs.reshape(-1, nty, ntx).argmin(0)
        d = np.stack([best // (2 * r + 1) - r, best % (2 * r + 1) - r], -1)
        prev = init + d
        if li == 0:
            sub = np.zeros(prev.shape, np.float64)
            centre = (d[..., 0] == 0) & (d[..., 1] == 0)
            c = costs
            dyy = c[0, 1] - 2 * c[1, 1] + c[2, 1]
            dxx = c[1, 0] - 2 * c[1, 1] + c[1, 2]
            sub[..., 0] = np.where(centre & (dyy > 1e-9), np.clip(0.5 * (c[0, 1] - c[2, 1]) / np.maximum(dyy, 1e-9), -0.5, 0.5), 0)
            sub[..., 1] = np.where(centre & (dxx > 1e-9), np.clip(0.5 * (c[1, 0] - c[1, 2]) / np.maximum(dxx, 1e-9), -0.5, 0.5), 0)
            return prev + sub
    return prev


def warp(q, offs, tile=16):
    """Bilinear sampling of each colour plane at p + offset of p's tile."""
    h, w = q.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    o = offs[np.minimum(yy.astype(int) // tile, offs.shape[0] - 1), np.minimum(xx.astype(int) // tile, offs.shape[1] - 1)]
    coords = [yy + o[..., 0], xx + o[..., 1]]
    return np.stack([ndimage.map_coordinates(q[..., c], coords, order=1, mode="nearest") for c in range(4)], -1)

# ---------------------------------------------------------------------------
# Noise model (as NOISE + fitNoise: median |difference| per brightness bin)

def fit_noise(ref, alt):
    x = np.clip(ref, 0, None).ravel()[::7]
    d = np.abs(alt - ref).ravel()[::7]
    xs, vs, ws = [], [], []
    bins = (np.sqrt(x) * 16).astype(int).clip(0, 15)
    for b in range(16):
        sel = d[bins == b]
        if sel.size < 400:
            continue
        sigma = np.median(sel) / (0.6745 * math.sqrt(2))
        xs.append(((b + 0.5) / 16) ** 2)
        vs.append(sigma ** 2)
        ws.append(sel.size)
    if len(xs) < 3:
        return 1e-4, 1e-6
    a, b = np.polyfit(xs, vs, 1, w=np.sqrt(ws))
    return max(a, 1e-6), max(b, 1e-7)

# ---------------------------------------------------------------------------
# Merges

def merge_robust(qs, offsets, noise, strength=0.8, hdr_short=0):
    """As MERGE_DECIDE + MERGE_ADD: weight from the local grey difference against the noise."""
    ref = qs[0]
    a, b = noise
    acc, wsum = ref.copy(), np.ones(ref.shape[:2])
    gref = gray(ref)
    sigma = np.sqrt(a * np.clip(gref, 0, None) + b)
    for q, o in zip(qs[1:], offsets[1:]):
        w_alt = warp(q, o)
        diff = ndimage.uniform_filter(np.abs(gray(w_alt) - gref), 4)
        dn = diff / (0.5 * sigma)
        w = np.exp(-np.square(np.maximum(dn - 1.3, 0) / strength))
        acc += w[..., None] * w_alt
        wsum += w
    return acc / wsum[..., None], wsum


def merge_wiener(qs, offsets, noise, c=8.0, tile=16):
    """
    HDR+ style frequency-domain merge (Hasinoff et al. 2016): per 16×16 tile and colour plane,
    each aligned frame is pulled towards the reference by A = |D|² / (|D|² + c·σ²), D = T0 − Tz,
    in the Fourier domain; tiles overlap by half with a raised-cosine window.
    """
    ref = qs[0]
    h, w = ref.shape[:2]
    step = tile // 2
    win1 = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(tile) + 0.5) / tile)
    win = np.outer(win1, win1)
    a, b = noise
    aligned = [ref] + [warp(q, o) for q, o in zip(qs[1:], offsets[1:])]
    out = np.zeros_like(ref)
    norm = np.zeros((h, w))
    pad = [np.pad(x, ((step, tile), (step, tile), (0, 0)), mode="reflect") for x in aligned]
    H, W = pad[0].shape[:2]
    ys = range(0, H - tile + 1, step)
    xs = range(0, W - tile + 1, step)
    acc = np.zeros(pad[0].shape)
    nacc = np.zeros(pad[0].shape[:2])
    for y in ys:
        for x in xs:
            t0 = pad[0][y:y + tile, x:x + tile]
            mean = np.clip(t0.mean(), 0, None)
            var = (a * mean + b) * tile * tile          # noise power per frequency (unnormalised FFT)
            F0 = np.fft.rfft2(t0 * win[..., None], axes=(0, 1))
            res = F0.copy()
            for z in range(1, len(pad)):
                Fz = np.fft.rfft2(pad[z][y:y + tile, x:x + tile] * win[..., None], axes=(0, 1))
                D = F0 - Fz
                A = np.abs(D) ** 2 / (np.abs(D) ** 2 + c * var)
                res += Fz + A * (F0 - Fz)
            res /= len(pad)
            acc[y:y + tile, x:x + tile] += np.fft.irfft2(res, s=(tile, tile), axes=(0, 1)) * win[..., None]
            nacc[y:y + tile, x:x + tile] += win * win
    out = acc / np.maximum(nacc, 1e-6)[..., None]
    return out[step:step + h, step:step + w]

# ---------------------------------------------------------------------------
# Finishing (as PREP + FINISH + SHARPEN)

M_LUM = np.array([0.2126, 0.7152, 0.0722])


def lens_shading(meta, h, w):
    lsc = meta.get("lsc")
    if not lsc:
        return np.ones((h, w, 4))
    cols, rows = lsc["cols"], lsc["rows"]
    grid = np.array(lsc["rggb"], np.float64).reshape(rows, cols, 4)
    yy = np.linspace(0, rows - 1, h)
    xx = np.linspace(0, cols - 1, w)
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    return np.stack([ndimage.map_coordinates(grid[..., c], [Y, X], order=1) for c in range(4)], -1).clip(1, 8)


def mosaic_from_quad(mq, chan_at):
    h, w = mq.shape[:2]
    m = np.zeros((h * 2, w * 2))
    for i, ch in enumerate(chan_at):
        m[i // 2::2, i % 2::2] = mq[..., ch]
    return m


def demosaic_malvar(m, chan_at):
    """Malvar-He-Cutler 5×5 kernels (as FINISH)."""
    k = lambda rows: np.array(rows, np.float64) / 8.0
    G_at_RB = k([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]])
    R_at_G_rrow = k([[0, 0, .5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, .5, 0, 0]])
    R_at_G_rcol = R_at_G_rrow.T
    R_at_B = k([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]])
    conv = lambda kern: ndimage.convolve(m, kern, mode="mirror")
    g_rb, h_g, v_g, o_rb = conv(G_at_RB), conv(R_at_G_rrow), conv(R_at_G_rcol), conv(R_at_B)
    H, W = m.shape
    ch = np.zeros((H, W), int)
    for i, c in enumerate(chan_at):
        ch[i // 2::2, i % 2::2] = c
    red_right = np.roll(ch, -1, axis=1) == 0
    rgb = np.zeros((H, W, 3))
    isR, isB, isG = ch == 0, ch == 3, (ch == 1) | (ch == 2)
    rgb[..., 0] = np.where(isR, m, np.where(isB, o_rb, np.where(red_right, h_g, v_g)))
    rgb[..., 1] = np.where(isG, m, g_rb)
    rgb[..., 2] = np.where(isB, m, np.where(isR, o_rb, np.where(red_right, v_g, h_g)))
    return np.clip(rgb, 0, None)


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def shoulder(x, t=0.62):
    return np.where(x < t, x, t + (1 - t) * (1 - np.exp(-(x - t) / (1 - t))))


def local_laplacian_log(logl, sigma=1.0, alpha=0.6, levels=5, k=8):
    """
    Local Laplacian filtering of log luminance (Paris et al. 2011, fast version of Aubry et al.
    2014): k remapped copies, Laplacian pyramids, per-pixel interpolation. alpha < 1 compresses
    large edges (halo-free), detail below sigma is kept.
    """
    def gpyr(x):
        p = [x]
        for _ in range(levels - 1):
            p.append(ndimage.gaussian_filter(p[-1], 1.0)[::2, ::2])
        return p

    def lpyr(x):
        g = gpyr(x)
        out = []
        for i in range(levels - 1):
            up = ndimage.zoom(g[i + 1], np.array(g[i].shape) / np.array(g[i + 1].shape), order=1)
            out.append(g[i] - up)
        out.append(g[-1])
        return out

    gin = gpyr(logl)
    lo, hi = logl.min(), logl.max()
    refs = np.linspace(lo, hi, k)
    lout = [np.zeros_like(x) for x in gin]
    step = (hi - lo) / (k - 1) if k > 1 else 1
    for j, g0 in enumerate(refs):
        d = logl - g0
        remapped = g0 + np.sign(d) * np.where(np.abs(d) < sigma, np.abs(d), sigma + alpha * (np.abs(d) - sigma))
        lp = lpyr(remapped)
        for i in range(levels - 1):
            wgt = np.clip(1 - np.abs(gin[i] - g0) / step, 0, 1)
            lout[i] += wgt * lp[i]
    lout[-1] = gin[-1]
    out = lout[-1]
    for i in range(levels - 2, -1, -1):
        out = ndimage.zoom(out, np.array(lout[i].shape) / np.array(out.shape), order=1) + lout[i]
    return out


def finish(mq, meta, gains, ccm, opts, tone="app"):
    """Merged quad (normalised, before shading/WB) → sRGB uint8, as the app's finishing."""
    h, w = mq.shape[:2]
    k = lens_shading(meta, h, w) * np.array(gains)
    clip = k.min(-1, keepdims=True) * (meta.get("shortScale", 4) if meta.get("shortFrames", 0) else 1)
    v = np.minimum(np.clip(mq, 0, None) * k, clip)
    chan_at = meta["chanAt"]
    rgb = demosaic_malvar(mosaic_from_quad(v, chan_at), chan_at)
    lin = rgb @ np.array(ccm).reshape(3, 3).T
    if opts.get("chroma", True):
        half = np.stack([v[..., 0], 0.5 * (v[..., 1] + v[..., 2]), v[..., 3]], -1) @ np.array(ccm).reshape(3, 3).T
        half = ndimage.gaussian_filter(half, (2, 2, 0))
        blur = np.repeat(np.repeat(half, 2, 0), 2, 1)[:lin.shape[0], :lin.shape[1]]
        y, yb = lin @ M_LUM, blur @ M_LUM
        lin = np.clip(lin + opts.get("chroma_amount", 0.6) * (blur + (y - yb)[..., None] - lin), 0, None)
    gain = meta.get("gain", 1.0) * 0.85
    lin *= gain
    l = np.maximum(lin @ M_LUM, 1e-6)
    if opts.get("local", True):
        logl = np.log2(l)
        if tone == "laplacian":
            small = ndimage.zoom(logl, 0.25, order=1)
            out = local_laplacian_log(small, sigma=opts.get("ll_sigma", 1.0), alpha=opts.get("ll_alpha", 0.6))
            delta = ndimage.zoom(out - small, np.array(logl.shape) / np.array(small.shape), order=1)
            lin *= np.exp2(delta)[..., None]
        else:
            fine = ndimage.uniform_filter(ndimage.zoom(logl, 0.125, order=1), 3)
            coarse = ndimage.gaussian_filter(ndimage.zoom(logl, 1 / 32, order=1), 2)
            fine = ndimage.zoom(fine, np.array(logl.shape) / np.array(fine.shape), order=1)
            coarse = ndimage.zoom(coarse, np.array(logl.shape) / np.array(coarse.shape), order=1)
            base = np.where(np.abs(fine - coarse) > 0, coarse + (fine - coarse) * np.clip(np.abs(fine - coarse) / 1.5, 0, 1), coarse)
            hdr = np.clip((meta.get("underEv", 0.7) - 0.7) / 0.9, 0, 1)
            c = np.where(base > opts.get("anchor", -2.5), opts.get("compress", 0.72) - 0.1 * hdr, opts.get("lift", 0.94) - 0.07 * hdr)
            lin *= np.exp2((c - 1) * (base - opts.get("anchor", -2.5)))[..., None]
        l = np.maximum(lin @ M_LUM, 1e-6)
    bl = opts.get("black", 0.012)
    lb = np.maximum(l - bl, 0) / (1 - bl)
    lin *= (lb / l)[..., None]
    l = np.maximum(lb, 1e-6)
    lt = shoulder(l)
    lin *= (lt / l)[..., None]
    mx = lin.max(-1)
    lin = np.where((mx > 1)[..., None], lin + (lt[..., None] - lin) * np.clip((mx - 1) / np.maximum(mx, 1e-6), 0, 1)[..., None], lin)
    lin = np.clip(lt[..., None] + (lin - lt[..., None]) * opts.get("sat", 1.15), 0, None)
    out = srgb(lin)
    con = opts.get("contrast", 0.35)
    out = out + con * (out * out * (3 - 2 * out) - out)
    if opts.get("sharpen", True):
        y = out @ np.array([0.299, 0.587, 0.114])
        d = y - ndimage.gaussian_filter(y, 0.8)
        d = np.sign(d) * np.maximum(np.abs(d) - 0.004, 0)
        out = out + 0.55 * d[..., None]
    out = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
    rot = meta.get("rotation", 0)
    return np.rot90(out, k=-(rot // 90)) if rot else out

# ---------------------------------------------------------------------------
# Super-resolution prototype

def super_resolve(qs, offsets, noise, chan_at, crop, scale=2, centre=None, sigmas=(0.9, 0.65)):
    """
    Multi-frame super-resolution of a centred crop of `crop` raw pixels (the 2x zoom case), in the
    spirit of Wronski et al. 2019 without their anisotropic kernels: every Bayer sample of every
    frame is splatted with a Gaussian onto a grid `scale` times finer than the raw grid, at its
    true position (sub-pixel alignment), per colour, weighted by the same robustness as the merge.
    Gives linear RGB directly (no demosaic). Hand shake is what fills the grid: on a tripod the
    result is no better than a demosaic.
    """
    h, w = qs[0].shape[:2]
    half = crop // 4                       # half the crop, in quad pixels
    cy, cx = centre if centre else (h // 2, w // 2)
    y0, x0 = cy - half, cx - half
    n = crop * scale
    acc = np.zeros((n, n, 3))
    wsum = np.zeros((n, n, 3))
    gref = gray(qs[0])
    a, b = noise
    pos = channel_positions(chan_at)
    plane = [0, 1, 1, 2]
    sigma_raw = [sigmas[0], sigmas[1], sigmas[1], sigmas[0]]   # kernel width in raw pixels: R and B are sparser
    for f, (q, o) in enumerate(zip(qs, offsets)):
        if f == 0:
            rob = np.ones(gref.shape)
        else:
            diff = ndimage.uniform_filter(np.abs(gray(warp(q, o)) - gref), 4)
            sigma = 0.5 * np.sqrt(a * np.clip(gref, 0, None) + b)
            rob = np.exp(-np.square(np.maximum(diff / sigma - 1.3, 0) / 0.8))
        m = 4
        qy, qx = np.mgrid[max(y0 - m, 0):min(y0 + 2 * half + m, h), max(x0 - m, 0):min(x0 + 2 * half + m, w)]
        ty = np.minimum(qy // 16, o.shape[0] - 1)
        tx = np.minimum(qx // 16, o.shape[1] - 1)
        oy, ox = o[ty, tx, 0], o[ty, tx, 1]
        # robustness is known in reference coordinates: read it where this sample lands
        wr = rob[np.clip(np.round(qy - oy).astype(int), 0, h - 1), np.clip(np.round(qx - ox).astype(int), 0, w - 1)]
        for c in range(4):
            px, py = pos[c]
            # raw position of this sample in the reference frame, relative to the crop corner
            ry = (qy * 2 + py) - 2 * oy - 2 * y0
            rx = (qx * 2 + px) - 2 * ox - 2 * x0
            val = q[qy, qx, c]
            gy, gx = (ry + 0.5) * scale - 0.5, (rx + 0.5) * scale - 0.5
            s = sigma_raw[c] * scale
            r = int(math.ceil(2 * s))
            cy, cx = np.round(gy).astype(int), np.round(gx).astype(int)
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    iy, ix = cy + dy, cx + dx
                    ok = (iy >= 0) & (iy < n) & (ix >= 0) & (ix < n)
                    k = np.exp(-((iy - gy) ** 2 + (ix - gx) ** 2) / (2 * s * s)) * wr
                    np.add.at(acc[..., plane[c]], (iy[ok], ix[ok]), (k * val)[ok])
                    np.add.at(wsum[..., plane[c]], (iy[ok], ix[ok]), k[ok])
    return acc / np.maximum(wsum, 1e-6)


# ---------------------------------------------------------------------------
# Synthetic bursts with known ground truth

def synth_scene(n, seed=7):
    """
    Test chart, linear RGB, n x n: a textured background (detail at every scale, like foliage or
    fabric, so alignment has something to hold on everywhere), a zone plate and bars up to near
    the raw Nyquist limit, and coloured patches.
    """
    rng = np.random.default_rng(seed)
    tex = sum(ndimage.zoom(rng.random((n // s + 1, n // s + 1)), s, order=3)[:n, :n] / (1 + i)
              for i, s in enumerate((64, 16, 6, 3)))
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    lin = np.stack([0.08 + 0.35 * tex, 0.1 + 0.3 * tex, 0.06 + 0.25 * tex], -1)
    yy, xx = np.mgrid[0:n, 0:n] / n
    r2 = (xx - 0.3) ** 2 + (yy - 0.3) ** 2
    zone = 0.5 + 0.45 * np.cos(np.pi * n * 0.7 * r2)        # up to 0.35 cycles per raw pixel
    disc = r2 < 0.06
    lin[disc] = np.stack([zone] * 3, -1)[disc]
    bars = (yy > 0.6) & (xx > 0.55)
    lin[bars] = (0.15 + 0.7 * (np.floor(xx * n / 4) % 2))[bars][:, None]
    patch = (xx < 0.45) & (yy > 0.65)
    lin[patch] = np.array([0.6, 0.25, 0.1]) * (0.8 + 0.4 * tex[patch])[:, None]
    sky = (xx > 0.6) & (yy < 0.4)
    lin[sky] = np.array([0.1, 0.35, 0.6]) * (0.9 + 0.2 * tex[sky])[:, None]
    return lin.clip(0.02, 1)


def synth_burst(out, frames=8, image=None, size=384, iso_noise=(2e-3, 2e-5), seed=1):
    """
    A burst with known truth: the scene is drawn at twice the raw resolution, moved by a random
    sub-pixel hand shake, integrated over each raw pixel (2x2 box), sampled through an RGGB
    mosaic, given shot and read noise, quantised to 10 bits. Saves truth.npy (raw resolution) and
    truth_hr.npy (twice that, for super-resolution).
    """
    rng = np.random.default_rng(seed)
    n = size * 4                                 # raw is size*2 square, the scene twice that
    if image:
        img = np.asarray(Image.open(image).convert("RGB").resize((n, n)), np.float64) / 255.0
        hr = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    else:
        hr = synth_scene(n)
    os.makedirs(out, exist_ok=True)
    box = lambda x: x.reshape(n // 2, 2, n // 2, 2, 3).mean((1, 3))
    shifts = rng.uniform(-3, 3, (frames, 2))      # raw pixels
    shifts[0] = 0
    black, white = 64, 1023
    a, b = iso_noise
    hot = rng.random((n // 2, n // 2)) < 2e-4                     # isolated hot pixels
    pairs = np.argwhere(rng.random((n // 2, n // 2 - 1)) < 2e-5)  # a few horizontal pairs
    hot[pairs[:, 0], pairs[:, 1]] = hot[pairs[:, 0], pairs[:, 1] + 1] = True
    hot_level = rng.uniform(0.3, 1.0, hot.shape)
    for f in range(frames):
        moved = np.stack([ndimage.shift(hr[..., c], 2 * shifts[f], order=3, mode="nearest") for c in range(3)], -1)
        lin = box(moved).clip(0, 1)
        raw = np.zeros(lin.shape[:2])
        raw[0::2, 0::2] = lin[0::2, 0::2, 0]
        raw[0::2, 1::2] = lin[0::2, 1::2, 1]
        raw[1::2, 0::2] = lin[1::2, 0::2, 1]
        raw[1::2, 1::2] = lin[1::2, 1::2, 2]
        noisy = raw + rng.normal(0, 1, raw.shape) * np.sqrt(a * raw + b)
        noisy = np.where(hot, np.maximum(noisy, hot_level), noisy)
        dn = np.clip(noisy * (white - black) + black, 0, white).round().astype("<u2")
        dn.tofile(os.path.join(out, "frame_%02d.raw16" % f))
    W = H = n // 2
    meta = dict(width=W, height=H, chanAt=[0, 1, 2, 3], black=[black] * 4, white=white, rotation=0, crop=0,
                frames=frames, received=frames, shortFrames=0, shortScale=4, gain=1.0, underEv=0.7, zoom=1,
                iso=342, exposureNs=33333333, gains=[1, 1, 1, 1], ccm=[1, 0, 0, 0, 1, 0, 0, 0, 1])
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    np.save(os.path.join(out, "truth.npy"), box(hr).astype(np.float32))
    np.save(os.path.join(out, "truth_hr.npy"), hr.astype(np.float32))
    np.save(os.path.join(out, "shifts.npy"), shifts)
    np.save(os.path.join(out, "hot.npy"), hot)
    return meta


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 10 * math.log10(1 / max(mse, 1e-12))

# ---------------------------------------------------------------------------
# Commands

def merged(meta, frames, method):
    qs = [to_quad(f, meta) for f in frames]
    pyrs = [pyramid(gray(q)) for q in qs]
    offsets = [np.zeros((math.ceil(qs[0].shape[0] / 16), math.ceil(qs[0].shape[1] / 16), 2))]
    for p in pyrs[1:]:
        offsets.append(align(pyrs[0], p))
    noise = fit_noise(qs[0], warp(qs[1], offsets[1])) if len(qs) > 1 else (1e-4, 1e-6)
    if method == "single" or len(qs) == 1:
        return qs[0], qs, offsets, noise
    if method == "wiener":
        return merge_wiener(qs, offsets, noise), qs, offsets, noise
    return merge_robust(qs, offsets, noise)[0], qs, offsets, noise


def cmd_develop(args):
    global HOT_CROSS
    HOT_CROSS = args.hot == "cross"
    meta, frames = load_burst(args.burst)
    mq, _, _, noise = merged(meta, frames, args.merge)
    opts = dict(compress=args.compress, lift=args.lift, black=args.black, contrast=args.contrast, sat=args.sat)
    img = finish(mq, meta, meta["gains"], meta["ccm"], opts, tone=args.tone)
    Image.fromarray(img).save(args.output, quality=95)
    print("noise a=%.3g b=%.3g -> %s" % (noise[0], noise[1], args.output))
    if args.compare:
        for name in ("hal.jpg", "arco.jpg"):
            p = os.path.join(args.burst, name)
            if os.path.exists(p):
                other = Image.open(p).convert("RGB").resize((img.shape[1], img.shape[0]))
                side = np.concatenate([np.asarray(other), img], axis=1)
                out = os.path.splitext(args.output)[0] + "_vs_" + name
                Image.fromarray(side).save(out, quality=90)
                print("comparison", out)


def cmd_sr(args):
    meta, frames = load_burst(args.burst)
    _, qs, offsets, noise = merged(meta, frames, "single")
    rgb = super_resolve(qs, offsets, noise, meta["chanAt"], args.size)
    rgb = rgb * np.array(meta["gains"])[[0, 1, 3]] @ np.array(meta["ccm"]).reshape(3, 3).T
    Image.fromarray((srgb(rgb * meta.get("gain", 1)) * 255).astype(np.uint8)).save(args.output)
    print("super-resolved crop ->", args.output)


def cmd_synth(args):
    synth_burst(args.out, args.frames, args.image)
    print("synthetic burst ->", args.out)


def cmd_bench(args):
    """Synthetic burst: merge methods against the truth, then the 2x zoom against the app's way."""
    d = args.dir
    synth_burst(d, frames=args.frames)
    meta, frames = load_burst(d)
    truth = np.load(os.path.join(d, "truth.npy"))
    tq = np.stack([truth[0::2, 0::2, 0], truth[0::2, 1::2, 1], truth[1::2, 0::2, 1], truth[1::2, 1::2, 2]], -1)
    m = 24
    hot = np.load(os.path.join(d, "hot.npy"))
    hq = np.stack([hot[0::2, 0::2], hot[0::2, 1::2], hot[1::2, 0::2], hot[1::2, 1::2]], -1)
    r0 = frames[0].astype(np.float32)
    nq = (np.stack([r0[0::2, 0::2], r0[0::2, 1::2], r0[1::2, 0::2], r0[1::2, 1::2]], -1) - meta["black"][0]) / (meta["white"] - meta["black"][0])
    visible = hq & (nq - tq > 0.1)
    for cross in (False, True):
        q = to_quad(frames[0], meta, cross=cross)
        print("hot filter %-12s PSNR %.2f dB (detail kept), visible hot pixels left %d of %d" % (
            "cross-colour" if cross else "app", psnr(q[m:-m, m:-m], tq[m:-m, m:-m]),
            (visible & (q - tq > 0.1)).sum(), visible.sum()))
    global HOT_CROSS
    HOT_CROSS = True     # the app's filter flattens this chart's fine detail and would hide the merge differences
    Q = tq.shape[0]
    yy, xx = np.mgrid[0:Q, 0:Q] / Q
    flat = ~(((xx - 0.3) ** 2 + (yy - 0.3) ** 2 < 0.07) | ((yy > 0.58) & (xx > 0.53)))   # no zone plate, no bars
    flat[:m] = flat[-m:] = False
    flat[:, :m] = flat[:, -m:] = False
    results = {}
    for method in ("single", "robust", "wiener"):
        mq, qs, offsets, noise = merged(meta, frames, method)
        results[method] = mq
        print("merge %-7s PSNR %.2f dB, %.2f dB without the aliasing patterns (noise fit a=%.2g b=%.2g, true %.2g %.2g)" % (
            method, psnr(mq[m:-m, m:-m], tq[m:-m, m:-m]), psnr(mq[flat], tq[flat]), noise[0], noise[1], 2e-3, 2e-5))
    # 2x zoom of a centred crop: robust merge + demosaic + bicubic (as the app) vs super-resolution
    hr = np.load(os.path.join(d, "truth_hr.npy"))
    crop = args.crop
    H = truth.shape[0]
    c0 = H // 2 - crop // 2
    t = hr[2 * c0: 2 * c0 + 2 * crop, 2 * c0: 2 * c0 + 2 * crop]
    rgb = demosaic_malvar(mosaic_from_quad(results["robust"], meta["chanAt"]), meta["chanAt"])
    app = np.stack([ndimage.zoom(rgb[c0:c0 + crop, c0:c0 + crop, c], 2, order=3) for c in range(3)], -1)
    _, qs, offsets, noise = merged(meta, frames, "single")
    sr = super_resolve(qs, offsets, noise, meta["chanAt"], crop)
    k = 16
    print("zoom 2x  merge+bicubic PSNR %.2f dB, super-resolution PSNR %.2f dB" % (
        psnr(np.clip(app, 0, 1)[k:-k, k:-k], t[k:-k, k:-k]), psnr(np.clip(sr, 0, 1)[k:-k, k:-k], t[k:-k, k:-k])))
    side = np.concatenate([app, sr, t], 1)
    Image.fromarray((srgb(side) * 255).astype(np.uint8)).save(os.path.join(d, "zoom_app_sr_truth.png"))
    # tone mapping: both run on the merged burst (judged by eye, not by PSNR)
    for tone in ("app", "laplacian"):
        img = finish(results["robust"], meta, meta["gains"], meta["ccm"], {}, tone=tone)
        Image.fromarray(img).save(os.path.join(d, "tone_%s.jpg" % tone), quality=92)
    print("images in", d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("develop")
    p.add_argument("burst")
    p.add_argument("-o", "--output", default="arco_lab.jpg")
    p.add_argument("--merge", choices=["robust", "wiener", "single"], default="robust")
    p.add_argument("--tone", choices=["app", "laplacian"], default="app")
    p.add_argument("--hot", choices=["app", "cross"], default="app", help="hot pixel filter")
    p.add_argument("--compress", type=float, default=0.72)
    p.add_argument("--lift", type=float, default=0.94)
    p.add_argument("--black", type=float, default=0.012)
    p.add_argument("--contrast", type=float, default=0.35)
    p.add_argument("--sat", type=float, default=1.15)
    p.add_argument("--compare", action="store_true", help="side by side with hal.jpg and arco.jpg")
    p.set_defaults(func=cmd_develop)
    p = sub.add_parser("sr")
    p.add_argument("burst")
    p.add_argument("-o", "--output", default="arco_sr.png")
    p.add_argument("--size", type=int, default=512, help="crop size in raw pixels")
    p.set_defaults(func=cmd_sr)
    p = sub.add_parser("synth")
    p.add_argument("out")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--image")
    p.set_defaults(func=cmd_synth)
    p = sub.add_parser("bench")
    p.add_argument("dir", help="where to write the synthetic burst and the comparison images")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--crop", type=int, default=192, help="zoom test crop, raw pixels")
    p.set_defaults(func=cmd_bench)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
