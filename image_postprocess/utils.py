"""
utils.py

Helper functions for image postprocessing, including EXIF removal, noise addition,
color correction, and Fourier spectrum matching.
"""

from PIL import Image, ImageOps
import numpy as np
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False
from scipy.ndimage import gaussian_filter1d

def remove_exif_pil(img: Image.Image) -> Image.Image:
    data = img.tobytes()
    new = Image.frombytes(img.mode, img.size, data)
    return new

def add_gaussian_noise(img_arr: np.ndarray, std_frac=0.02, seed=None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    std = std_frac * 255.0
    noise = np.random.normal(loc=0.0, scale=std, size=img_arr.shape)
    out = img_arr.astype(np.float32) + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def clahe_color_correction(img_arr: np.ndarray, clip_limit=2.0, tile_grid_size=(8,8)) -> np.ndarray:
    if _HAS_CV2:
        lab = cv2.cvtColor(img_arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        return out
    else:
        pil = Image.fromarray(img_arr)
        channels = pil.split()
        new_ch = []
        for ch in channels:
            eq = ImageOps.equalize(ch)
            new_ch.append(eq)
        merged = Image.merge('RGB', new_ch)
        return np.array(merged)

def randomized_perturbation(img_arr: np.ndarray, magnitude_frac=0.008, seed=None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    mag = magnitude_frac * 255.0
    perturb = np.random.uniform(low=-mag, high=mag, size=img_arr.shape)
    out = img_arr.astype(np.float32) + perturb
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def radial_profile(mag: np.ndarray, center=None, nbins=None):
    h, w = mag.shape
    if center is None:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = center

    if nbins is None:
        nbins = int(max(h, w) / 2)
    nbins = max(1, int(nbins))

    y = np.arange(h) - cy
    x = np.arange(w) - cx
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X * X + Y * Y)

    Rmax = R.max()
    if Rmax <= 0:
        Rnorm = R
    else:
        Rnorm = R / (Rmax + 1e-12)
        Rnorm = np.minimum(Rnorm, 1.0 - 1e-12)

    bin_edges = np.linspace(0.0, 1.0, nbins + 1)
    bin_idx = np.digitize(Rnorm.ravel(), bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)

    sums = np.bincount(bin_idx, weights=mag.ravel(), minlength=nbins)
    counts = np.bincount(bin_idx, minlength=nbins)

    radial_mean = np.zeros(nbins, dtype=np.float64)
    nonzero = counts > 0
    radial_mean[nonzero] = sums[nonzero] / counts[nonzero]

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, radial_mean

def fourier_match_spectrum(img_arr: np.ndarray,
                           ref_img_arr: np.ndarray = None,
                           mode='auto',
                           alpha=1.0,
                           cutoff=0.25,
                           strength=0.9,
                           randomness=0.05,
                           phase_perturb=0.08,
                           radial_smooth=5,
                           seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    h, w = img_arr.shape[:2]
    cy, cx = h // 2, w // 2
    nbins = max(8, int(max(h, w) / 2))

    if mode == 'auto':
        mode = 'ref' if ref_img_arr is not None else 'model'

    bin_centers_src = np.linspace(0.0, 1.0, nbins)

    model_radial = None
    if mode == 'model':
        eps = 1e-8
        model_radial = (1.0 / (bin_centers_src + eps)) ** (alpha / 2.0)
        lf = max(1, nbins // 8)
        model_radial = model_radial / (np.median(model_radial[:lf]) + 1e-12)
        model_radial = gaussian_filter1d(model_radial, sigma=max(1, radial_smooth))

    ref_radial = None
    ref_bin_centers = None
    if mode == 'ref' and ref_img_arr is not None:
        if ref_img_arr.shape[0] != h or ref_img_arr.shape[1] != w:
            ref_img = Image.fromarray(ref_img_arr).resize((w, h), resample=Image.BICUBIC)
            ref_img_arr = np.array(ref_img)
        ref_gray = np.mean(ref_img_arr.astype(np.float32), axis=2) if ref_img_arr.ndim == 3 else ref_img_arr.astype(np.float32)
        Fref = np.fft.fftshift(np.fft.fft2(ref_gray))
        Mref = np.abs(Fref)
        ref_bin_centers, ref_radial = radial_profile(Mref, center=(h // 2, w // 2), nbins=nbins)
        ref_radial = gaussian_filter1d(ref_radial, sigma=max(1, radial_smooth))

    out = np.zeros_like(img_arr, dtype=np.float32)

    y = np.linspace(-1, 1, h, endpoint=False)[:, None]
    x = np.linspace(-1, 1, w, endpoint=False)[None, :]
    r = np.sqrt(x * x + y * y)
    r = np.clip(r, 0.0, 1.0 - 1e-6)

    for c in range(img_arr.shape[2]):
        channel = img_arr[:, :, c].astype(np.float32)
        F = np.fft.fft2(channel)
        Fshift = np.fft.fftshift(F)
        mag = np.abs(Fshift)
        phase = np.angle(Fshift)

        bin_centers_src_calc, src_radial = radial_profile(mag, center=(h // 2, w // 2), nbins=nbins)
        src_radial = gaussian_filter1d(src_radial, sigma=max(1, radial_smooth))
        bin_centers_src = bin_centers_src_calc

        if mode == 'ref' and ref_radial is not None:
            ref_interp = np.interp(bin_centers_src, ref_bin_centers, ref_radial)
            eps = 1e-8
            ratio = (ref_interp + eps) / (src_radial + eps)
            desired_radial = src_radial * ratio
        elif mode == 'model' and model_radial is not None:
            lf = max(1, nbins // 8)
            scale = (np.median(src_radial[:lf]) + 1e-12) / (np.median(model_radial[:lf]) + 1e-12)
            desired_radial = model_radial * scale
        else:
            desired_radial = src_radial.copy()

        eps = 1e-8
        multiplier_1d = (desired_radial + eps) / (src_radial + eps)
        multiplier_1d = np.clip(multiplier_1d, 0.2, 5.0)
        mult_2d = np.interp(r.ravel(), bin_centers_src, multiplier_1d).reshape(h, w)

        edge = 0.05 + 0.02 * (1.0 - cutoff) if 'cutoff' in globals() else 0.05
        edge = max(edge, 1e-6)
        weight = np.where(r <= 0.25, 1.0,
                         np.where(r <= 0.25 + edge,
                                  0.5 * (1 + np.cos(np.pi * (r - 0.25) / edge)),
                                  0.0))

        final_multiplier = 1.0 + (mult_2d - 1.0) * (weight * strength)

        if randomness and randomness > 0.0:
            noise = rng.normal(loc=1.0, scale=randomness, size=final_multiplier.shape)
            final_multiplier *= (1.0 + (noise - 1.0) * weight)

        mag2 = mag * final_multiplier

        if phase_perturb and phase_perturb > 0.0:
            phase_sigma = phase_perturb * np.clip((r - 0.25) / (1.0 - 0.25 + 1e-6), 0.0, 1.0)
            phase_noise = rng.standard_normal(size=phase_sigma.shape) * phase_sigma
            phase2 = phase + phase_noise
        else:
            phase2 = phase

        Fshift2 = mag2 * np.exp(1j * phase2)
        F_ishift = np.fft.ifftshift(Fshift2)
        img_back = np.fft.ifft2(F_ishift)
        img_back = np.real(img_back)

        blended = (1.0 - strength) * channel + strength * img_back
        out[:, :, c] = blended

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def auto_white_balance_ref(img_arr: np.ndarray, ref_img_arr: np.ndarray = None) -> np.ndarray:
    """
    Auto white-balance correction using a reference image.
    If ref_img_arr is None, uses a gray-world assumption instead.
    """
    img = img_arr.astype(np.float32)

    if ref_img_arr is not None:
        ref = ref_img_arr.astype(np.float32)
        ref_mean = ref.reshape(-1, 3).mean(axis=0)
    else:
        # Gray-world assumption: target is neutral gray
        ref_mean = np.array([128.0, 128.0, 128.0], dtype=np.float32)

    img_mean = img.reshape(-1, 3).mean(axis=0)

    # Avoid divide-by-zero
    eps = 1e-6
    scale = (ref_mean + eps) / (img_mean + eps)

    corrected = img * scale
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    return corrected