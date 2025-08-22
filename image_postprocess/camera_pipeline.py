"""
camera_pipeline.py

Functions for simulating a realistic camera pipeline, including Bayer mosaic/demosaic,
chromatic aberration, vignette, sensor noise, hot pixels, banding, motion blur, and JPEG recompression.
"""

from io import BytesIO
from PIL import Image
import numpy as np
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False
from scipy.ndimage import convolve

def _bayer_mosaic(img: np.ndarray, pattern='RGGB') -> np.ndarray:
    """Create a single-channel Bayer mosaic from an RGB image.

    pattern currently supports 'RGGB' (most common). Returns uint8 2D array.
    """
    h, w = img.shape[:2]
    mosaic = np.zeros((h, w), dtype=np.uint8)

    # pattern mapping for RGGB:
    # (0,0) R, (0,1) G
    # (1,0) G, (1,1) B
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # fill mosaic according to RGGB
    mosaic[0::2, 0::2] = R[0::2, 0::2]
    mosaic[0::2, 1::2] = G[0::2, 1::2]
    mosaic[1::2, 0::2] = G[1::2, 0::2]
    mosaic[1::2, 1::2] = B[1::2, 1::2]
    return mosaic

def _demosaic_bilinear(mosaic: np.ndarray) -> np.ndarray:
    """Simple bilinear demosaic fallback (no cv2). Outputs RGB uint8 image.

    Not perfect but good enough to add demosaic artifacts.
    """
    h, w = mosaic.shape
    # Work in float to avoid overflow
    m = mosaic.astype(np.float32)

    # We'll compute each channel by averaging available mosaic samples
    R = np.zeros_like(m)
    G = np.zeros_like(m)
    B = np.zeros_like(m)

    # RGGB pattern
    R[0::2, 0::2] = m[0::2, 0::2]
    G[0::2, 1::2] = m[0::2, 1::2]
    G[1::2, 0::2] = m[1::2, 0::2]
    B[1::2, 1::2] = m[1::2, 1::2]

    # Convolution kernels for interpolation (simple)
    k_cross = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float32) / 8.0
    k_diag = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.float32) / 4.0

    # convolve using scipy.ndimage.convolve
    R_interp = convolve(R, k_cross, mode='mirror')
    G_interp = convolve(G, k_cross, mode='mirror')
    B_interp = convolve(B, k_cross, mode='mirror')

    out = np.stack((R_interp, G_interp, B_interp), axis=2)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def _apply_chromatic_aberration(img: np.ndarray, strength=1.0, seed=None):
    """Shift R and B channels slightly in opposite directions to emulate CA.

    strength is in pixels (float). Uses cv2.warpAffine if available; integer
    fallback uses np.roll.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    h, w = img.shape[:2]
    max_shift = max(1.0, strength)
    # small random subpixel shift sampled from normal distribution
    shift_r = rng.normal(loc=0.0, scale=max_shift * 0.6)
    shift_b = rng.normal(loc=0.0, scale=max_shift * 0.6)
    # apply opposite horizontal shifts to R and B for lateral CA
    r_x = shift_r
    r_y = rng.normal(scale=0.3 * abs(shift_r))
    b_x = -shift_b
    b_y = rng.normal(scale=0.3 * abs(shift_b))

    out = img.copy().astype(np.float32)
    if _HAS_CV2:
        def warp_channel(ch, tx, ty):
            M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
            return cv2.warpAffine(ch, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out[:, :, 0] = warp_channel(out[:, :, 0], r_x, r_y)
        out[:, :, 2] = warp_channel(out[:, :, 2], b_x, b_y)
    else:
        # integer fallback
        ix_r = int(round(r_x))
        iy_r = int(round(r_y))
        ix_b = int(round(b_x))
        iy_b = int(round(b_y))
        out[:, :, 0] = np.roll(out[:, :, 0], shift=(iy_r, ix_r), axis=(0, 1))
        out[:, :, 2] = np.roll(out[:, :, 2], shift=(iy_b, ix_b), axis=(0, 1))

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def _apply_vignette(img: np.ndarray, strength=0.4):
    h, w = img.shape[:2]
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]
    r = np.sqrt(x * x + y * y)
    mask = 1.0 - (r ** 2) * strength
    mask = np.clip(mask, 0.0, 1.0)
    out = (img.astype(np.float32) * mask[:, :, None])
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def _add_poisson_gaussian_noise(img: np.ndarray, iso_scale=1.0, read_noise_std=2.0, seed=None):
    """Poisson-Gaussian sensor noise model.

    iso_scale scales the signal before Poisson sampling (higher -> more Poisson),
    read_noise_std is the sigma (in DN) of additive Gaussian read noise.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    img_f = img.astype(np.float32)
    # scale to simulate exposure/iso
    scaled = img_f * iso_scale
    # Poisson: we need integer counts; scale to a reasonable photon budget
    # choose scale so that typical pixel values map to ~[0..2000] photons
    photon_scale = 4.0
    lam = np.clip(scaled * photon_scale, 0, 1e6)
    noisy = rng.poisson(lam).astype(np.float32) / photon_scale
    # add read noise
    noisy += rng.normal(loc=0.0, scale=read_noise_std, size=noisy.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def _add_hot_pixels_and_banding(img: np.ndarray, hot_pixel_prob=1e-6, banding_strength=0.0, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)
    # hot pixels
    n_pixels = int(h * w * hot_pixel_prob)
    if n_pixels > 0:
        ys = rng.integers(0, h, size=n_pixels)
        xs = rng.integers(0, w, size=n_pixels)
        vals = rng.integers(200, 256, size=n_pixels)
        for y, x, v in zip(ys, xs, vals):
            out[y, x, :] = v
    # banding: add low-amplitude sinusoidal horizontal banding
    if banding_strength > 0.0:
        rows = np.arange(h)[:, None]
        band = (np.sin(rows * 0.5) * 255.0 * banding_strength)
        out += band[:, :, None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def _motion_blur(img: np.ndarray, kernel_size=5):
    if kernel_size <= 1:
        return img
    # simple linear motion kernel horizontally
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    out = np.zeros_like(img)
    for c in range(3):
        out[:, :, c] = convolve(img[:, :, c].astype(np.float32), kernel, mode='mirror')
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def _jpeg_recompress(img: np.ndarray, quality=90) -> np.ndarray:
    pil = Image.fromarray(img)
    buf = BytesIO()
    pil.save(buf, format='JPEG', quality=int(quality), optimize=False)
    buf.seek(0)
    rec = Image.open(buf).convert('RGB')
    return np.array(rec)

def simulate_camera_pipeline(img_arr: np.ndarray,
                             bayer=True,
                             jpeg_cycles=1,
                             jpeg_quality_range=(88, 96),
                             vignette_strength=0.35,
                             chroma_aberr_strength=1.2,
                             iso_scale=1.0,
                             read_noise_std=2.0,
                             hot_pixel_prob=1e-6,
                             banding_strength=0.0,
                             motion_blur_kernel=1,
                             seed=None):
    """Apply a set of realistic camera/capture artifacts to img_arr (RGB uint8).

    Returns an RGB uint8 image.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    out = img_arr.copy()

    # 1) Bayer mosaic + demosaic (if enabled)
    if bayer:
        try:
            mosaic = _bayer_mosaic(out[:, :, ::-1])  # we built mosaic assuming R,G,B order; send RGB
            if _HAS_CV2:
                # cv2 expects a single-channel Bayer and provides demosaicing codes
                # We'll use RGGB code (COLOR_BAYER_RG2BGR) so convert back to RGB after
                dem = cv2.demosaicing(mosaic, cv2.COLOR_BAYER_RG2BGR)
                # cv2 returns BGR
                dem = dem[:, :, ::-1]
                out = dem
            else:
                out = _demosaic_bilinear(mosaic)
        except Exception:
            # if anything fails, keep original
            out = img_arr.copy()

    # 2) chromatic aberration
    out = _apply_chromatic_aberration(out, strength=chroma_aberr_strength, seed=seed)

    # 3) vignette
    out = _apply_vignette(out, strength=vignette_strength)

    # 4) noise (Poisson-Gaussian)
    out = _add_poisson_gaussian_noise(out, iso_scale=iso_scale, read_noise_std=read_noise_std, seed=seed)

    # 5) hot pixels and banding
    out = _add_hot_pixels_and_banding(out, hot_pixel_prob=hot_pixel_prob, banding_strength=banding_strength, seed=seed)

    # 6) motion blur
    if motion_blur_kernel and motion_blur_kernel > 1:
        out = _motion_blur(out, kernel_size=motion_blur_kernel)

    # 7) JPEG recompression cycles
    for i in range(max(1, int(jpeg_cycles))):
        q = int(rng.integers(jpeg_quality_range[0], jpeg_quality_range[1] + 1))
        out = _jpeg_recompress(out, quality=q)

    return out