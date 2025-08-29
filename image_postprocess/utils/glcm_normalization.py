import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import gaussian_filter
from PIL import Image

def glcm_normalize(img_arr: np.ndarray,
                   ref_img_arr: np.ndarray = None,
                   distances: list = [1],
                   angles: list = [0, np.pi/4, np.pi/2, 3*np.pi/4],
                   levels: int = 256,
                   strength: float = 0.9,
                   seed: int = None,
                   max_levels_for_speed: int = None,
                   eps: float = 1e-8):
    """
    Optimized GLCM normalization.

    Key optimizations:
    - quantize grayscale to fewer levels if max_levels_for_speed is provided (speeds up graycomatrix drastically)
    - compute glcm / properties once (not per channel)
    - use gaussian_filter (single multi-dimensional filter) instead of two gaussian_filter1d calls
    - generate noise once for all channels
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    img = np.asarray(img_arr)
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1

    # Grayscale and quantization
    img_gray_f = np.mean(img.astype(np.float32), axis=2) if img.ndim == 3 else img.astype(np.float32)
    img_gray = (img_gray_f / 255.0 * (levels - 1)).astype(np.int32)

    # Optionally reduce levels for speed (safe; only if caller requests)
    use_levels = levels
    if max_levels_for_speed is not None and max_levels_for_speed < levels:
        use_levels = max_levels_for_speed
        # quantize into `use_levels` bins
        img_gray = np.floor(img_gray_f / 255.0 * (use_levels - 1)).astype(np.uint8)
    else:
        img_gray = img_gray.astype(np.uint8)

    # Compute GLCM and properties (image-level)
    glcm = graycomatrix(img_gray, distances=distances, angles=angles,
                        levels=use_levels, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()

    ref_contrast = None
    ref_homogeneity = None
    if ref_img_arr is not None:
        ref = np.asarray(ref_img_arr)
        # Resize reference only once if needed
        if ref.shape[0] != h or ref.shape[1] != w:
            ref_img = Image.fromarray(ref).resize((w, h), resample=Image.BICUBIC)
            ref = np.array(ref_img)
        ref_gray_f = np.mean(ref.astype(np.float32), axis=2) if ref.ndim == 3 else ref.astype(np.float32)
        if max_levels_for_speed is not None and max_levels_for_speed < levels:
            ref_gray = np.floor(ref_gray_f / 255.0 * (use_levels - 1)).astype(np.uint8)
        else:
            ref_gray = (ref_gray_f / 255.0 * (use_levels - 1)).astype(np.uint8)
        ref_glcm = graycomatrix(ref_gray, distances=distances, angles=angles,
                                levels=use_levels, symmetric=True, normed=True)
        ref_contrast = graycoprops(ref_glcm, 'contrast').mean()
        ref_homogeneity = graycoprops(ref_glcm, 'homogeneity').mean()

    out = np.empty_like(img, dtype=np.float32)

    # Pre-generate noise if needed for all channels
    if strength > 0.0:
        noise_all = rng.normal(loc=0.0, scale=0.02 * strength, size=(h, w, channels)).astype(np.float32) * 255.0
    else:
        noise_all = np.zeros((h, w, channels), dtype=np.float32)

    # If reference features exist, precompute global transforms
    if (ref_contrast is not None) and (ref_homogeneity is not None):
        contrast_ratio = ref_contrast / (contrast + eps)
        homogeneity_ratio = ref_homogeneity / (homogeneity + eps)
        # contrast adjustment uses sqrt scaling
        contrast_scale = np.sqrt(contrast_ratio).astype(np.float32)
        # homogeneity: sigma for smoothing - keep within reasonable bounds
        sigma = float(np.clip(1.0 / (homogeneity_ratio + eps), 0.5, 5.0))
    else:
        contrast_scale = None
        sigma = None

    # Apply per-channel transforms (vectorized where possible)
    if channels == 1:
        channel = img.astype(np.float32)
        if contrast_scale is not None:
            adjusted = channel * contrast_scale
            # single multi-dimensional gaussian instead of two 1D passes
            adjusted = gaussian_filter(adjusted, sigma=(sigma, sigma))
            blended = (1.0 - strength) * channel + strength * adjusted
        else:
            blended = channel
        blended += noise_all[:, :, 0]
        out[:, :] = blended
    else:
        for c in range(channels):
            channel = img[:, :, c].astype(np.float32)
            if contrast_scale is not None:
                adjusted = channel * contrast_scale
                adjusted = gaussian_filter(adjusted, sigma=(sigma, sigma))
                blended = (1.0 - strength) * channel + strength * adjusted
            else:
                blended = channel
            blended += noise_all[:, :, c]
            out[:, :, c] = blended

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out
