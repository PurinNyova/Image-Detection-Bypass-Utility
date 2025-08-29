import numpy as np
from skimage.feature import local_binary_pattern
from PIL import Image

def lbp_normalize(img_arr: np.ndarray,
                  ref_img_arr: np.ndarray = None,
                  radius: int = 3,
                  n_points: int = 24,
                  method: str = 'uniform',
                  strength: float = 0.9,
                  seed: int = None,
                  eps: float = 1e-8):
    """
    Optimized LBP histogram normalization.

    Key optimizations:
    - compute LBP and its histogram once (not per channel)
    - use np.bincount (faster) instead of np.histogram for integer LBP
    - compute mapping & per-bin operations once, then apply vectorized indexing
    - generate noise once for all channels
    - fewer temporaries, consistent dtypes
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    img = np.asarray(img_arr)
    h, w = img.shape[:2]
    n_bins = n_points + 2 if method == 'uniform' else 2 ** n_points

    # Grayscale conversion (float32)
    img_gray = np.mean(img.astype(np.float32), axis=2) if img.ndim == 3 else img.astype(np.float32)

    # Compute LBP for input (float or int result)
    lbp_img = local_binary_pattern(img_gray, n_points, radius, method=method)
    # Convert LBP to integer indices for bincount (safe cast)
    lbp_int = np.rint(lbp_img).astype(np.int32)
    # Use bincount for integer labels which is faster than histogram
    lbp_counts = np.bincount(lbp_int.ravel(), minlength=n_bins).astype(np.float64)
    lbp_hist = lbp_counts / (lbp_counts.sum() + eps)

    ref_lbp_hist = None
    if ref_img_arr is not None:
        ref = np.asarray(ref_img_arr)
        # Resize reference only once if needed
        if ref.shape[0] != h or ref.shape[1] != w:
            ref_img = Image.fromarray(ref).resize((w, h), resample=Image.BICUBIC)
            ref = np.array(ref_img)
        ref_gray = np.mean(ref.astype(np.float32), axis=2) if ref.ndim == 3 else ref.astype(np.float32)
        ref_lbp = local_binary_pattern(ref_gray, n_points, radius, method=method)
        ref_int = np.rint(ref_lbp).astype(np.int32)
        ref_counts = np.bincount(ref_int.ravel(), minlength=n_bins).astype(np.float64)
        ref_lbp_hist = ref_counts / (ref_counts.sum() + eps)

    out = np.empty_like(img, dtype=np.float32)
    channels = img.shape[2] if img.ndim == 3 else 1

    # Precompute mapping and scale-image-level arrays only once
    if ref_lbp_hist is not None:
        cdf_img = np.cumsum(lbp_hist)
        cdf_ref = np.cumsum(ref_lbp_hist)
        # Vectorized mapping: for each possible lbp bin value find target bin
        mapping = np.searchsorted(cdf_ref, cdf_img, side='left')
        mapping = np.clip(mapping, 0, n_bins - 1).astype(np.float32)
        # mapping_per_pixel (h,w)
        mapping_per_pixel = mapping[lbp_int]
        # denom per pixel (avoid divide by zero)
        denom = (lbp_int.astype(np.float32) + eps)
        # precompute scale per pixel
        scale_per_pixel = mapping_per_pixel / denom
    else:
        # Unused but create placeholders to keep code simpler
        scale_per_pixel = None

    # Prepare noise for all channels at once (if needed)
    if strength > 0.0:
        noise_all = rng.normal(loc=0.0, scale=0.02 * strength, size=(h, w, channels)).astype(np.float32) * 255.0
    else:
        noise_all = np.zeros((h, w, channels), dtype=np.float32)

    # Process channels: mostly vectorized
    if channels == 1:
        channel = img.astype(np.float32)
        if scale_per_pixel is not None:
            adjusted = channel * scale_per_pixel
            blended = (1.0 - strength) * channel + strength * adjusted
        else:
            blended = channel
        blended += noise_all[:, :, 0]
        out[:, :] = blended
    else:
        for c in range(channels):
            channel = img[:, :, c].astype(np.float32)
            if scale_per_pixel is not None:
                # vectorized multiply
                adjusted = channel * scale_per_pixel
                blended = (1.0 - strength) * channel + strength * adjusted
            else:
                blended = channel
            blended += noise_all[:, :, c]
            out[:, :, c] = blended

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out
