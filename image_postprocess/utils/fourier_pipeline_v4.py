import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.fft import next_fast_len
from scipy.stats import binned_statistic


GAIN_LIMIT = 0.5
MULT_MIN = 0.5
MULT_MAX = 2.0
PEAK_THRESHOLD = 2.0
PEAK_MIN_MULT = 0.1
RESIDUAL_CAP = 10.0


def _validated_float(value, name, minimum=None, maximum=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and v < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and v > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return v


def _to_gray(img_arr):
    if img_arr.ndim == 3:
        return np.mean(img_arr.astype(np.float64), axis=2)
    if img_arr.ndim == 2:
        return img_arr.astype(np.float64)
    raise ValueError("reference image must be 2D or 3D")


def _validate_real_numeric(arr, name):
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError(f"{name} must have a real numeric dtype, got {arr.dtype}")


def _sanitize(arr):
    return np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=255.0, neginf=0.0)


def _radial_log_profile(power, rn, nbins, sigma):
    log_power = np.log10(power + 1e-12)
    centers = (np.arange(nbins) + 0.5) / nbins
    stat, _, _ = binned_statistic(
        rn.ravel(), log_power.ravel(), statistic='median', bins=nbins, range=(0.0, 1.0)
    )
    valid = np.isfinite(stat)
    if not np.any(valid):
        return np.zeros(nbins, dtype=np.float64)
    profile = np.interp(centers, centers[valid], stat[valid])
    if sigma >= 1:
        profile = gaussian_filter1d(profile, sigma=sigma, mode='nearest')
    return profile


def fourier_match_spectrum_v4(img_arr: np.ndarray,
                              ref_img_arr: np.ndarray = None,
                              mode='auto',
                              alpha=1.0,
                              cutoff=0.25,
                              strength=0.9,
                              randomness=0.05,
                              radial_smooth=5,
                              seed=None):
    if not isinstance(img_arr, np.ndarray) or img_arr.ndim not in (2, 3):
        raise ValueError("img_arr must be a 2D grayscale or 3D RGB numpy array")
    if img_arr.ndim == 3 and img_arr.shape[2] not in (1, 3, 4):
        raise ValueError("img_arr must have 1, 3, or 4 channels")
    if img_arr.size == 0:
        raise ValueError("img_arr must not be empty")
    _validate_real_numeric(img_arr, "img_arr")
    alpha = _validated_float(alpha, 'alpha', 0.0, 4.0)
    cutoff = _validated_float(cutoff, 'cutoff', 0.01, 1.0)
    strength = _validated_float(strength, 'strength', 0.0, 1.0)
    randomness = _validated_float(randomness, 'randomness', 0.0, 1.0)
    try:
        radial_smooth = int(radial_smooth)
    except (TypeError, ValueError):
        raise ValueError("radial_smooth must be an integer")
    if radial_smooth < 0:
        raise ValueError("radial_smooth must be >= 0")
    if mode not in ('auto', 'ref', 'model'):
        raise ValueError("mode must be one of 'auto', 'ref', 'model'")
    if mode == 'ref' and ref_img_arr is None:
        raise ValueError("mode 'ref' requires ref_img_arr")
    if seed is not None and not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer or None")
    if strength <= 0.0:
        return img_arr.copy()
    rng = np.random.default_rng(seed)

    if mode == 'auto':
        mode = 'ref' if ref_img_arr is not None else 'model'

    if img_arr.ndim == 2:
        channels = [_sanitize(img_arr)]
    else:
        channels = [_sanitize(img_arr[:, :, c]) for c in range(img_arr.shape[2])]

    h, w = channels[0].shape

    if mode == 'ref':
        if not isinstance(ref_img_arr, np.ndarray):
            raise ValueError("ref_img_arr must be a numpy array")
        if ref_img_arr.ndim not in (2, 3):
            raise ValueError("ref_img_arr must be a 2D or 3D numpy array")
        if ref_img_arr.ndim == 3 and ref_img_arr.shape[2] not in (1, 3, 4):
            raise ValueError("ref_img_arr must have 1, 3, or 4 channels")
        if ref_img_arr.size == 0:
            raise ValueError("ref_img_arr must not be empty")
        _validate_real_numeric(ref_img_arr, "ref_img_arr")
        ref_gray = _to_gray(ref_img_arr)
        if ref_gray.shape != (h, w):
            ref_pil = Image.fromarray(np.clip(ref_gray, 0.0, 255.0).astype(np.uint8))
            ref_gray = np.array(ref_pil.resize((w, h), resample=Image.BILINEAR)).astype(np.float64)

    top = h - 1 if h >= 2 else 0
    left = w - 1 if w >= 2 else 0
    bottom = next_fast_len(2 * h - 1) - top - h
    right = next_fast_len(2 * w - 1) - left - w
    pad_h = top + h + bottom
    pad_w = left + w + right
    fy = np.abs(np.fft.fftfreq(pad_h))[:, None] * 2.0
    fx = np.fft.rfftfreq(pad_w)[None, :] * 2.0
    r = np.sqrt(fx * fx + fy * fy)
    r_max = r.max()
    rn = r / r_max if r_max > 0 else np.zeros_like(r)
    nbins = int(min(512, max(16, max(h, w) // 2)))
    centers = (np.arange(nbins) + 0.5) / nbins
    effective = min(cutoff, 1.0)
    edge = max(min(0.05 + 0.02 * (1.0 - effective), 1.0 - effective), 1e-6)
    weight = np.where(
        rn <= effective,
        0.0,
        np.where(rn >= effective + edge, 1.0, 0.5 * (1.0 - np.cos(np.pi * (rn - effective) / edge))),
    )

    low_bins = centers <= effective
    pad_spec = ((top, bottom), (left, right))

    ref_log_base = None
    if mode == 'ref':
        ref_padded = np.pad(ref_gray, pad_spec, mode='reflect')
        Fref = np.fft.rfft2(ref_padded)
        ref_log_base = _radial_log_profile(np.abs(Fref) ** 2, rn, nbins, radial_smooth)

    out_channels = []
    noise = None
    if randomness > 0.0:
        noise = rng.uniform(-1.0, 1.0, size=(pad_h, pad_w // 2 + 1)) * strength
    for ch in channels:
        padded = np.pad(ch, pad_spec, mode='reflect')
        F = np.fft.rfft2(padded)
        power = np.abs(F) ** 2
        src_log = _radial_log_profile(power, rn, nbins, radial_smooth)

        if mode == 'ref':
            tgt_log = ref_log_base
        else:
            tgt_log = -alpha * np.log10(np.maximum(centers, 1.0 / nbins))

        if np.any(low_bins):
            offset = np.median(src_log[low_bins]) - np.median(tgt_log[low_bins])
        else:
            offset = src_log[0] - tgt_log[0]
        tgt_log = tgt_log + offset
        if mode == 'model':
            tgt_log = np.clip(tgt_log, src_log.min(), src_log.max())

        gain_log = np.clip(tgt_log - src_log, -GAIN_LIMIT, GAIN_LIMIT)
        gain_2d = np.interp(rn, centers, gain_log)
        mult = np.clip(10.0 ** gain_2d, MULT_MIN, MULT_MAX)

        log_power = np.log10(power + 1e-12)
        src_log_2d = np.interp(rn, centers, src_log)
        peaks = log_power > src_log_2d + PEAK_THRESHOLD
        if np.any(peaks):
            baseline_mult = np.clip(10.0 ** (src_log_2d - log_power), PEAK_MIN_MULT, 1.0)
            mult = np.where(peaks, baseline_mult, mult)

        if h > 1 and w > 1:
            gy, gx = np.gradient(ch)
            activity = gaussian_filter(np.hypot(gx, gy), sigma=1.0, mode='reflect')
            act = np.clip(activity / (np.percentile(activity, 75) + 1e-6), 0.0, 1.0)
        else:
            act = np.zeros((h, w), dtype=np.float64)

        final_mult = 1.0 + (mult - 1.0) * (weight * strength)
        if noise is not None:
            final_mult = final_mult * (1.0 + randomness * noise * weight)

        back = np.fft.irfft2(F * final_mult, s=(pad_h, pad_w))[top:top + h, left:left + w]
        back = np.nan_to_num(back, nan=0.0, posinf=255.0, neginf=0.0)
        delta = (back - ch) * act
        back = ch + np.clip(delta, -RESIDUAL_CAP, RESIDUAL_CAP)
        out_channels.append(np.round(np.clip(back, 0.0, 255.0)).astype(np.uint8))

    if img_arr.ndim == 2:
        return out_channels[0]
    return np.stack(out_channels, axis=2)
