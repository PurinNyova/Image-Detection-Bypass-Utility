import numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image

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

    # Ensure image well defined
    if img_arr.ndim == 2:
        h, w = img_arr.shape
        nch = 1
    elif img_arr.ndim == 3:
        h, w, nch = img_arr.shape
    else:
        raise ValueError("img_arr must be 2D or 3D")

    nbins = max(8, int(max(h, w) / 2))

    # Determine mode if auto: use 'ref' if reference image is provided, else 'model'
    if mode == 'auto':
        mode = 'ref' if ref_img_arr is not None else 'model'

    # Create a coordinate grid for building a 2D radial map
    y = np.linspace(-1, 1, h, endpoint=False)[:, None]
    x = np.linspace(-1, 1, w, endpoint=False)[None, :]
    r = np.sqrt(x * x + y * y)
    r = np.clip(r, 0.0, 1.0 - 1e-6)

    # Compute luminance (or gray) from img_arr once.
    if nch == 1:
        src_gray = img_arr.astype(np.float32)
    else:
        # Using simple average; optionally use luma weights (0.2126,0.7152,0.0722)
        src_gray = np.mean(img_arr.astype(np.float32), axis=2)

    # FFT of the source luminance & compute radial profile
    Fsrc = np.fft.fftshift(np.fft.fft2(src_gray))
    Msrc = np.abs(Fsrc)
    bin_centers_src, src_radial = radial_profile(Msrc, center=(h//2, w//2), nbins=nbins)
    src_radial = gaussian_filter1d(src_radial, sigma=max(1, radial_smooth))

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
        # Resize ref image if needed
        if ref_img_arr.shape[0] != h or ref_img_arr.shape[1] != w:
            ref_img = Image.fromarray(ref_img_arr).resize((w, h), resample=Image.BICUBIC)
            ref_img_arr = np.array(ref_img)
        # Convert ref image to grayscale
        if ref_img_arr.ndim == 3:
            ref_gray = np.mean(ref_img_arr.astype(np.float32), axis=2)
        else:
            ref_gray = ref_img_arr.astype(np.float32)
        Fref = np.fft.fftshift(np.fft.fft2(ref_gray))
        Mref = np.abs(Fref)
        ref_bin_centers, ref_radial = radial_profile(Mref, center=(h//2, w//2), nbins=nbins)
        ref_radial = gaussian_filter1d(ref_radial, sigma=max(1, radial_smooth))

    # Compute desired radial profile based on mode
    eps = 1e-8
    if mode == 'ref' and ref_radial is not None:
        ref_interp = np.interp(bin_centers_src, ref_bin_centers, ref_radial)
        lf = max(1, nbins // 8)
        scale = (np.median(src_radial[:lf]) + eps) / (np.median(ref_interp[:lf]) + eps)
        ref_interp *= scale
        desired_radial = ref_interp.copy()
    elif mode == 'model' and model_radial is not None:
        lf = max(1, nbins // 8)
        scale = (np.median(src_radial[:lf]) + eps) / (np.median(model_radial[:lf]) + eps)
        desired_radial = model_radial * scale
    else:
        desired_radial = src_radial.copy()

    # Compute 1D multiplier and clip
    eps = 1e-8
    # adjust clip range and re-introduce strength into multiplier
    multiplier_1d = (desired_radial + eps) / (src_radial + eps)
    multiplier_1d = np.clip(multiplier_1d, 0.05, 10.0)   # wider range -> stronger effect

    # Build the 2D multiplier map (weight remains computed as before)
    edge = 0.05 + 0.02 * (1.0 - cutoff)
    edge = max(edge, 1e-6)
    weight = np.where(r <= cutoff, 1.0,
                      np.where(r <= cutoff + edge,
                               0.5 * (1 + np.cos(np.pi * (r - cutoff) / edge)),
                               0.0))
    mult_2d = np.interp(r.ravel(), bin_centers_src, multiplier_1d).reshape(h, w)
    # include strength in multiplier application (stronger spectral change)
    final_multiplier = 1.0 + (mult_2d - 1.0) * (weight * strength)

    # optional randomness (kept weighted)
    if randomness and randomness > 0.0:
        noise = rng.normal(loc=1.0, scale=randomness, size=final_multiplier.shape)
        final_multiplier *= (1.0 + (noise - 1.0) * weight)

    # Prepare output buffer.
    if nch == 1:
        out = np.zeros((h, w), dtype=np.uint8)
    else:
        out = np.zeros((h, w, nch), dtype=np.uint8)

    # Process each channel (for grayscale, loop once)
    for c in range(nch):
        if nch == 1:
            channel = img_arr.astype(np.float32)
        else:
            channel = img_arr[:, :, c].astype(np.float32)
    
        F = np.fft.fft2(channel)
        Fshift = np.fft.fftshift(F)
        mag = np.abs(Fshift)
        phase = np.angle(Fshift)

        # Apply final multiplier computed from luminance.
        mag2 = mag * final_multiplier

        # Apply phase perturbation using cutoff instead of hard-coded value.
        if phase_perturb and phase_perturb > 0.0:
            phase_sigma = phase_perturb * np.clip((r - cutoff) / (1.0 - cutoff + 1e-6), 0.0, 1.0)
            phase_noise = rng.standard_normal(size=phase_sigma.shape) * phase_sigma
            phase2 = phase + phase_noise
        else:
            phase2 = phase

        Fshift2 = mag2 * np.exp(1j * phase2)
        F_ishift = np.fft.ifftshift(Fshift2)
        img_back = np.fft.ifft2(F_ishift)
        img_back = np.real(img_back)

        # Blend lightly (so you still can dial strength)
        blended = (1.0 - min(0.5, 1.0 - strength)) * channel + min(1.0, strength + 0.2) * img_back

        if nch == 1:
            out[:, :] = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            out[:, :, c] = np.clip(blended, 0, 255).astype(np.uint8)

    return out