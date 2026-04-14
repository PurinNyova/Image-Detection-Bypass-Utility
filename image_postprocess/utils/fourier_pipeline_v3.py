import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image


def fourier_match_spectrum_v3(img_arr: np.ndarray,
                              ref_img_arr: np.ndarray = None,
                              mode='auto',
                              alpha=1.0,
                              cutoff=0.25,
                              strength=0.9,
                              randomness=0.05,
                              radial_smooth=7,
                              seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if img_arr.ndim == 2:
        is_gray = True
        h, w = img_arr.shape
    elif img_arr.ndim == 3:
        is_gray = False
        h, w, nch = img_arr.shape
    else:
        raise ValueError("img_arr must be 2D or 3D")

    # --- Color Space Conversion: work on L channel only ---
    if is_gray:
        L_float = img_arr.astype(np.float32)
        A_chan = None
        B_chan = None
    else:
        lab = cv2.cvtColor(img_arr, cv2.COLOR_RGB2LAB)
        L_float = lab[:, :, 0].astype(np.float32)
        A_chan = lab[:, :, 1]
        B_chan = lab[:, :, 2]

    # Determine mode
    if mode == 'auto':
        mode = 'ref' if ref_img_arr is not None else 'model'

    # --- Coordinate grid for frequency mask ---
    y = np.linspace(-1, 1, h, endpoint=False)[:, None]
    x = np.linspace(-1, 1, w, endpoint=False)[None, :]
    r = np.sqrt(x * x + y * y)
    r = np.clip(r, 0.0, 1.0 - 1e-6)

    # --- FFT of source L channel ---
    Fsrc = np.fft.fftshift(np.fft.fft2(L_float))
    Msrc = np.abs(Fsrc)
    phase_src = np.angle(Fsrc)

    # --- 2D Gaussian blur of source magnitude ---
    sigma_blur = max(1, radial_smooth)
    blurred_Msrc = gaussian_filter(Msrc, sigma=sigma_blur)

    eps = 1e-8

    # --- Compute target (blurred) magnitude ---
    if mode == 'ref' and ref_img_arr is not None:
        # Resize reference if needed
        if ref_img_arr.shape[0] != h or ref_img_arr.shape[1] != w:
            ref_pil = Image.fromarray(ref_img_arr).resize((w, h), resample=Image.BICUBIC)
            ref_img_arr = np.array(ref_pil)
        # Convert reference to L channel
        if ref_img_arr.ndim == 3:
            ref_lab = cv2.cvtColor(ref_img_arr, cv2.COLOR_RGB2LAB)
            ref_L = ref_lab[:, :, 0].astype(np.float32)
        else:
            ref_L = ref_img_arr.astype(np.float32)
        Fref = np.fft.fftshift(np.fft.fft2(ref_L))
        Mref = np.abs(Fref)
        blurred_Mref = gaussian_filter(Mref, sigma=sigma_blur)

        # Scale reference magnitude to match source energy in low-frequency region
        lf_mask = r < cutoff
        lf_src = np.mean(blurred_Msrc[lf_mask]) + eps
        lf_ref = np.mean(blurred_Mref[lf_mask]) + eps
        blurred_Mref *= (lf_src / lf_ref)

        multiplier_2d = blurred_Mref / (blurred_Msrc + eps)

    elif mode == 'model':
        # Build a 2D 1/f^alpha power-law map
        freq_r = r.copy()
        freq_r[freq_r < eps] = eps
        power_law_2d = (1.0 / freq_r) ** (alpha / 2.0)
        blurred_model = gaussian_filter(power_law_2d, sigma=sigma_blur)

        # Scale to match source low-frequency energy
        lf_mask = r < cutoff
        lf_src = np.mean(blurred_Msrc[lf_mask]) + eps
        lf_model = np.mean(blurred_model[lf_mask]) + eps
        blurred_model *= (lf_src / lf_model)

        multiplier_2d = blurred_model / (blurred_Msrc + eps)

    else:
        multiplier_2d = np.ones((h, w), dtype=np.float64)

    multiplier_2d = np.clip(multiplier_2d, 0.1, 10.0)

    # --- Inverted weight map: protect LOW frequencies, modify HIGH ---
    edge = 0.05 + 0.02 * (1.0 - cutoff)
    edge = max(edge, 1e-6)
    weight = np.where(
        r < cutoff,
        0.0,
        np.where(
            r < cutoff + edge,
            0.5 * (1.0 - np.cos(np.pi * (r - cutoff) / edge)),
            1.0,
        ),
    )

    final_multiplier = 1.0 + (multiplier_2d - 1.0) * (weight * strength)

    # Optional randomness (weighted to high-freq region)
    if randomness and randomness > 0.0:
        noise = rng.normal(loc=1.0, scale=randomness, size=final_multiplier.shape)
        final_multiplier *= (1.0 + (noise - 1.0) * weight)

    # --- Apply multiplier to magnitude, keep original phase ---
    mag_modified = Msrc * final_multiplier
    Fshift_modified = mag_modified * np.exp(1j * phase_src)

    # --- Inverse FFT ---
    F_ishift = np.fft.ifftshift(Fshift_modified)
    L_back = np.real(np.fft.ifft2(F_ishift))

    # Blend with original L based on strength
    L_blended = (1.0 - strength) * L_float + strength * L_back
    L_out = np.clip(L_blended, 0, 255).astype(np.uint8)

    # --- Recombine ---
    if is_gray:
        return L_out
    else:
        lab_out = np.stack([L_out, A_chan, B_chan], axis=2)
        rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)
        return rgb_out
