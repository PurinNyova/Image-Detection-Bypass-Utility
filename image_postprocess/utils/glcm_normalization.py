import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

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
    GLCM normalization on localized luminance only.

    The RGB image is converted to LAB, only the L channel is transformed, and the
    original A/B channels are preserved to avoid chromatic artifacts.
    """
    rng = np.random.default_rng(seed)

    img = np.asarray(img_arr, dtype=np.uint8)
    h, w = img.shape[:2]

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel_f = l_channel.astype(np.float32)

    # Quantize luminance for GLCM computation.
    l_quantized = (l_channel_f / 255.0 * (levels - 1)).astype(np.int32)

    use_levels = levels
    if max_levels_for_speed is not None and max_levels_for_speed < levels:
        use_levels = max_levels_for_speed
        l_quantized = np.floor(l_channel_f / 255.0 * (use_levels - 1)).astype(np.uint8)
    else:
        l_quantized = l_quantized.astype(np.uint8)

    glcm = graycomatrix(l_quantized, distances=distances, angles=angles,
                        levels=use_levels, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()

    ref_contrast = None
    ref_homogeneity = None
    if ref_img_arr is not None:
        ref = np.asarray(ref_img_arr, dtype=np.uint8)
        if ref.shape[0] != h or ref.shape[1] != w:
            ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_CUBIC)

        ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB)
        ref_l = ref_lab[:, :, 0].astype(np.float32)
        if max_levels_for_speed is not None and max_levels_for_speed < levels:
            ref_quantized = np.floor(ref_l / 255.0 * (use_levels - 1)).astype(np.uint8)
        else:
            ref_quantized = (ref_l / 255.0 * (use_levels - 1)).astype(np.uint8)

        ref_glcm = graycomatrix(ref_quantized, distances=distances, angles=angles,
                                levels=use_levels, symmetric=True, normed=True)
        ref_contrast = graycoprops(ref_glcm, 'contrast').mean()
        ref_homogeneity = graycoprops(ref_glcm, 'homogeneity').mean()

    if strength > 0.0:
        noise_l = rng.normal(loc=0.0, scale=0.02 * strength, size=(h, w)).astype(np.float32) * 255.0
        noise_l = cv2.GaussianBlur(noise_l, (3, 3), sigmaX=0.5, sigmaY=0.5)
    else:
        noise_l = np.zeros((h, w), dtype=np.float32)

    if (ref_contrast is not None) and (ref_homogeneity is not None):
        contrast_ratio = ref_contrast / (contrast + eps)
        homogeneity_ratio = ref_homogeneity / (homogeneity + eps)
        contrast_scale = np.sqrt(contrast_ratio).astype(np.float32)
        bilateral_sigma = float(np.clip(75.0 / (homogeneity_ratio + eps), 25.0, 150.0))
    else:
        contrast_scale = None
        bilateral_sigma = None

    if contrast_scale is not None:
        adjusted_l = l_channel_f * contrast_scale
        adjusted_l = cv2.bilateralFilter(adjusted_l, d=9, sigmaColor=bilateral_sigma, sigmaSpace=bilateral_sigma)
        blended_l = (1.0 - strength) * l_channel_f + strength * adjusted_l
    else:
        blended_l = l_channel_f.copy()

    blended_l += noise_l
    out_l = np.clip(blended_l, 0, 255).astype(np.uint8)

    out_lab = cv2.merge((out_l, a_channel, b_channel))
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
