import numpy as np

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