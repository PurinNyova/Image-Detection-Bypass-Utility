import numpy as np

def add_gaussian_noise(img_arr: np.ndarray, std_frac=0.02, seed=None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    std = std_frac * 255.0
    noise = np.random.normal(loc=0.0, scale=std, size=img_arr.shape)
    out = img_arr.astype(np.float32) + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out