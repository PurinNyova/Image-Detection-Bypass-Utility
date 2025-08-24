import numpy as np

def randomized_perturbation(img_arr: np.ndarray, magnitude_frac=0.008, seed=None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    mag = magnitude_frac * 255.0
    perturb = np.random.uniform(low=-mag, high=mag, size=img_arr.shape)
    out = img_arr.astype(np.float32) + perturb
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out