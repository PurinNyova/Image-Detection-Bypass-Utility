import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

def clahe_color_correction(img_arr: np.ndarray, clip_limit=2.0, tile_grid_size=(8,8)) -> np.ndarray:
    if _HAS_CV2:
        lab = cv2.cvtColor(img_arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        return out
    else:
        pil = Image.fromarray(img_arr)
        channels = pil.split()
        new_ch = []
        for ch in channels:
            eq = ImageOps.equalize(ch)
            new_ch.append(eq)
        merged = Image.merge('RGB', new_ch)
        return np.array(merged)