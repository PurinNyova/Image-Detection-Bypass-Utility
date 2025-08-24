import numpy as np
import re, os
from PIL import Image

def apply_1d_lut(img_arr: np.ndarray, lut: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Apply a 1D LUT to an image.
    - img_arr: HxWx3 uint8
    - lut: either shape (256,) (applied equally to all channels), (256,3) (per-channel),
           or (N,) / (N,3) (interpolated across [0..255])
    - strength: 0..1 blending between original and LUT result
    Returns uint8 array.
    """
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        raise ValueError("apply_1d_lut expects an HxWx3 image array")

    # Normalize indices 0..255
    arr = img_arr.astype(np.float32)
    # Prepare LUT as float in 0..255 range if necessary
    lut_arr = np.array(lut, dtype=np.float32)

    # If single channel LUT (N,) expand to three channels
    if lut_arr.ndim == 1:
        lut_arr = np.stack([lut_arr, lut_arr, lut_arr], axis=1)  # (N,3)

    if lut_arr.shape[1] != 3:
        raise ValueError("1D LUT must have shape (N,) or (N,3)")

    # Build index positions in source LUT space (0..255)
    N = lut_arr.shape[0]
    src_positions = np.linspace(0, 255, N)

    # Flatten and interpolate per channel
    out = np.empty_like(arr)
    for c in range(3):
        channel = arr[..., c].ravel()
        mapped = np.interp(channel, src_positions, lut_arr[:, c])
        out[..., c] = mapped.reshape(arr.shape[0], arr.shape[1])

    out = np.clip(out, 0, 255).astype(np.uint8)
    if strength >= 1.0:
        return out
    else:
        blended = ((1.0 - strength) * img_arr.astype(np.float32) + strength * out.astype(np.float32))
        return np.clip(blended, 0, 255).astype(np.uint8)

def _trilinear_sample_lut(img_float: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Vectorized trilinear sampling of 3D LUT.
    - img_float: HxWx3 floats in [0,1]
    - lut: SxSxS x 3 floats in [0,1]
    Returns HxWx3 floats in [0,1]
    """
    S = lut.shape[0]
    if lut.shape[0] != lut.shape[1] or lut.shape[1] != lut.shape[2]:
        raise ValueError("3D LUT must be cubic (SxSxSx3)")

    # map [0,1] -> [0, S-1]
    idx = img_float * (S - 1)
    r_idx = idx[..., 0]
    g_idx = idx[..., 1]
    b_idx = idx[..., 2]

    r0 = np.floor(r_idx).astype(np.int32)
    g0 = np.floor(g_idx).astype(np.int32)
    b0 = np.floor(b_idx).astype(np.int32)

    r1 = np.clip(r0 + 1, 0, S - 1)
    g1 = np.clip(g0 + 1, 0, S - 1)
    b1 = np.clip(b0 + 1, 0, S - 1)

    dr = (r_idx - r0)[..., None]
    dg = (g_idx - g0)[..., None]
    db = (b_idx - b0)[..., None]

    # gather 8 corners: c000 ... c111
    c000 = lut[r0, g0, b0]
    c001 = lut[r0, g0, b1]
    c010 = lut[r0, g1, b0]
    c011 = lut[r0, g1, b1]
    c100 = lut[r1, g0, b0]
    c101 = lut[r1, g0, b1]
    c110 = lut[r1, g1, b0]
    c111 = lut[r1, g1, b1]

    # interpolate along b
    c00 = c000 * (1 - db) + c001 * db
    c01 = c010 * (1 - db) + c011 * db
    c10 = c100 * (1 - db) + c101 * db
    c11 = c110 * (1 - db) + c111 * db

    # interpolate along g
    c0 = c00 * (1 - dg) + c01 * dg
    c1 = c10 * (1 - dg) + c11 * dg

    # interpolate along r
    c = c0 * (1 - dr) + c1 * dr

    return c  # float in same range as lut (expected [0,1])

def apply_3d_lut(img_arr: np.ndarray, lut3d: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Apply a 3D LUT to the image.
    - img_arr: HxWx3 uint8
    - lut3d: SxSxSx3 float (expected range 0..1)
    - strength: blending 0..1
    Returns uint8 image.
    """
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        raise ValueError("apply_3d_lut expects an HxWx3 image array")

    img_float = img_arr.astype(np.float32) / 255.0
    sampled = _trilinear_sample_lut(img_float, lut3d)  # HxWx3 floats in [0,1]
    out = np.clip(sampled * 255.0, 0, 255).astype(np.uint8)
    if strength >= 1.0:
        return out
    else:
        blended = ((1.0 - strength) * img_arr.astype(np.float32) + strength * out.astype(np.float32))
        return np.clip(blended, 0, 255).astype(np.uint8)

def apply_lut(img_arr: np.ndarray, lut: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Auto-detect LUT type and apply.
    - If lut.ndim in (1,2) treat as 1D LUT (per-channel if shape (N,3)).
    - If lut.ndim == 4 treat as 3D LUT (SxSxSx3) in [0,1].
    """
    lut = np.array(lut)
    if lut.ndim == 4 and lut.shape[3] == 3:
        # 3D LUT (assumed normalized [0..1])
        # If lut is in 0..255, normalize
        if lut.dtype != np.float32 and lut.max() > 1.0:
            lut = lut.astype(np.float32) / 255.0
        return apply_3d_lut(img_arr, lut, strength=strength)
    elif lut.ndim in (1, 2):
        return apply_1d_lut(img_arr, lut, strength=strength)
    else:
        raise ValueError("Unsupported LUT shape: {}".format(lut.shape))

def load_cube_lut(path: str) -> np.ndarray:
    """
    Parse a .cube file and return a 3D LUT array of shape (S,S,S,3) with float values in [0,1].
    Note: .cube file order sometimes varies; this function assumes standard ordering
    where data lines are triples of floats and LUT_3D_SIZE specifies S.
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]

    size = None
    data = []
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    for ln in lines:
        if ln.upper().startswith('LUT_3D_SIZE'):
            parts = ln.split()
            if len(parts) >= 2:
                size = int(parts[1])
        elif ln.upper().startswith('DOMAIN_MIN'):
            parts = ln.split()
            domain_min = np.array([float(p) for p in parts[1:4]], dtype=np.float32)
        elif ln.upper().startswith('DOMAIN_MAX'):
            parts = ln.split()
            domain_max = np.array([float(p) for p in parts[1:4]], dtype=np.float32)
        elif re.match(r'^-?\d+(\.\d+)?\s+-?\d+(\.\d+)?\s+-?\d+(\.\d+)?$', ln):
            parts = [float(x) for x in ln.split()]
            data.append(parts)

    if size is None:
        raise ValueError("LUT_3D_SIZE not found in .cube file: {}".format(path))

    data = np.array(data, dtype=np.float32)
    if data.shape[0] != size**3:
        raise ValueError("Cube LUT data length does not match size^3 (got {}, expected {})".format(data.shape[0], size**3))

    # Data ordering in many .cube files is: for r in 0..S-1: for g in 0..S-1: for b in 0..S-1: write RGB
    # We'll reshape into (S,S,S,3) with indices [r,g,b]
    lut = data.reshape((size, size, size, 3))
    # Map domain_min..domain_max to 0..1 if domain specified (rare)
    if not np.allclose(domain_min, [0.0, 0.0, 0.0]) or not np.allclose(domain_max, [1.0, 1.0, 1.0]):
        # scale lut values from domain range into 0..1
        lut = (lut - domain_min) / (domain_max - domain_min + 1e-12)
        lut = np.clip(lut, 0.0, 1.0)
    else:
        # ensure LUT is in [0,1] if not already
        if lut.max() > 1.0 + 1e-6:
            lut = lut / 255.0
    return lut.astype(np.float32)

def load_lut(path: str) -> np.ndarray:
    """
    Load a LUT from:
     - .npy (numpy saved array)
     - .cube (3D LUT)
     - image (PNG/JPG) that is a 1D LUT strip (common 256x1 or 1x256)
    Returns numpy array (1D, 2D, or 4D LUT).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path)
    elif ext == '.cube':
        return load_cube_lut(path)
    else:
        # try interpreting as image-based 1D LUT
        try:
            im = Image.open(path).convert('RGB')
            arr = np.array(im)
            h, w = arr.shape[:2]
            # 256x1 or 1x256 typical 1D LUT
            if (w == 256 and h == 1) or (h == 256 and w == 1):
                if h == 1:
                    lut = arr[0, :, :].astype(np.float32)
                else:
                    lut = arr[:, 0, :].astype(np.float32)
                return lut  # shape (256,3)
            # sometimes embedded as 512x16 or other tile layouts; attempt to flatten
            # fallback: flatten and try to build (N,3)
            flat = arr.reshape(-1, 3).astype(np.float32)
            # if length is perfect power-of-two and <= 1024, assume 1D
            L = flat.shape[0]
            if L <= 4096:
                return flat  # (L,3)
            raise ValueError("Image LUT not recognized size")
        except Exception as e:
            raise ValueError(f"Unsupported LUT file or parse error for {path}: {e}")
