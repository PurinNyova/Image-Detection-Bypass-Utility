#!/usr/bin/env python3
"""
Utility functions for image processing GUI.
"""

from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def qpixmap_from_path(p: str, max_size=(480, 360)) -> QPixmap:
    """
    Load an image from a path into a QPixmap, scaled to a maximum size.

    Parameters:
    - p (str): The file path of the image.
    - max_size (tuple): A tuple (width, height) for the maximum dimensions.

    Returns:
    - QPixmap: The scaled pixmap. Returns an empty QPixmap if the path is invalid.
    """
    pix = QPixmap(p)
    if pix.isNull():
        return QPixmap()
    w, h = max_size
    return pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

def make_canvas(width=4, height=3, dpi=100):
    """
    Create a Matplotlib canvas and axes for embedding in a PyQt GUI.

    Parameters:
    - width (int): The width of the figure in inches.
    - height (int): The height of the figure in inches.
    - dpi (int): The resolution of the figure in dots per inch.

    Returns:
    - tuple: A tuple containing the FigureCanvas and the Axes object (canvas, ax).
    """
    fig = Figure(figsize=(width, height), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    fig.tight_layout()
    return canvas, ax

def compute_gray_array(path):
    """
    Open an image, convert it to grayscale, and return it as a NumPy array.

    The conversion uses the luminosity method: Y = 0.299*R + 0.587*G + 0.114*B.

    Parameters:
    - path (str): The file path of the image.

    Returns:
    - np.ndarray: A 2D NumPy array of type float32 representing the grayscale image.
    """
    img = Image.open(path).convert('RGB')
    arr = np.array(img)
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
    return gray

def compute_fft_magnitude(gray_arr, eps=1e-8):
    """
    Compute the 2D FFT magnitude spectrum of a grayscale image.

    This function calculates the Fast Fourier Transform, shifts the zero-frequency
    component to the center, and returns both the absolute magnitude and a
    log-scaled version for better visualization.

    Parameters:
    - gray_arr (np.ndarray): The input 2D grayscale image array.
    - eps (float): A small epsilon value (currently unused in the implementation).

    Returns:
    - tuple: A tuple containing:
        - mag (np.ndarray): The centered FFT magnitude spectrum.
        - mag_log (np.ndarray): The log-scaled magnitude spectrum (log(1 + mag)).
    """
    f = np.fft.fft2(gray_arr)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag_log = np.log1p(mag)
    return mag, mag_log

def radial_profile(mag, center=None, nbins=100):
    """
    Compute the radially averaged profile of a 2D array (e.g., FFT magnitude).

    This calculates the average value in concentric rings starting from a center point.

    Parameters:
    - mag (np.ndarray): The 2D input array.
    - center (tuple, optional): The (y, x) coordinates of the center. If None,
      the geometric center of the array is used.
    - nbins (int): The number of radial bins to use.

    Returns:
    - tuple: A tuple containing:
        - centers (np.ndarray): The normalized radial distance for each bin (0 to 1).
        - radial_mean (np.ndarray): The mean value for each radial bin.
    """
    h, w = mag.shape
    if center is None:
        center = (int(h / 2), int(w / 2))
    y, x = np.indices((h, w))
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r_flat = r.ravel()
    mag_flat = mag.ravel()
    max_r = np.max(r_flat)
    if max_r <= 0:
        return np.linspace(0, 1, nbins), np.zeros(nbins)
    bins = np.linspace(0, max_r, nbins + 1)
    inds = np.digitize(r_flat, bins) - 1
    radial_mean = np.zeros(nbins)
    for i in range(nbins):
        sel = inds == i
        if np.any(sel):
            radial_mean[i] = mag_flat[sel].mean()
        else:
            radial_mean[i] = 0.0
    centers = 0.5 * (bins[:-1] + bins[1:]) / max_r
    return centers, radial_mean


def compute_glcm(gray_arr, levels=8, offsets=[(0, 1), (1, 0), (1, 1), (-1, 1)], eps=1e-8):
    """
    Vectorized Gray-Level Co-occurrence Matrix (GLCM) computation.

    Replaces per-pixel Python loops with NumPy-indexed operations and
    np.bincount to accumulate co-occurrence counts. This yields large
    speedups for typical image sizes.
    """
    # Quantize grayscale image to 'levels' bins
    gray = gray_arr.astype(np.float32)
    gray_min, gray_max = gray.min(), gray.max()
    if gray_max > gray_min:
        gray_quant = np.floor((gray - gray_min) / (gray_max - gray_min + eps) * (levels - 1)).astype(np.int32)
    else:
        gray_quant = np.zeros_like(gray, dtype=np.int32)

    h, w = gray_quant.shape
    glcm = np.zeros((levels, levels, len(offsets)), dtype=np.float32)
    features = {
        'contrast': np.zeros(len(offsets), dtype=np.float32),
        'correlation': np.zeros(len(offsets), dtype=np.float32),
        'energy': np.zeros(len(offsets), dtype=np.float32),
        'homogeneity': np.zeros(len(offsets), dtype=np.float32)
    }

    # Precompute index grids
    rows, cols = np.indices((h, w))

    # Precompute i,j matrices for feature computation
    i_idx = np.arange(levels).reshape(levels, 1)
    j_idx = np.arange(levels).reshape(1, levels)

    for k, (dy, dx) in enumerate(offsets):
        r2 = rows + dy
        c2 = cols + dx
        valid = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
        if not np.any(valid):
            continue

        q1 = gray_quant[rows[valid], cols[valid]]
        q2 = gray_quant[r2[valid], c2[valid]]

        # Flatten pair indices to accumulate with bincount (fast C loop)
        pairs = q1 * levels + q2
        counts = np.bincount(pairs, minlength=levels * levels).astype(np.float32)
        glcm_k = counts.reshape((levels, levels))

        # Normalize
        total = glcm_k.sum()
        if total > 0:
            glcm_k /= total

        # Features
        contrast = np.sum((i_idx - j_idx) ** 2 * glcm_k)
        mu_i = np.sum(i_idx * glcm_k)
        mu_j = np.sum(j_idx * glcm_k)
        sigma_i = np.sqrt(np.sum((i_idx - mu_i) ** 2 * glcm_k) + eps)
        sigma_j = np.sqrt(np.sum((j_idx - mu_j) ** 2 * glcm_k) + eps)
        if sigma_i > 0 and sigma_j > 0:
            correlation = np.sum((i_idx - mu_i) * (j_idx - mu_j) * glcm_k) / (sigma_i * sigma_j)
        else:
            correlation = 0.0
        energy = np.sum(glcm_k ** 2)
        homogeneity = np.sum(glcm_k / (1.0 + np.abs(i_idx - j_idx) + eps))

        glcm[:, :, k] = glcm_k
        features['contrast'][k] = contrast
        features['correlation'][k] = correlation
        features['energy'][k] = energy
        features['homogeneity'][k] = homogeneity

    return glcm, features


# ---------------------- Optimized LBP ----------------------

def compute_lbp(gray_arr, radius=1, n_points=8, eps=1e-8):
    """
    Vectorized Local Binary Pattern (uniform, rotation-invariant).

    Uses NumPy operations to sample all neighbor points at once and
    build LBP codes without explicit per-pixel Python loops.
    """
    gray = gray_arr.astype(np.float32)
    h, w = gray.shape

    # Pad to avoid boundary checks during interpolation
    pad = int(np.ceil(radius)) + 1
    padded = np.pad(gray, pad_width=pad, mode='edge')

    # Center coordinates in padded image
    Y, X = np.indices((h, w))
    Y = Y + pad
    X = X + pad

    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    lbp = np.zeros((h, w), dtype=np.uint32)

    for k, theta in enumerate(angles):
        # Note: y increases downward in image coordinates
        dy = -radius * np.sin(theta)
        dx = radius * np.cos(theta)

        sample_y = Y.astype(np.float32) + dy
        sample_x = X.astype(np.float32) + dx

        # Bilinear interpolation indices and weights
        y0 = np.floor(sample_y).astype(np.int32)
        x0 = np.floor(sample_x).astype(np.int32)
        y1 = y0 + 1
        x1 = x0 + 1

        wy = sample_y - y0
        wx = sample_x - x0

        # Gather values
        v00 = padded[y0, x0]
        v10 = padded[y1, x0]
        v01 = padded[y0, x1]
        v11 = padded[y1, x1]

        vals = (1 - wy) * (1 - wx) * v00 + wy * (1 - wx) * v10 + (1 - wy) * wx * v01 + wy * wx * v11

        # Compare with center
        center = padded[Y, X]
        bit = (vals >= center).astype(np.uint32)
        lbp |= (bit << k)

    # Map to rotation-invariant uniform patterns
    n_codes = 1 << n_points
    codes = np.arange(n_codes, dtype=np.uint32)
    bits = ((codes[:, None] >> np.arange(n_points)) & 1).astype(np.uint8)
    transitions = np.sum(bits != np.roll(bits, -1, axis=1), axis=1)
    uniform_mask = transitions <= 2

    lbp_map = np.full(n_codes, n_points + 1, dtype=np.int32)
    lbp_map[uniform_mask] = np.arange(np.count_nonzero(uniform_mask), dtype=np.int32)[np.argsort(np.nonzero(uniform_mask)[0])]
    # The above ensures uniform codes get unique indices from 0..(num_uniform-1)

    lbp_mapped = lbp_map[lbp]

    n_bins = lbp_map.max() + 1
    hist = np.bincount(lbp_mapped.ravel(), minlength=n_bins).astype(np.float32)
    hist /= (hist.sum() + eps)

    return lbp_mapped, hist
