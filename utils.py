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

def qpixmap_from_path(p: str, max_size=(480, 360)) -> QPixmap:
    pix = QPixmap(p)
    if pix.isNull():
        return QPixmap()
    w, h = max_size
    return pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

def make_canvas(width=4, height=3, dpi=100):
    fig = Figure(figsize=(width, height), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    fig.tight_layout()
    return canvas, ax

def compute_gray_array(path):
    img = Image.open(path).convert('RGB')
    arr = np.array(img)
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
    return gray

def compute_fft_magnitude(gray_arr, eps=1e-8):
    f = np.fft.fft2(gray_arr)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag_log = np.log1p(mag)
    return mag, mag_log

def radial_profile(mag, center=None, nbins=100):
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