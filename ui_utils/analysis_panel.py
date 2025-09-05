#!/usr/bin/env python3
"""
Analysis panel for histogram, FFT, radial profile, GLCM, and LBP plots.
Designed to plug straight into the provided run.py / MainWindow.

Exposes AnalysisPanel(title: str) with method update_from_path(path)
and clear_plots(). Uses helpers from utils:
- compute_gray_array(path) -> 2D numpy.ndarray (grayscale 0-255)
- compute_fft_magnitude(gray) -> (mag, mag_log)
- radial_profile(mag) -> (centers, radial)
- compute_glcm(gray) -> (glcm, features)
- compute_lbp(gray) -> (lbp, hist)
- make_canvas(width, height) -> (FigureCanvas, Axes)
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QSizePolicy, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import os

from utils import compute_gray_array, compute_fft_magnitude, radial_profile, compute_glcm, compute_lbp, make_canvas


class AnalysisPanel(QWidget):
    def __init__(self, title: str = "Analysis", parent=None):
        super().__init__(parent)
        self.setMinimumHeight(360)  # Increased to accommodate additional plots

        # Top-level layout + framed group
        v = QVBoxLayout(self)
        box = QGroupBox(title)
        vbox = QVBoxLayout()
        box.setLayout(vbox)

        # Two rows of plots: top row for histogram/FFT/radial, bottom for GLCM/LBP
        row1 = QHBoxLayout()
        row2 = QHBoxLayout()

        # Create canvases using project's make_canvas helper
        self.hist_canvas, self.hist_ax = make_canvas(width=3, height=2)
        self.fft_canvas, self.fft_ax = make_canvas(width=3, height=2)
        self.radial_canvas, self.radial_ax = make_canvas(width=3, height=2)
        self.glcm_canvas, self.glcm_ax = make_canvas(width=4.5, height=2)  # Wider for multiple features
        self.lbp_canvas, self.lbp_ax = make_canvas(width=4.5, height=2)

        # Configure size policy and margins for all canvases
        for c in (self.hist_canvas, self.fft_canvas, self.radial_canvas, self.glcm_canvas, self.lbp_canvas):
            c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            try:
                c.figure.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.96)
            except Exception:
                pass

        # Add to layouts
        row1.addWidget(self.hist_canvas)
        row1.addWidget(self.fft_canvas)
        row1.addWidget(self.radial_canvas)
        row2.addWidget(self.glcm_canvas)
        row2.addWidget(self.lbp_canvas)

        vbox.addLayout(row1)
        vbox.addLayout(row2)

        # Status label for diagnostics
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        vbox.addWidget(self.status_label)

        v.addWidget(box)

    def update_from_path(self, path: str):
        """Update all five plots using the image at `path`.

        If path is invalid or an error occurs, plots are cleared and a status message is shown.
        """
        if not path or not os.path.exists(path):
            self.status_label.setText(f"No image: {path}")
            self.status_label.setVisible(True)
            self.clear_plots()
            return

        try:
            gray = compute_gray_array(path)
            if gray is None:
                raise ValueError("compute_gray_array returned None")
            gray = np.asarray(gray)
            if gray.ndim != 2:
                raise ValueError("expected 2D grayscale array")
        except Exception as e:
            self.status_label.setText(f"Failed to load image: {e}")
            self.status_label.setVisible(True)
            self.clear_plots()
            return

        self.status_label.setVisible(False)

        # -------------------- Histogram --------------------
        try:
            self.hist_ax.cla()
            self.hist_ax.set_title('Grayscale histogram')
            self.hist_ax.set_xlabel('Intensity')
            self.hist_ax.set_ylabel('Count')
            flat = gray.ravel()
            if flat.dtype.kind == 'f' and flat.max() <= 1.0:
                flat = (flat * 255.0).astype(np.uint8)
            self.hist_ax.hist(flat, bins=256, range=(0, 255))
            self.hist_canvas.draw()
        except Exception as e:
            self.hist_ax.cla()
            self.hist_canvas.draw()
            self.status_label.setText(f"Histogram error: {e}")
            self.status_label.setVisible(True)

        # -------------------- FFT magnitude --------------------
        try:
            mag, mag_log = compute_fft_magnitude(gray)
            if mag_log is None:
                raise ValueError("compute_fft_magnitude returned None")
            self.fft_ax.cla()
            self.fft_ax.set_title('FFT magnitude (log)')
            self.fft_ax.imshow(mag_log, origin='lower', aspect='auto', cmap='inferno')
            self.fft_ax.set_xticks([])
            self.fft_ax.set_yticks([])
            try:
                self.fft_canvas.figure.subplots_adjust(right=0.92)
            except Exception:
                pass
            self.fft_canvas.draw()
        except Exception as e:
            self.fft_ax.cla()
            self.fft_canvas.draw()
            self.status_label.setText(f"FFT error: {e}")
            self.status_label.setVisible(True)

        # -------------------- Radial profile --------------------
        try:
            centers, radial = radial_profile(mag)
            if centers is None or radial is None:
                raise ValueError("radial_profile returned invalid data")
            self.radial_ax.cla()
            self.radial_ax.set_title('Radial freq profile')
            self.radial_ax.set_xlabel('Normalized radius')
            self.radial_ax.set_ylabel('Mean magnitude')
            self.radial_ax.plot(centers, radial)
            self.radial_canvas.draw()
        except Exception as e:
            self.radial_ax.cla()
            self.radial_canvas.draw()
            self.status_label.setText(f"Radial profile error: {e}")
            self.status_label.setVisible(True)

        # -------------------- GLCM Features --------------------
        try:
            _, features = compute_glcm(gray)
            if features is None:
                raise ValueError("compute_glcm returned None")
            self.glcm_ax.cla()
            self.glcm_ax.set_title('GLCM Features')
            self.glcm_ax.set_xlabel('Offset')
            self.glcm_ax.set_ylabel('Value')
            offsets = ['(0,1)', '(1,0)', '(1,1)', '(-1,1)']
            x = np.arange(len(offsets))
            for feature in ['contrast', 'correlation', 'energy', 'homogeneity']:
                self.glcm_ax.plot(x, features[feature], label=feature, marker='o')
            self.glcm_ax.set_xticks(x)
            self.glcm_ax.set_xticklabels(offsets)
            self.glcm_ax.legend()
            self.glcm_canvas.draw()
        except Exception as e:
            self.glcm_ax.cla()
            self.glcm_canvas.draw()
            self.status_label.setText(f"GLCM error: {e}")
            self.status_label.setVisible(True)

        # -------------------- LBP Histogram --------------------
        try:
            _, hist = compute_lbp(gray)
            if hist is None:
                raise ValueError("compute_lbp returned None")
            self.lbp_ax.cla()
            self.lbp_ax.set_title('LBP Histogram')
            self.lbp_ax.set_xlabel('Pattern')
            self.lbp_ax.set_ylabel('Frequency')
            self.lbp_ax.bar(range(len(hist)), hist)
            self.lbp_ax.set_xticks(range(len(hist)))
            self.lbp_canvas.draw()
        except Exception as e:
            self.lbp_ax.cla()
            self.lbp_canvas.draw()
            self.status_label.setText(f"LBP error: {e}")
            self.status_label.setVisible(True)

    def clear_plots(self):
        """Clear all axes and redraw empty canvases."""
        for ax, canvas in (
            (self.hist_ax, self.hist_canvas),
            (self.fft_ax, self.fft_canvas),
            (self.radial_ax, self.radial_canvas),
            (self.glcm_ax, self.glcm_canvas),
            (self.lbp_ax, self.lbp_canvas)
        ):
            try:
                ax.cla()
                if ax is self.hist_ax:
                    ax.text(0.5, 0.5, 'No image', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                canvas.draw()
            except Exception:
                pass