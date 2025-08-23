#!/usr/bin/env python3
"""
Analysis panel for histogram, FFT, and radial profile plots.
Designed to plug straight into the provided run.py / MainWindow.

Exposes AnalysisPanel(title: str) with method update_from_path(path)
and clear_plots(). Uses helpers from utils:
- compute_gray_array(path) -> 2D numpy.ndarray (grayscale 0-255)
- compute_fft_magnitude(gray) -> (mag, mag_log)
- radial_profile(mag) -> (centers, radial)
- make_canvas(width, height) -> (FigureCanvas, Axes)

This module is intentionally defensive (catches errors) and keeps
its own layout compact so it will fit in the scrollable right-hand
panel in MainWindow.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QSizePolicy, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import os

from utils import compute_gray_array, compute_fft_magnitude, radial_profile, make_canvas


class AnalysisPanel(QWidget):
    def __init__(self, title: str = "Analysis", parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)

        # Top-level layout + framed group
        v = QVBoxLayout(self)
        box = QGroupBox(title)
        vbox = QVBoxLayout()
        box.setLayout(vbox)

        # Row of three matplotlib canvases
        row = QHBoxLayout()

        # create canvases using project's make_canvas helper so styles match
        self.hist_canvas, self.hist_ax = make_canvas(width=3, height=2)
        self.fft_canvas, self.fft_ax = make_canvas(width=3, height=2)
        self.radial_canvas, self.radial_ax = make_canvas(width=3, height=2)

        for c in (self.hist_canvas, self.fft_canvas, self.radial_canvas):
            c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            # give figures a consistent, compact margin so they sit well inside the GroupBox
            try:
                c.figure.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.96)
            except Exception:
                pass

        row.addWidget(self.hist_canvas)
        row.addWidget(self.fft_canvas)
        row.addWidget(self.radial_canvas)

        vbox.addLayout(row)

        # small status label below canvases for quick diagnostics
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        vbox.addWidget(self.status_label)

        v.addWidget(box)

    def update_from_path(self, path: str):
        """Update all three plots using the image at `path`.

        If path is invalid or an error occurs while loading/processing,
        plots are cleared and a status message is shown.
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
            # ensure grayscale array is 2D and finite
            gray = np.asarray(gray)
            if gray.ndim != 2:
                raise ValueError("expected 2D grayscale array")

        except Exception as e:
            self.status_label.setText(f"Failed to load image: {e}")
            self.status_label.setVisible(True)
            self.clear_plots()
            return

        # Good: hide status
        self.status_label.setVisible(False)

        # -------------------- Histogram --------------------
        try:
            self.hist_ax.cla()
            self.hist_ax.set_title('Grayscale histogram')
            self.hist_ax.set_xlabel('Intensity')
            self.hist_ax.set_ylabel('Count')
            # ensure int range 0..255
            flat = gray.ravel()
            # handle float data by scaling if necessary
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
            # use imshow with origin='lower' so low-frequencies sit near the centre visually
            self.fft_ax.imshow(mag_log, origin='lower', aspect='auto')
            self.fft_ax.set_xticks([])
            self.fft_ax.set_yticks([])
            # leave some room for colorbar in wider layouts (MainWindow adjusts overall sizes)
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

    def clear_plots(self):
        """Clear all axes and redraw empty canvases."""
        for ax, canvas in ((self.hist_ax, self.hist_canvas), (self.fft_ax, self.fft_canvas), (self.radial_ax, self.radial_canvas)):
            try:
                ax.cla()
                # give an empty centered message on histogram axis only
                if ax is self.hist_ax:
                    ax.text(0.5, 0.5, 'No image', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                canvas.draw()
            except Exception:
                pass
