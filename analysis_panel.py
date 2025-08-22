#!/usr/bin/env python3
"""
Analysis panel for histogram, FFT, and radial profile plots.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import os
from utils import compute_gray_array, compute_fft_magnitude, radial_profile, make_canvas

class AnalysisPanel(QWidget):
    def __init__(self, title="Analysis", parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        box = QGroupBox(title)
        vbox = QVBoxLayout()
        box.setLayout(vbox)

        row = QHBoxLayout()
        self.hist_canvas, self.hist_ax = make_canvas(width=3, height=2)
        self.fft_canvas, self.fft_ax = make_canvas(width=3, height=2)
        self.radial_canvas, self.radial_ax = make_canvas(width=3, height=2)

        for c in (self.hist_canvas, self.fft_canvas, self.radial_canvas):
            c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        row.addWidget(self.hist_canvas)
        row.addWidget(self.fft_canvas)
        row.addWidget(self.radial_canvas)

        vbox.addLayout(row)
        v.addWidget(box)

    def update_from_path(self, path):
        if not path or not os.path.exists(path):
            self.clear_plots()
            return
        try:
            gray = compute_gray_array(path)
        except Exception:
            self.clear_plots()
            return

        # Histogram
        self.hist_ax.cla()
        self.hist_ax.set_title('Grayscale histogram')
        self.hist_ax.set_xlabel('Intensity')
        self.hist_ax.set_ylabel('Count')
        self.hist_ax.hist(gray.ravel(), bins=256)
        self.hist_canvas.draw()

        # FFT magnitude
        mag, mag_log = compute_fft_magnitude(gray)
        self.fft_ax.cla()
        self.fft_ax.set_title('FFT magnitude (log)')
        self.fft_ax.imshow(mag_log, origin='lower', aspect='auto')
        self.fft_canvas.figure.subplots_adjust(right=0.85)
        self.fft_canvas.draw()

        # Radial profile
        centers, radial = radial_profile(mag)
        self.radial_ax.cla()
        self.radial_ax.set_title('Radial freq profile')
        self.radial_ax.set_xlabel('Normalized radius')
        self.radial_ax.set_ylabel('Mean magnitude')
        self.radial_ax.plot(centers, radial)
        self.radial_canvas.draw()

    def clear_plots(self):
        for ax, canvas in ((self.hist_ax, self.hist_canvas), (self.fft_ax, self.fft_canvas), (self.radial_ax, self.radial_canvas)):
            ax.cla()
            canvas.draw()