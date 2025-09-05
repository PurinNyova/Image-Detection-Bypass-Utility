#!/usr/bin/env python3
"""
MainWindow definition extracted from the original single-file GUI.
All GUI wiring, widgets, and the MainWindow class live here.
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QFormLayout, QSlider, QSpinBox, QDoubleSpinBox,
    QProgressBar, QMessageBox, QLineEdit, QComboBox, QCheckBox, QToolButton, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from worker import Worker
from analysis_panel import AnalysisPanel
from utils import qpixmap_from_path
from collapsible_box import CollapsibleBox
from theme import apply_dark_palette
import numpy as np
import configparser

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- Load config.ini ---
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), "config.ini"))

        def get(section, key, default, cast=str):
            try:
                val = config.get(section, key)
                return cast(val)
            except Exception:
                return default

        def getbool(section, key, default):
            try:
                return config.getboolean(section, key)
            except Exception:
                return default

        # --- Window ---
        self.setWindowTitle("Image Detection Bypass Utility V1.3R2")
        self.setMinimumSize(1200, 760)

        central = QWidget()
        self.setCentralWidget(central)
        main_h = QHBoxLayout(central)

        # Left: previews & file selection
        left_v = QVBoxLayout()
        main_h.addLayout(left_v, 2)

        # Input/Output collapsible
        io_box = CollapsibleBox("Input / Output")
        left_v.addWidget(io_box)
        in_layout = QFormLayout()
        io_container = QWidget()
        io_container.setLayout(in_layout)
        io_box.content_layout.addWidget(io_container)

        self.input_line = QLineEdit()
        self.input_btn = QPushButton("Choose Input")
        self.input_btn.clicked.connect(self.choose_input)

        self.ref_line = QLineEdit()
        self.ref_btn = QPushButton("Choose AWB Reference (optional)")
        self.ref_btn.clicked.connect(self.choose_ref)

        self.fft_ref_line = QLineEdit()
        self.fft_ref_btn = QPushButton("Choose Reference (FFT, GLCM, LBP) (Optional)")
        self.fft_ref_btn.clicked.connect(self.choose_fft_ref)

        self.output_line = QLineEdit()
        self.output_btn = QPushButton("Choose Output")
        self.output_btn.clicked.connect(self.choose_output)

        in_layout.addRow(self.input_btn, self.input_line)
        in_layout.addRow(self.ref_btn, self.ref_line)
        in_layout.addRow(self.fft_ref_btn, self.fft_ref_line)
        in_layout.addRow(self.output_btn, self.output_line)

        # Previews
        self.preview_in = QLabel(alignment=Qt.AlignCenter)
        self.preview_in.setFixedSize(480, 300)
        self.preview_in.setStyleSheet("background:#121213; border:1px solid #2b2b2b; color:#ddd; border-radius:6px")
        self.preview_in.setText("Input preview")

        self.preview_out = QLabel(alignment=Qt.AlignCenter)
        self.preview_out.setFixedSize(480, 300)
        self.preview_out.setStyleSheet("background:#121213; border:1px solid #2b2b2b; color:#ddd; border-radius:6px")
        self.preview_out.setText("Output preview")

        left_v.addWidget(self.preview_in)
        left_v.addWidget(self.preview_out)

        # Actions
        actions_h = QHBoxLayout()
        self.run_btn = QPushButton("Run — Process Image")
        self.run_btn.clicked.connect(self.on_run)
        self.open_out_btn = QPushButton("Open Output Folder")
        self.open_out_btn.clicked.connect(self.open_output_folder)
        actions_h.addWidget(self.run_btn)
        actions_h.addWidget(self.open_out_btn)
        left_v.addLayout(actions_h)

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        left_v.addWidget(self.progress)

        # Right: controls + analysis panels (with scroll area)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        main_h.addWidget(scroll_area, 3)

        scroll_widget = QWidget()
        right_v = QVBoxLayout(scroll_widget)
        scroll_area.setWidget(scroll_widget)

        # Auto Mode toggle
        self.auto_mode_chk = QCheckBox("Enable Auto Mode")
        self.auto_mode_chk.setChecked(getbool("General", "auto_mode", False))
        self.auto_mode_chk.stateChanged.connect(self._on_auto_mode_toggled)
        right_v.addWidget(self.auto_mode_chk)

        # Auto Mode section collapsible
        self.auto_box = CollapsibleBox("Auto Mode")
        right_v.addWidget(self.auto_box)
        auto_layout = QFormLayout()
        auto_container = QWidget()
        auto_container.setLayout(auto_layout)
        self.auto_box.content_layout.addWidget(auto_container)

        strength_layout = QHBoxLayout()
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(get("AutoMode", "strength", 25, int))
        self.strength_slider.valueChanged.connect(self._update_strength_label)
        self.strength_label = QLabel(str(self.strength_slider.value()))
        self.strength_label.setFixedWidth(30)
        strength_layout.addWidget(self.strength_slider)
        strength_layout.addWidget(self.strength_label)
        auto_layout.addRow("Aberration Strength", strength_layout)

        # AI Normalizer
        self.ai_norm_box = CollapsibleBox("AI Normalizer")
        right_v.addWidget(self.ai_norm_box)
        ai_layout = QFormLayout()
        ai_container = QWidget()
        ai_container.setLayout(ai_layout)
        self.ai_norm_box.content_layout.addWidget(ai_container)

        self.ns_chk = QCheckBox("Enable AI Normalizer (Torch Required)")
        self.ns_chk.setToolTip("Enable AI Normalizer. Requires PyTorch.")
        self.ns_chk.setChecked(getbool("AINormalizer", "enabled", False))
        ai_layout.addRow(self.ns_chk)

        self.ns_iterations_spin = QSpinBox()
        self.ns_iterations_spin.setRange(1, 10000)
        self.ns_iterations_spin.setValue(get("AINormalizer", "iterations", 500, int))
        self.ns_iterations_spin.setToolTip("Number of iterations for the AI Normalizer optimization.")
        ai_layout.addRow("Iterations", self.ns_iterations_spin)

        self.ns_lr_spin = QDoubleSpinBox()
        self.ns_lr_spin.setDecimals(6)
        self.ns_lr_spin.setRange(0.000001, 0.1)
        self.ns_lr_spin.setSingleStep(0.0001)
        self.ns_lr_spin.setValue(get("AINormalizer", "learning_rate", 0.0003, float))
        self.ns_lr_spin.setToolTip("Learning rate for the AI Normalizer optimization.")
        ai_layout.addRow("Learning Rate", self.ns_lr_spin)

        self.ns_t_lpips_spin = QDoubleSpinBox()
        self.ns_t_lpips_spin.setDecimals(6)
        self.ns_t_lpips_spin.setRange(0.000001, 1.0)
        self.ns_t_lpips_spin.setSingleStep(0.0001)
        self.ns_t_lpips_spin.setValue(get("AINormalizer", "t_lpips", 0.04, float))
        self.ns_t_lpips_spin.setToolTip("Temporally weighted LPIPS loss parameter.")
        ai_layout.addRow("T LPIPS", self.ns_t_lpips_spin)

        self.ns_t_l2_spin = QDoubleSpinBox()
        self.ns_t_l2_spin.setDecimals(6)
        self.ns_t_l2_spin.setRange(0.000001, 1.0)
        self.ns_t_l2_spin.setSingleStep(0.00001)
        self.ns_t_l2_spin.setValue(get("AINormalizer", "t_l2", 3e-05, float))
        self.ns_t_l2_spin.setToolTip("Temporally weighted L2 loss parameter.")
        ai_layout.addRow("T L2", self.ns_t_l2_spin)

        self.ns_c_lpips_spin = QDoubleSpinBox()
        self.ns_c_lpips_spin.setDecimals(6)
        self.ns_c_lpips_spin.setRange(0.000001, 1.0)
        self.ns_c_lpips_spin.setSingleStep(0.0001)
        self.ns_c_lpips_spin.setValue(get("AINormalizer", "c_lpips", 0.01, float))
        self.ns_c_lpips_spin.setToolTip("Content loss LPIPS weight.")
        ai_layout.addRow("C LPIPS", self.ns_c_lpips_spin)

        self.ns_c_l2_spin = QDoubleSpinBox()
        self.ns_c_l2_spin.setDecimals(6)
        self.ns_c_l2_spin.setRange(0.000001, 10.0)
        self.ns_c_l2_spin.setSingleStep(0.01)
        self.ns_c_l2_spin.setValue(get("AINormalizer", "c_l2", 0.6, float))
        self.ns_c_l2_spin.setToolTip("Content loss L2 weight.")
        ai_layout.addRow("C L2", self.ns_c_l2_spin)

        self.ns_grad_clip_spin = QDoubleSpinBox()
        self.ns_grad_clip_spin.setDecimals(6)
        self.ns_grad_clip_spin.setRange(0.000001, 1.0)
        self.ns_grad_clip_spin.setSingleStep(0.0001)
        self.ns_grad_clip_spin.setValue(get("AINormalizer", "grad_clip", 0.05, float))
        self.ns_grad_clip_spin.setToolTip("Gradient clipping threshold to stabilize training.")
        ai_layout.addRow("Gradient Clip", self.ns_grad_clip_spin)

        # Parameters (Manual Mode) collapsible
        self.params_box = CollapsibleBox("Parameters (Manual Mode)")
        right_v.addWidget(self.params_box)
        params_layout = QFormLayout()
        params_container = QWidget()
        params_container.setLayout(params_layout)
        self.params_box.content_layout.addWidget(params_container)

        # New optional flags for processing steps
        self.noise_enable_chk = QCheckBox("Enable Gaussian Noise")
        self.noise_enable_chk.setChecked(getbool("ManualParameters", "noise_enable", True))
        params_layout.addRow(self.noise_enable_chk)

        self.clahe_enable_chk = QCheckBox("Enable CLAHE Color Correction")
        self.clahe_enable_chk.setChecked(getbool("ManualParameters", "clahe_enable", True))
        params_layout.addRow(self.clahe_enable_chk)

        self.fft_enable_chk = QCheckBox("Enable FFT Spectral Matching")
        self.fft_enable_chk.setChecked(getbool("ManualParameters", "fft_enable", True))
        params_layout.addRow(self.fft_enable_chk)

        self.perturb_enable_chk = QCheckBox("Enable Randomized Perturbation")
        self.perturb_enable_chk.setChecked(getbool("ManualParameters", "perturb_enable", True))
        params_layout.addRow(self.perturb_enable_chk)

        # Noise-std
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 0.1)
        self.noise_spin.setSingleStep(0.001)
        self.noise_spin.setValue(get("ManualParameters", "noise_std", 0.02, float))
        self.noise_spin.setToolTip("Gaussian noise std fraction of 255")
        params_layout.addRow("Noise std (0-0.1)", self.noise_spin)

        # CLAHE-clip
        self.clahe_spin = QDoubleSpinBox()
        self.clahe_spin.setRange(0.1, 10.0)
        self.clahe_spin.setSingleStep(0.1)
        self.clahe_spin.setValue(get("ManualParameters", "clahe_clip", 2.0, float))
        params_layout.addRow("CLAHE clip", self.clahe_spin)

        # Tile
        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(1, 64)
        self.tile_spin.setValue(get("ManualParameters", "tile", 8, int))
        params_layout.addRow("CLAHE tile", self.tile_spin)

        # Cutoff
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0.01, 1.0)
        self.cutoff_spin.setSingleStep(0.01)
        self.cutoff_spin.setValue(get("ManualParameters", "cutoff", 0.25, float))
        params_layout.addRow("Fourier cutoff (0-1)", self.cutoff_spin)

        # Fstrength
        self.fstrength_spin = QDoubleSpinBox()
        self.fstrength_spin.setRange(0.0, 1.0)
        self.fstrength_spin.setSingleStep(0.01)
        self.fstrength_spin.setValue(get("ManualParameters", "fstrength", 0.9, float))
        params_layout.addRow("Fourier strength (0-1)", self.fstrength_spin)

        # Randomness
        self.randomness_spin = QDoubleSpinBox()
        self.randomness_spin.setRange(0.0, 1.0)
        self.randomness_spin.setSingleStep(0.01)
        self.randomness_spin.setValue(get("ManualParameters", "randomness", 0.05, float))
        params_layout.addRow("Fourier randomness", self.randomness_spin)

        # Phase_perturb
        self.phase_perturb_spin = QDoubleSpinBox()
        self.phase_perturb_spin.setRange(0.0, 1.0)
        self.phase_perturb_spin.setSingleStep(0.001)
        self.phase_perturb_spin.setValue(get("ManualParameters", "phase_perturb", 0.08, float))
        self.phase_perturb_spin.setToolTip("Phase perturbation std (radians)")
        params_layout.addRow("Phase perturb (rad)", self.phase_perturb_spin)

        # Radial_smooth
        self.radial_smooth_spin = QSpinBox()
        self.radial_smooth_spin.setRange(0, 50)
        self.radial_smooth_spin.setValue(get("ManualParameters", "radial_smooth", 5, int))
        params_layout.addRow("Radial smooth (bins)", self.radial_smooth_spin)

        # FFT_mode
        self.fft_mode_combo = QComboBox()
        self.fft_mode_combo.addItems(["auto", "ref", "model"])
        self.fft_mode_combo.setCurrentText(get("ManualParameters", "fft_mode", "auto"))
        params_layout.addRow("FFT mode", self.fft_mode_combo)

        # FFT_alpha
        self.fft_alpha_spin = QDoubleSpinBox()
        self.fft_alpha_spin.setRange(0.1, 4.0)
        self.fft_alpha_spin.setSingleStep(0.1)
        self.fft_alpha_spin.setValue(get("ManualParameters", "fft_alpha", 1.0, float))
        self.fft_alpha_spin.setToolTip("Alpha exponent for 1/f model when using model mode")
        params_layout.addRow("FFT alpha (model)", self.fft_alpha_spin)

        # Perturb
        self.perturb_spin = QDoubleSpinBox()
        self.perturb_spin.setRange(0.0, 0.05)
        self.perturb_spin.setSingleStep(0.001)
        self.perturb_spin.setValue(get("ManualParameters", "perturb", 0.008, float))
        params_layout.addRow("Pixel perturb", self.perturb_spin)

        # Seed
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2 ** 31 - 1)
        self.seed_spin.setValue(get("ManualParameters", "seed", 0, int))
        params_layout.addRow("Seed (0=none)", self.seed_spin)

        # AWB checkbox
        self.awb_chk = QCheckBox("Enable auto white-balance (AWB)")
        self.awb_chk.setChecked(getbool("AWB", "enabled", False))
        self.awb_chk.setToolTip("If checked, AWB is applied. If a reference image is chosen, it will be used; otherwise gray-world AWB is applied.")
        params_layout.addRow(self.awb_chk)

        # Camera simulator toggle
        self.sim_camera_chk = QCheckBox("Enable camera pipeline simulation")
        self.sim_camera_chk.setChecked(getbool("CameraSimulator", "enabled", False))
        self.sim_camera_chk.stateChanged.connect(self._on_sim_camera_toggled)
        params_layout.addRow(self.sim_camera_chk)

        # LUT support UI
        self.lut_chk = QCheckBox("Enable LUT")
        self.lut_chk.setChecked(getbool("LUT", "enabled", False))
        self.lut_chk.setToolTip("Enable applying a 1D/.npy/.cube LUT to the output image")
        self.lut_chk.stateChanged.connect(self._on_lut_toggled)
        params_layout.addRow(self.lut_chk)

        self.lut_line = QLineEdit(get("LUT", "file", ""))
        self.lut_btn = QPushButton("Choose LUT")
        self.lut_btn.clicked.connect(self.choose_lut)
        lut_box = QWidget()
        lut_box_layout = QHBoxLayout()
        lut_box_layout.setContentsMargins(0, 0, 0, 0)
        lut_box.setLayout(lut_box_layout)
        lut_box_layout.addWidget(self.lut_line)
        lut_box_layout.addWidget(self.lut_btn)
        self.lut_file_label = QLabel("LUT file (png/.npy/.cube)")
        params_layout.addRow(self.lut_file_label, lut_box)

        self.lut_strength_spin = QDoubleSpinBox()
        self.lut_strength_spin.setRange(0.0, 1.0)
        self.lut_strength_spin.setSingleStep(0.01)
        self.lut_strength_spin.setValue(get("LUT", "strength", 1.0, float))
        self.lut_strength_spin.setToolTip("Blend strength for LUT (0.0 = no effect, 1.0 = full LUT)")
        self.lut_strength_label = QLabel("LUT strength")
        params_layout.addRow(self.lut_strength_label, self.lut_strength_spin)

        # Initially hide LUT controls and their labels
        self.lut_file_label.setVisible(False)
        lut_box.setVisible(False)
        self.lut_strength_label.setVisible(False)
        self.lut_strength_spin.setVisible(False)

        self._lut_controls = (self.lut_file_label, lut_box, self.lut_strength_label, self.lut_strength_spin)

        # Texture Normalization collapsible group
        self.texture_box = CollapsibleBox("Texture Normalization")
        right_v.addWidget(self.texture_box)
        texture_layout = QFormLayout()
        texture_container = QWidget()
        texture_container.setLayout(texture_layout)
        self.texture_box.content_layout.addWidget(texture_container)

        # GLCM checkbox
        self.glcm_chk = QCheckBox("Enable GLCM Normalization")
        self.glcm_chk.setChecked(getbool("TextureNormalization", "glcm_enabled", False))
        self.glcm_chk.setToolTip("Enable GLCM normalization using FFT reference image")
        texture_layout.addRow(self.glcm_chk)

        # GLCM distances
        self.glcm_distances_line = QLineEdit(get("TextureNormalization", "glcm_distances", "1"))
        self.glcm_distances_line.setToolTip("Space-separated list of distances for GLCM computation (e.g., '1 2 3')")
        texture_layout.addRow("GLCM Distances", self.glcm_distances_line)

        # GLCM angles
        self.glcm_angles_line = QLineEdit(get("TextureNormalization", "glcm_angles", "0 0.785 1.571 2.356"))
        self.glcm_angles_line.setToolTip("Space-separated list of angles in radians for GLCM (e.g., '0 0.785 1.571 2.356')")
        texture_layout.addRow("GLCM Angles (rad)", self.glcm_angles_line)

        # GLCM levels
        self.glcm_levels_spin = QSpinBox()
        self.glcm_levels_spin.setRange(2, 256)
        self.glcm_levels_spin.setValue(get("TextureNormalization", "glcm_levels", 256, int))
        self.glcm_levels_spin.setToolTip("Number of gray levels for GLCM")
        texture_layout.addRow("GLCM Levels", self.glcm_levels_spin)

        # GLCM strength
        self.glcm_strength_spin = QDoubleSpinBox()
        self.glcm_strength_spin.setRange(0.0, 1.0)
        self.glcm_strength_spin.setSingleStep(0.01)
        self.glcm_strength_spin.setValue(get("TextureNormalization", "glcm_strength", 0.9, float))
        self.glcm_strength_spin.setToolTip("Strength of GLCM feature matching (0.0 = no effect, 1.0 = full effect)")
        texture_layout.addRow("GLCM Strength", self.glcm_strength_spin)

        # LBP checkbox
        self.lbp_chk = QCheckBox("Enable LBP Normalization")
        self.lbp_chk.setChecked(getbool("TextureNormalization", "lbp_enabled", False))
        self.lbp_chk.setToolTip("Enable LBP normalization using FFT reference image")
        texture_layout.addRow(self.lbp_chk)

        # LBP radius
        self.lbp_radius_spin = QSpinBox()
        self.lbp_radius_spin.setRange(1, 10)
        self.lbp_radius_spin.setValue(get("TextureNormalization", "lbp_radius", 3, int))
        self.lbp_radius_spin.setToolTip("Radius of LBP operator")
        texture_layout.addRow("LBP Radius", self.lbp_radius_spin)

        # LBP n_points
        self.lbp_n_points_spin = QSpinBox()
        self.lbp_n_points_spin.setRange(8, 64)
        self.lbp_n_points_spin.setValue(get("TextureNormalization", "lbp_n_points", 24, int))
        self.lbp_n_points_spin.setToolTip("Number of circularly symmetric neighbor set points for LBP")
        texture_layout.addRow("LBP N Points", self.lbp_n_points_spin)

        # LBP method
        self.lbp_method_combo = QComboBox()
        self.lbp_method_combo.addItems(["default", "ror", "uniform", "var"])
        self.lbp_method_combo.setCurrentText(get("TextureNormalization", "lbp_method", "uniform"))
        self.lbp_method_combo.setToolTip("LBP method: default, ror, uniform, or var")
        texture_layout.addRow("LBP Method", self.lbp_method_combo)

        # LBP strength
        self.lbp_strength_spin = QDoubleSpinBox()
        self.lbp_strength_spin.setRange(0.0, 1.0)
        self.lbp_strength_spin.setSingleStep(0.01)
        self.lbp_strength_spin.setValue(get("TextureNormalization", "lbp_strength", 0.9, float))
        self.lbp_strength_spin.setToolTip("Strength of LBP histogram matching (0.0 = no effect, 1.0 = full effect)")
        texture_layout.addRow("LBP Strength", self.lbp_strength_spin)

        # Camera simulator collapsible group
        self.camera_box = CollapsibleBox("Camera simulator options")
        right_v.addWidget(self.camera_box)
        cam_layout = QFormLayout()
        cam_container = QWidget()
        cam_container.setLayout(cam_layout)
        self.camera_box.content_layout.addWidget(cam_container)

        # Enable bayer
        self.bayer_chk = QCheckBox("Enable Bayer / demosaic (RGGB)")
        self.bayer_chk.setChecked(getbool("CameraSimulator", "bayer", True))
        cam_layout.addRow(self.bayer_chk)

        # JPEG cycles
        self.jpeg_cycles_spin = QSpinBox()
        self.jpeg_cycles_spin.setRange(0, 10)
        self.jpeg_cycles_spin.setValue(get("CameraSimulator", "jpeg_cycles", 1, int))
        cam_layout.addRow("JPEG cycles", self.jpeg_cycles_spin)

        # JPEG quality min/max
        self.jpeg_qmin_spin = QSpinBox()
        self.jpeg_qmin_spin.setRange(1, 100)
        self.jpeg_qmin_spin.setValue(get("CameraSimulator", "jpeg_qmin", 88, int))
        self.jpeg_qmax_spin = QSpinBox()
        self.jpeg_qmax_spin.setRange(1, 100)
        self.jpeg_qmax_spin.setValue(get("CameraSimulator", "jpeg_qmax", 96, int))
        qbox = QHBoxLayout()
        qbox.addWidget(self.jpeg_qmin_spin)
        qbox.addWidget(QLabel("to"))
        qbox.addWidget(self.jpeg_qmax_spin)
        cam_layout.addRow("JPEG quality (min to max)", qbox)

        # Vignette strength
        self.vignette_spin = QDoubleSpinBox()
        self.vignette_spin.setRange(0.0, 1.0)
        self.vignette_spin.setSingleStep(0.01)
        self.vignette_spin.setValue(get("CameraSimulator", "vignette_strength", 0.35, float))
        cam_layout.addRow("Vignette strength", self.vignette_spin)

        # Chromatic aberration strength
        self.chroma_spin = QDoubleSpinBox()
        self.chroma_spin.setRange(0.0, 10.0)
        self.chroma_spin.setSingleStep(0.1)
        self.chroma_spin.setValue(get("CameraSimulator", "chroma_strength", 1.2, float))
        cam_layout.addRow("Chromatic aberration (px)", self.chroma_spin)

        # ISO scale
        self.iso_spin = QDoubleSpinBox()
        self.iso_spin.setRange(0.1, 16.0)
        self.iso_spin.setSingleStep(0.1)
        self.iso_spin.setValue(get("CameraSimulator", "iso_scale", 1.0, float))
        cam_layout.addRow("ISO/exposure scale", self.iso_spin)

        # Read noise
        self.read_noise_spin = QDoubleSpinBox()
        self.read_noise_spin.setRange(0.0, 50.0)
        self.read_noise_spin.setSingleStep(0.1)
        self.read_noise_spin.setValue(get("CameraSimulator", "read_noise", 2.0, float))
        cam_layout.addRow("Read noise (DN)", self.read_noise_spin)

        # Hot pixel prob
        self.hot_pixel_spin = QDoubleSpinBox()
        self.hot_pixel_spin.setDecimals(9)
        self.hot_pixel_spin.setRange(0.0, 1.0)
        self.hot_pixel_spin.setSingleStep(1e-6)
        self.hot_pixel_spin.setValue(get("CameraSimulator", "hot_pixel_prob", 1e-6, float))
        cam_layout.addRow("Hot pixel prob", self.hot_pixel_spin)

        # Banding strength
        self.banding_spin = QDoubleSpinBox()
        self.banding_spin.setRange(0.0, 1.0)
        self.banding_spin.setSingleStep(0.01)
        self.banding_spin.setValue(get("CameraSimulator", "banding_strength", 0.0, float))
        cam_layout.addRow("Banding strength", self.banding_spin)

        # Motion blur kernel
        self.motion_blur_spin = QSpinBox()
        self.motion_blur_spin.setRange(1, 51)
        self.motion_blur_spin.setValue(get("CameraSimulator", "motion_blur_kernel", 1, int))
        cam_layout.addRow("Motion blur kernel", self.motion_blur_spin)

        self.camera_box.setVisible(False)

        self.ref_hint = QLabel("AWB uses the 'AWB reference' chooser. FFT spectral matching uses the 'FFT Reference' chooser.")
        right_v.addWidget(self.ref_hint)

        self.analysis_input = AnalysisPanel(title="Input analysis")
        self.analysis_output = AnalysisPanel(title="Output analysis")
        right_v.addWidget(self.analysis_input)
        right_v.addWidget(self.analysis_output)

        right_v.addStretch(1)

        # Status bar
        self.status = QLabel("Ready")
        self.status.setStyleSheet("color:#bdbdbd;padding:6px")
        self.status.setAlignment(Qt.AlignLeft)
        self.status.setFixedHeight(28)
        self.status.setContentsMargins(6, 6, 6, 6)
        self.statusBar().addWidget(self.status)

        self.worker = None
        self._on_auto_mode_toggled(self.auto_mode_chk.checkState())

    def _on_sim_camera_toggled(self, state):
        enabled = state == Qt.Checked
        self.camera_box.setVisible(enabled)

    def _on_auto_mode_toggled(self, state):
        is_auto = (state == Qt.Checked)
        self.auto_box.setVisible(is_auto)
        self.params_box.setVisible(not is_auto)
        self.texture_box.setVisible(not is_auto)

    def _update_strength_label(self, value):
        self.strength_label.setText(str(value))

    def choose_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose input image", str(Path.home()), "Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if path:
            self.input_line.setText(path)
            self.load_preview(self.preview_in, path)
            self.analysis_input.update_from_path(path)
            out_suggest = str(Path(path).with_name(Path(path).stem + "_out" + Path(path).suffix))
            if not self.output_line.text():
                self.output_line.setText(out_suggest)

    def choose_ref(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose AWB reference image", str(Path.home()), "Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if path:
            self.ref_line.setText(path)

    def choose_fft_ref(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose FFT reference image", str(Path.home()), "Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if path:
            self.fft_ref_line.setText(path)

    def choose_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Choose output path", str(Path.home()), "JPEG (*.jpg *.jpeg);;PNG (*.png);;TIFF (*.tif)")
        if path:
            self.output_line.setText(path)

    def choose_lut(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose LUT file", str(Path.home()), "LUTs (*.png *.npy *.cube);;All files (*)")
        if path:
            self.lut_line.setText(path)

    def _on_lut_toggled(self, state):
        visible = (state == Qt.Checked)
        for w in self._lut_controls:
            w.setVisible(visible)

    def load_preview(self, widget: QLabel, path: str):
        if not path or not os.path.exists(path):
            widget.setText("No image")
            widget.setPixmap(QPixmap())
            return
        pix = qpixmap_from_path(path, max_size=(widget.width(), widget.height()))
        widget.setPixmap(pix)

    def set_enabled_all(self, enabled: bool):
        for w in self.findChildren((QPushButton, QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox, QCheckBox, QSlider, QToolButton)):
            w.setEnabled(enabled)

    def on_run(self):
        from types import SimpleNamespace
        inpath = self.input_line.text().strip()
        outpath = self.output_line.text().strip()
        if not inpath or not os.path.exists(inpath):
            QMessageBox.warning(self, "Missing input", "Please choose a valid input image.")
            return
        if not outpath:
            QMessageBox.warning(self, "Missing output", "Please choose an output path.")
            return

        awb_ref_val = self.ref_line.text() or None
        fft_ref_val = self.fft_ref_line.text() or None
        args = SimpleNamespace()

        if self.auto_mode_chk.isChecked():
            strength = self.strength_slider.value() / 100.0
            args.noise_std = strength * 0.04
            args.clahe_clip = 1.0 + strength * 3.0
            args.cutoff = max(0.01, 0.4 - strength * 0.3)
            args.fstrength = strength * 0.95
            args.phase_perturb = strength * 0.1
            # Update perturb: now separate flag and magnitude
            args.perturb = True
            args.perturb_magnitude = strength * 0.015
            args.jpeg_cycles = int(strength * 2)
            args.jpeg_qmin = max(1, int(95 - strength * 35))
            args.jpeg_qmax = max(1, int(99 - strength * 25))
            args.vignette_strength = strength * 0.6
            args.chroma_strength = strength * 4.0
            args.motion_blur_kernel = 1 + 2 * int(strength * 6)
            args.banding_strength = strength * 0.1
            args.tile = 8
            args.randomness = 0.05
            args.radial_smooth = 5
            args.fft_mode = "auto"
            args.fft_alpha = 1.0
            args.alpha = 1.0
            args.glcm = False
            args.glcm_distances = [1]
            args.glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            args.glcm_levels = 256
            args.glcm_strength = 0.9
            args.lbp = False
            args.lbp_radius = 3
            args.lbp_n_points = 24
            args.lbp_method = "uniform"
            args.lbp_strength = 0.9
            seed_val = int(self.seed_spin.value())
            args.seed = None if seed_val == 0 else seed_val
            args.sim_camera = bool(self.sim_camera_chk.isChecked())
            args.no_no_bayer = True
            args.iso_scale = 1.0
            args.read_noise = 2.0
            args.hot_pixel_prob = 1e-6
            args.clahe = True
            args.noise = True
            args.fft = True
        else:
            seed_val = int(self.seed_spin.value())
            args.seed = None if seed_val == 0 else seed_val
            sim_camera = bool(self.sim_camera_chk.isChecked())
            enable_bayer = bool(self.bayer_chk.isChecked())
            args.noise_std = float(self.noise_spin.value())
            args.clahe_clip = float(self.clahe_spin.value())
            args.tile = int(self.tile_spin.value())
            args.cutoff = float(self.cutoff_spin.value())
            args.fstrength = float(self.fstrength_spin.value())
            args.strength = float(self.fstrength_spin.value())
            args.randomness = float(self.randomness_spin.value())
            args.phase_perturb = float(self.phase_perturb_spin.value())
            args.fft_mode = self.fft_mode_combo.currentText()
            args.fft_alpha = float(self.fft_alpha_spin.value())
            args.alpha = float(self.fft_alpha_spin.value())
            args.radial_smooth = int(self.radial_smooth_spin.value())
            args.sim_camera = sim_camera
            args.no_no_bayer = bool(enable_bayer)
            args.jpeg_cycles = int(self.jpeg_cycles_spin.value())
            args.jpeg_qmin = int(self.jpeg_qmin_spin.value())
            args.jpeg_qmax = int(self.jpeg_qmax_spin.value())
            args.vignette_strength = float(self.vignette_spin.value())
            args.chroma_strength = float(self.chroma_spin.value())
            args.iso_scale = float(self.iso_spin.value())
            args.read_noise = float(self.read_noise_spin.value())
            args.hot_pixel_prob = float(self.hot_pixel_spin.value())
            args.banding_strength = float(self.banding_spin.value())
            args.motion_blur_kernel = int(self.motion_blur_spin.value())
            args.glcm = bool(self.glcm_chk.isChecked())
            args.glcm_distances = [int(x) for x in self.glcm_distances_line.text().split()]
            args.glcm_angles = [float(x) for x in self.glcm_angles_line.text().split()]
            args.glcm_levels = int(self.glcm_levels_spin.value())
            args.glcm_strength = float(self.glcm_strength_spin.value())
            args.lbp = bool(self.lbp_chk.isChecked())
            args.lbp_radius = int(self.lbp_radius_spin.value())
            args.lbp_n_points = int(self.lbp_n_points_spin.value())
            args.lbp_method = self.lbp_method_combo.currentText()
            args.lbp_strength = float(self.lbp_strength_spin.value())
            # Set the new optional processing flags based on checkboxes
            args.noise = self.noise_enable_chk.isChecked()
            args.clahe = self.clahe_enable_chk.isChecked()
            args.fft = self.fft_enable_chk.isChecked()
            args.perturb = self.perturb_enable_chk.isChecked()
            args.perturb_magnitude = float(self.perturb_spin.value())

        # AI Normalizer
        args.non_semantic = bool(self.ns_chk.isChecked())
        if args.non_semantic:
            try:
                import torch
            except ImportError:
                QMessageBox.warning(self, "Missing Dependency", "Torch (PyTorch) is required for AI Normalizer but is not installed.")
                self.set_enabled_all(True)
                return
            args.ns_iterations = int(self.ns_iterations_spin.value())
            args.ns_learning_rate = float(self.ns_lr_spin.value())
            args.ns_t_lpips = float(self.ns_t_lpips_spin.value())
            args.ns_t_l2 = float(self.ns_t_l2_spin.value())
            args.ns_c_lpips = float(self.ns_c_lpips_spin.value())
            args.ns_c_l2 = float(self.ns_c_l2_spin.value())
            args.ns_grad_clip = float(self.ns_grad_clip_spin.value())

        # AWB handling
        if self.awb_chk.isChecked():
            args.awb = True
            args.ref = awb_ref_val
        else:
            args.awb = False
            args.ref = None

        # FFT spectral matching reference
        args.fft_ref = fft_ref_val

        # LUT handling
        if self.lut_chk.isChecked():
            lut_path = self.lut_line.text().strip()
            args.lut = lut_path if lut_path else None
            args.lut_strength = float(self.lut_strength_spin.value())
        else:
            args.lut = None
            args.lut_strength = 1.0

        self.worker = Worker(inpath, outpath, args)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.started.connect(lambda: self.on_worker_started())
        self.worker.start()

        self.progress.setRange(0, 0)
        self.status.setText("Processing...")
        self.set_enabled_all(False)

    def on_worker_started(self):
        pass

    def on_finished(self, outpath):
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.status.setText("Done — saved to: " + outpath)
        self.load_preview(self.preview_out, outpath)
        self.analysis_output.update_from_path(outpath)
        self.set_enabled_all(True)

    def on_error(self, msg, traceback_text):
        from PyQt5.QtWidgets import QDialog, QTextEdit
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText("Error")

        dialog = QDialog(self)
        dialog.setWindowTitle("Processing Error")
        dialog.setMinimumSize(700, 480)
        layout = QVBoxLayout(dialog)

        error_label = QLabel(f"Error: {msg}")
        error_label.setWordWrap(True)
        layout.addWidget(error_label)

        traceback_edit = QTextEdit()
        traceback_edit.setReadOnly(True)
        traceback_edit.setText(traceback_text)
        traceback_edit.setStyleSheet("font-family: monospace; font-size: 12px;")
        layout.addWidget(traceback_edit)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)

        dialog.exec_()
        self.set_enabled_all(True)

    def open_output_folder(self):
        out = self.output_line.text().strip()
        if not out:
            QMessageBox.information(self, "No output", "No output path set yet.")
            return
        folder = os.path.dirname(os.path.abspath(out))
        if not os.path.exists(folder):
            QMessageBox.warning(self, "Not found", "Output folder does not exist: " + folder)
            return
        if sys.platform.startswith('darwin'):
            os.system(f'open "{folder}"')
        elif os.name == 'nt':
            os.startfile(folder)