#!/usr/bin/env python3
"""
Main GUI application for image_postprocess pipeline with camera-simulator controls.

Updated: added LUT support UI (enable checkbox, file chooser, strength) and wiring to on_run.
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

try:
    from image_postprocess import process_image
except Exception as e:
    process_image = None
    IMPORT_ERROR = str(e)
else:
    IMPORT_ERROR = None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Postprocess — GUI (Camera Simulator)")
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
        self.fft_ref_btn = QPushButton("Choose FFT Reference (optional)")
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

        # Auto Mode toggle (keeps top-level quick switch visible)
        self.auto_mode_chk = QCheckBox("Enable Auto Mode")
        self.auto_mode_chk.setChecked(False)
        self.auto_mode_chk.stateChanged.connect(self._on_auto_mode_toggled)
        right_v.addWidget(self.auto_mode_chk)

        # Make Auto Mode section collapsible
        self.auto_box = CollapsibleBox("Auto Mode")
        right_v.addWidget(self.auto_box)
        auto_layout = QFormLayout()
        auto_container = QWidget()
        auto_container.setLayout(auto_layout)
        self.auto_box.content_layout.addWidget(auto_container)

        strength_layout = QHBoxLayout()
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(25)
        self.strength_slider.valueChanged.connect(self._update_strength_label)
        self.strength_label = QLabel("25")
        self.strength_label.setFixedWidth(30)
        strength_layout.addWidget(self.strength_slider)
        strength_layout.addWidget(self.strength_label)

        auto_layout.addRow("Aberration Strength", strength_layout)

        # Parameters (Manual Mode) collapsible
        self.params_box = CollapsibleBox("Parameters (Manual Mode)")
        right_v.addWidget(self.params_box)
        params_layout = QFormLayout()
        params_container = QWidget()
        params_container.setLayout(params_layout)
        self.params_box.content_layout.addWidget(params_container)

        # Noise-std
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 0.1)
        self.noise_spin.setSingleStep(0.001)
        self.noise_spin.setValue(0.02)
        self.noise_spin.setToolTip("Gaussian noise std fraction of 255")
        params_layout.addRow("Noise std (0-0.1)", self.noise_spin)

        # CLAHE-clip
        self.clahe_spin = QDoubleSpinBox()
        self.clahe_spin.setRange(0.1, 10.0)
        self.clahe_spin.setSingleStep(0.1)
        self.clahe_spin.setValue(2.0)
        params_layout.addRow("CLAHE clip", self.clahe_spin)

        # Tile
        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(1, 64)
        self.tile_spin.setValue(8)
        params_layout.addRow("CLAHE tile", self.tile_spin)

        # Cutoff
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0.01, 1.0)
        self.cutoff_spin.setSingleStep(0.01)
        self.cutoff_spin.setValue(0.25)
        params_layout.addRow("Fourier cutoff (0-1)", self.cutoff_spin)

        # Fstrength
        self.fstrength_spin = QDoubleSpinBox()
        self.fstrength_spin.setRange(0.0, 1.0)
        self.fstrength_spin.setSingleStep(0.01)
        self.fstrength_spin.setValue(0.9)
        params_layout.addRow("Fourier strength (0-1)", self.fstrength_spin)

        # Randomness
        self.randomness_spin = QDoubleSpinBox()
        self.randomness_spin.setRange(0.0, 1.0)
        self.randomness_spin.setSingleStep(0.01)
        self.randomness_spin.setValue(0.05)
        params_layout.addRow("Fourier randomness", self.randomness_spin)

        # Phase_perturb
        self.phase_perturb_spin = QDoubleSpinBox()
        self.phase_perturb_spin.setRange(0.0, 1.0)
        self.phase_perturb_spin.setSingleStep(0.001)
        self.phase_perturb_spin.setValue(0.08)
        self.phase_perturb_spin.setToolTip("Phase perturbation std (radians)")
        params_layout.addRow("Phase perturb (rad)", self.phase_perturb_spin)

        # Radial_smooth
        self.radial_smooth_spin = QSpinBox()
        self.radial_smooth_spin.setRange(0, 50)
        self.radial_smooth_spin.setValue(5)
        params_layout.addRow("Radial smooth (bins)", self.radial_smooth_spin)

        # FFT_mode
        self.fft_mode_combo = QComboBox()
        self.fft_mode_combo.addItems(["auto", "ref", "model"])
        self.fft_mode_combo.setCurrentText("auto")
        params_layout.addRow("FFT mode", self.fft_mode_combo)

        # FFT_alpha
        self.fft_alpha_spin = QDoubleSpinBox()
        self.fft_alpha_spin.setRange(0.1, 4.0)
        self.fft_alpha_spin.setSingleStep(0.1)
        self.fft_alpha_spin.setValue(1.0)
        self.fft_alpha_spin.setToolTip("Alpha exponent for 1/f model when using model mode")
        params_layout.addRow("FFT alpha (model)", self.fft_alpha_spin)

        # Perturb
        self.perturb_spin = QDoubleSpinBox()
        self.perturb_spin.setRange(0.0, 0.05)
        self.perturb_spin.setSingleStep(0.001)
        self.perturb_spin.setValue(0.008)
        params_layout.addRow("Pixel perturb", self.perturb_spin)

        # Seed
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2 ** 31 - 1)
        self.seed_spin.setValue(0)
        params_layout.addRow("Seed (0=none)", self.seed_spin)

        # AWB checkbox
        self.awb_chk = QCheckBox("Enable auto white-balance (AWB)")
        self.awb_chk.setChecked(False)
        self.awb_chk.setToolTip("If checked, AWB is applied. If a reference image is chosen, it will be used; otherwise gray-world AWB is applied.")
        params_layout.addRow(self.awb_chk)

        # Camera simulator toggle
        self.sim_camera_chk = QCheckBox("Enable camera pipeline simulation")
        self.sim_camera_chk.setChecked(False)
        self.sim_camera_chk.stateChanged.connect(self._on_sim_camera_toggled)
        params_layout.addRow(self.sim_camera_chk)

        # --- LUT support UI ---
        self.lut_chk = QCheckBox("Enable LUT")
        self.lut_chk.setChecked(False)
        self.lut_chk.setToolTip("Enable applying a 1D/.npy/.cube LUT to the output image")
        self.lut_chk.stateChanged.connect(self._on_lut_toggled)
        params_layout.addRow(self.lut_chk)

        # LUT chooser (hidden until checkbox checked)
        self.lut_line = QLineEdit()
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
        self.lut_strength_spin.setValue(1.0)
        self.lut_strength_spin.setToolTip("Blend strength for LUT (0.0 = no effect, 1.0 = full LUT)")
        self.lut_strength_label = QLabel("LUT strength")
        params_layout.addRow(self.lut_strength_label, self.lut_strength_spin)

        # Initially hide LUT controls and their labels
        self.lut_file_label.setVisible(False)
        lut_box.setVisible(False)
        self.lut_strength_label.setVisible(False)
        self.lut_strength_spin.setVisible(False)

        # Store all widgets that need their visibility toggled
        self._lut_controls = (self.lut_file_label, lut_box, self.lut_strength_label, self.lut_strength_spin)

        # Camera simulator collapsible group
        self.camera_box = CollapsibleBox("Camera simulator options")
        right_v.addWidget(self.camera_box)
        cam_layout = QFormLayout()
        cam_container = QWidget()
        cam_container.setLayout(cam_layout)
        self.camera_box.content_layout.addWidget(cam_container)

        # Enable bayer
        self.bayer_chk = QCheckBox("Enable Bayer / demosaic (RGGB)")
        self.bayer_chk.setChecked(True)
        cam_layout.addRow(self.bayer_chk)

        # JPEG cycles
        self.jpeg_cycles_spin = QSpinBox()
        self.jpeg_cycles_spin.setRange(0, 10)
        self.jpeg_cycles_spin.setValue(1)
        cam_layout.addRow("JPEG cycles", self.jpeg_cycles_spin)

        # JPEG quality min/max
        self.jpeg_qmin_spin = QSpinBox()
        self.jpeg_qmin_spin.setRange(1, 100)
        self.jpeg_qmin_spin.setValue(88)
        self.jpeg_qmax_spin = QSpinBox()
        self.jpeg_qmax_spin.setRange(1, 100)
        self.jpeg_qmax_spin.setValue(96)
        qbox = QHBoxLayout()
        qbox.addWidget(self.jpeg_qmin_spin)
        qbox.addWidget(QLabel("to"))
        qbox.addWidget(self.jpeg_qmax_spin)
        cam_layout.addRow("JPEG quality (min to max)", qbox)

        # Vignette strength
        self.vignette_spin = QDoubleSpinBox()
        self.vignette_spin.setRange(0.0, 1.0)
        self.vignette_spin.setSingleStep(0.01)
        self.vignette_spin.setValue(0.35)
        cam_layout.addRow("Vignette strength", self.vignette_spin)

        # Chromatic aberration strength
        self.chroma_spin = QDoubleSpinBox()
        self.chroma_spin.setRange(0.0, 10.0)
        self.chroma_spin.setSingleStep(0.1)
        self.chroma_spin.setValue(1.2)
        cam_layout.addRow("Chromatic aberration (px)", self.chroma_spin)

        # ISO scale
        self.iso_spin = QDoubleSpinBox()
        self.iso_spin.setRange(0.1, 16.0)
        self.iso_spin.setSingleStep(0.1)
        self.iso_spin.setValue(1.0)
        cam_layout.addRow("ISO/exposure scale", self.iso_spin)

        # Read noise
        self.read_noise_spin = QDoubleSpinBox()
        self.read_noise_spin.setRange(0.0, 50.0)
        self.read_noise_spin.setSingleStep(0.1)
        self.read_noise_spin.setValue(2.0)
        cam_layout.addRow("Read noise (DN)", self.read_noise_spin)

        # Hot pixel prob
        self.hot_pixel_spin = QDoubleSpinBox()
        self.hot_pixel_spin.setDecimals(9)
        self.hot_pixel_spin.setRange(0.0, 1.0)
        self.hot_pixel_spin.setSingleStep(1e-6)
        self.hot_pixel_spin.setValue(1e-6)
        cam_layout.addRow("Hot pixel prob", self.hot_pixel_spin)

        # Banding strength
        self.banding_spin = QDoubleSpinBox()
        self.banding_spin.setRange(0.0, 1.0)
        self.banding_spin.setSingleStep(0.01)
        self.banding_spin.setValue(0.0)
        cam_layout.addRow("Banding strength", self.banding_spin)

        # Motion blur kernel
        self.motion_blur_spin = QSpinBox()
        self.motion_blur_spin.setRange(1, 51)
        self.motion_blur_spin.setValue(1)
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
            args.perturb = strength * 0.015
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
            seed_val = int(self.seed_spin.value())
            args.seed = None if seed_val == 0 else seed_val
            args.sim_camera = bool(self.sim_camera_chk.isChecked())
            args.no_no_bayer = True
            args.iso_scale = 1.0
            args.read_noise = 2.0
            args.hot_pixel_prob = 1e-6
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
            args.perturb = float(self.perturb_spin.value())
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

        # AWB handling: only apply if checkbox is checked
        if self.awb_chk.isChecked():
            args.ref = awb_ref_val  # may be None -> the pipeline will fall back to gray-world
        else:
            args.ref = None

        # FFT spectral matching reference
        args.fft_ref = fft_ref_val

        # LUT handling: only include if LUT checkbox is checked and a path is provided
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
        else:
            os.system(f'xdg-open "{folder}"')

def main():
    app = QApplication([])
    apply_dark_palette(app)
    if IMPORT_ERROR:
        QMessageBox.critical(None, "Import error", "Could not import image_postprocess module:\n" + IMPORT_ERROR)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
