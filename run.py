#!/usr/bin/env python3
"""
Entry point for Image Postprocess GUI (camera simulator).
Handles the import check for image_postprocess and launches the MainWindow.
"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMessageBox
from theme import apply_dark_palette

try:
    from image_postprocess import process_image
except Exception as e:
    IMPORT_ERROR = str(e)
else:
    IMPORT_ERROR = None

from main_window import MainWindow

def main():
    app = QApplication([])
    apply_dark_palette(app)

    if IMPORT_ERROR:
        QMessageBox.critical(None, "Import error",
                             "Could not import image_postprocess module:\n" + IMPORT_ERROR)

    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
