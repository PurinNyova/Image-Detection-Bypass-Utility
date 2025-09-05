#!/usr/bin/env python3
"""
Worker thread for image processing.
"""

from PyQt5.QtCore import QThread, pyqtSignal
import traceback

try:
    from image_postprocess import process_image
except Exception:
    process_image = None
    IMPORT_ERROR = "Could not import process_image module"
else:
    IMPORT_ERROR = None

class Worker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str, str)  # error message + traceback

    def __init__(self, inpath, outpath, args):
        super().__init__()
        self.inpath = inpath
        self.outpath = outpath
        self.args = args

    def run(self):
        try:
            if process_image is None:
                raise RuntimeError("Could not import process_image: " + (IMPORT_ERROR or "unknown"))
            process_image(self.inpath, self.outpath, self.args)
            self.finished.emit(self.outpath)
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(str(e), tb)