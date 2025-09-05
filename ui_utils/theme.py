from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor

def apply_dark_palette(app: QApplication):
    pal = QPalette()
    # base
    pal.setColor(QPalette.Window, QColor(18, 18, 19))
    pal.setColor(QPalette.WindowText, QColor(220, 220, 220))
    pal.setColor(QPalette.Base, QColor(28, 28, 30))
    pal.setColor(QPalette.AlternateBase, QColor(24, 24, 26))
    pal.setColor(QPalette.ToolTipBase, QColor(220, 220, 220))
    pal.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    pal.setColor(QPalette.Text, QColor(230, 230, 230))
    pal.setColor(QPalette.Button, QColor(40, 40, 42))
    pal.setColor(QPalette.ButtonText, QColor(230, 230, 230))
    pal.setColor(QPalette.Highlight, QColor(70, 130, 180))
    pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)

    # global stylesheet for a modern gray look
    app.setStyleSheet(r"""
        QWidget { font-family: 'Segoe UI', Roboto, Arial, sans-serif; font-size:11pt }
        QToolButton { padding:6px; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #1e1e1f; border: 1px solid #333; padding:4px; border-radius:6px }
        QPushButton { background: #2a2a2c; border: 1px solid #3a3a3c; padding:6px 10px; border-radius:8px }
        QPushButton:hover { background: #333336 }
        QPushButton:pressed { background: #232325 }
        QProgressBar { background: #222; border: 1px solid #333; border-radius:6px; text-align:center }
        QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4b9bd6, stop:1 #3b83c0); }
        QLabel { color: #ffffff }
        QCheckBox { padding:4px }
        QGroupBox { color : #e6e6e6; }
        QGroupBox:title { color : #e6e6e6; }
    """)