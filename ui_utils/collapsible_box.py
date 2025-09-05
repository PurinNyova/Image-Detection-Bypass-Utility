from PyQt5.QtWidgets import QWidget, QToolButton, QVBoxLayout
from PyQt5.QtCore import Qt

class CollapsibleBox(QWidget):
    """A simple collapsible container widget with a chevron arrow."""
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.toggle = QToolButton(text=title, checkable=True, checked=True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.DownArrow)
        self.toggle.clicked.connect(self.on_toggled)
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight:600; padding:6px; }")

        self.content = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(8, 4, 8, 8)
        self.content.setLayout(self.content_layout)

        lay = QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle)
        lay.addWidget(self.content)

    def on_toggled(self):
        checked = self.toggle.isChecked()
        self.toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content.setVisible(checked)