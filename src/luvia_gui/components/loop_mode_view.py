
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel
from PyQt6.QtGui import QPixmap

class LoopModeView(QWidget):
    def __init__(self):
        super().__init__()

        # Create layout
        layout = QVBoxLayout()

        # History JSON viewer
        self.history_view = QTextEdit()
        self.history_view.setReadOnly(True)
        self.history_view.setPlaceholderText("History JSON will appear here...")

        # Image labels
        self.image1 = QLabel("Image 1 placeholder")
        self.image2 = QLabel("Image 2 placeholder")

        # Add widgets to layout
        layout.addWidget(self.history_view)
        layout.addWidget(self.image1)
        layout.addWidget(self.image2)

        # Set layout
        self.setLayout(layout)
