
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit

class Terminal(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlainText("Terminal output will appear here.")
        layout.addWidget(self.output)
        self.setLayout(layout)
