
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QListWidget
import os

class OutputBrowser(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)
        self.setLayout(layout)
        self.refresh_output()

    def refresh_output(self):
        output_dir = os.path.expanduser("~/output")  # Change to your actual output path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.list_widget.clear()
        for filename in os.listdir(output_dir):
            self.list_widget.addItem(filename)
