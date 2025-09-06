
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
        output_dir = getattr(self, "output_dir", os.path.expanduser("~/output"))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.list_widget.clear()
        for filename in os.listdir(output_dir):
            self.list_widget.addItem(filename)


    def set_output_directory(self, path: str):
        self.output_dir = path
        self.refresh_output()
