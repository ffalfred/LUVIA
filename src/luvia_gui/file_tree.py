
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTreeView
from PyQt6.QtGui import QFileSystemModel
import os

class FileTree(QWidget):
    file_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.model = QFileSystemModel()
        self.model.setRootPath("")
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(os.path.expanduser("~")))
        self.tree.doubleClicked.connect(self.on_double_click)
        layout.addWidget(self.tree)
        self.setLayout(layout)

    def on_double_click(self, index):
        path = self.model.filePath(index)
        self.file_selected.emit(path)
