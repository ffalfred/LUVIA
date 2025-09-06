
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTreeView
from PyQt6.QtGui import QFileSystemModel
from PyQt6.QtWidgets import QComboBox
from PyQt6.QtCore import Qt
import os


class FileTree(QWidget):
    file_selected = pyqtSignal(str)

    def __init__(self, output_folder: str = None):
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
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(3, Qt.SortOrder.DescendingOrder)  # 3 = Date Modified

        # Inside __init__:
        self.sort_dropdown = QComboBox()
        self.sort_dropdown.addItems(["Name", "Size", "Type", "Date Modified"])
        self.sort_dropdown.currentIndexChanged.connect(self.sort_tree)
        layout.addWidget(self.sort_dropdown)

        if output_folder and os.path.isdir(output_folder):
            self.set_root(output_folder)

    def sort_tree(self, index):
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(index, Qt.SortOrder.AscendingOrder)

    def on_double_click(self, index):
        path = self.model.filePath(index)
        self.file_selected.emit(path)

    def set_root(self, path: str):
        if os.path.isdir(path):
            self.model.setRootPath(path)
            self.tree.setRootIndex(self.model.index(path))  # ✅


