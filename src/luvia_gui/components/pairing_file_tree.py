
from PyQt6.QtCore import pyqtSignal
from luvia_gui.components.file_tree import FileTree

class PairingFileTree(FileTree):
    pair_selected = pyqtSignal(str, str)

    def __init__(self, output_folder: str = None):
        super().__init__(output_folder)
        self.pairing_mode = False
        self.first_file = None

    def enable_pairing_mode(self):
        self.pairing_mode = True
        self.first_file = None

    def on_double_click(self, index):
        path = self.model.filePath(index)
        if not self.pairing_mode:
            self.file_selected.emit(path)
        else:
            if self.first_file is None:
                self.first_file = path
            else:
                self.pair_selected.emit(self.first_file, path)
                self.first_file = None
