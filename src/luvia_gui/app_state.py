
from PyQt6.QtCore import QObject, pyqtSignal

class AppState(QObject):
    # Signals to notify other components of changes
    mode_changed = pyqtSignal(str)
    input_folder_changed = pyqtSignal(str)
    output_folder_changed = pyqtSignal(str)
    run_requested = pyqtSignal(str, str)  # mode, input_folder

    def __init__(self):
        super().__init__()
        self._mode = "main"
        self._input_folder = ""
        self._output_folder = ""

    # Mode
    def set_mode(self, mode: str):
        if mode != self._mode:
            self._mode = mode
            self.mode_changed.emit(mode)

    def get_mode(self) -> str:
        return self._mode

    # Input folder
    def set_input_folder(self, folder: str):
        if folder != self._input_folder:
            self._input_folder = folder
            self.input_folder_changed.emit(folder)

    def get_input_folder(self) -> str:
        return self._input_folder

    # Output folder
    def set_output_folder(self, folder: str):
        if folder != self._output_folder:
            self._output_folder = folder
            self.output_folder_changed.emit(folder)

    def get_output_folder(self) -> str:
        return self._output_folder
