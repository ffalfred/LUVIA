import os
from PyQt6.QtWidgets import QMainWindow, QApplication

from app_state import AppState
from luvia_gui.windows.image_viewer_window import ImageView
from luvia_gui.windows.json_viewer_window import HistoryView


class WindowManager:
    """Owns auxiliary windows shown on extra screens.

    Reacts to AppState mode/output-folder changes by closing the current
    auxiliary windows and rebuilding them for the new state. MainControlWindow
    owns its own viewers (on-demand from file-tree clicks); the auxiliary
    windows here are the loop-mode live HistoryView and image comparison view.
    """

    def __init__(self, app_state: AppState):
        self.app_state = app_state
        self.windows = []
        self.screens = []
        self.app_state.mode_changed.connect(self._refresh)
        self.app_state.output_folder_changed.connect(self._refresh)

    def launch_windows(self, screens):
        self.screens = screens
        self._refresh()

    def _refresh(self, _=None):
        for win in self.windows:
            win.close()
        self.windows.clear()

        if self.app_state.get_mode() == "loop":
            output_folder = self.app_state.get_output_folder()
            if output_folder:
                self.windows.append(self._build_history_window(output_folder))
                self.windows.append(self._build_image_window(output_folder))

        screens = self.screens or [QApplication.primaryScreen()]
        for i, win in enumerate(self.windows):
            if i < len(screens):
                win.move(screens[i].geometry().topLeft())
            win.show()

    def _build_history_window(self, output_folder):
        history_path = os.path.join(output_folder, "LUVIA_history.jsonl")
        win = QMainWindow()
        win.setWindowTitle("History Viewer")
        win.setCentralWidget(HistoryView(history_path))
        return win

    def _build_image_window(self, output_folder):
        image_path = os.path.join(output_folder, "images", "image-transformation.jpg")
        reference_path = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "2023_84_40_2_0143_00113914_small.jpeg"))
        win = QMainWindow()
        win.setWindowTitle("Image Comparison")
        win.setCentralWidget(ImageView(
            reference_image_path=reference_path,
            dynamic_image_path=image_path,
        ))
        return win
