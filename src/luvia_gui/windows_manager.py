from PyQt6.QtWidgets import QMainWindow, QWidget, QLabel, QVBoxLayout, QApplication
from app_state import AppState
from luvia_gui.windows.image_viewer_window import ImageViewerWindow
from luvia_gui.windows.pdf_viewer_window import PDFViewerWindow
from luvia_gui.windows.json_viewer_window import HistoryView
import os

class HistoryWindow(QMainWindow):
    def __init__(self, jsonl_path: str):
        super().__init__()
        self.setWindowTitle("History Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.view = HistoryView(jsonl_path)
        self.setCentralWidget(self.view)


class WindowManager:
    def __init__(self, app_state: AppState):
        self.app_state = app_state
        self.windows = []
        self.screens = []
        self.pdf_viewer_window = PDFViewerWindow(app_state)
        self.app_state.mode_changed.connect(self.relaunch_windows)

    def launch_windows(self, screens):
        self.screens = screens
        self.relaunch_windows(self.app_state.get_mode())

    def relaunch_windows(self, mode):
        for win in self.windows:
            win.close()
        self.windows.clear()

        if mode == "loop":
            #history_path = self.app_state.get_history_path()
            #self.windows.append(HistoryWindow(history_path))
            #image_path = os.path.join(os.path.basename(history_path), "images/image-transformation.jpg")
            #viewer_a = ImageViewerWindow(self.app_state, title="Image Viewer A")
            #viewer_a.start_auto_refresh(image_path)
            #self.windows.append(viewer_a)

            pass
        else:
            #self.windows.append(self.pdf_viewer_window)
            #self.windows.append(ImageViewerWindow(self.app_state, title="Image Viewer A"))
            #self.windows.append(ImageViewerWindow(self.app_state, title="Image Viewer B"))
            pass
        if not self.screens:
            self.screens = [QApplication.primaryScreen()]

        for i, win in enumerate(self.windows):
            if i < len(self.screens):
                win.move(self.screens[i].geometry().topLeft())
            win.show()