from PyQt6.QtWidgets import QMainWindow, QWidget, QLabel, QVBoxLayout, QApplication
from app_state import AppState
from luvia_gui.windows.image_viewer_window import ImageViewerWindow
from luvia_gui.windows.pdf_viewer_window import PDFViewerWindow

class HistoryWindow(QMainWindow):
    def __init__(self, app_state: AppState):
        super().__init__()
        self.setWindowTitle("History Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.view = HistoryView(app_state)
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
            #self.windows.append(HistoryWindow(self.app_state))
            self.windows.append(ImageViewerWindow(self.app_state, title="Image Viewer A"))
            self.windows.append(ImageViewerWindow(self.app_state, title="Image Viewer B"))
        else:
            self.windows.append(self.pdf_viewer_window)
            self.windows.append(ImageViewerWindow(self.app_state, title="Image Viewer A"))
            self.windows.append(ImageViewerWindow(self.app_state, title="Image Viewer B"))

        if not self.screens:
            self.screens = [QApplication.primaryScreen()]

        for i, win in enumerate(self.windows):
            if i < len(self.screens):
                win.move(self.screens[i].geometry().topLeft())
            win.show()