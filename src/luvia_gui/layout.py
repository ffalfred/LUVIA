from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter, QLabel,
    QStackedWidget, QPushButton, QHBoxLayout
)
from PyQt6.QtGui import QMovie, QPixmap
from PyQt6.QtCore import Qt
from luvia_gui.components.input_panel import InputPanel
from luvia_gui.components.terminal_panel import Terminal
from luvia_gui.components.output_browser import OutputBrowser
from styles import apply_dark_theme
from luvia_gui.backend.command_management import CommandManager
from history_view import HistoryView
from style_helpers import style_button, apply_fonts
import os

class MainLUVIAView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.input_panel = InputPanel()
        self.terminal = Terminal()
        self.output_browser = OutputBrowser()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStretchFactor(0, 2)  # Input panel
        splitter.setStretchFactor(1, 1)  # Terminal
        splitter.addWidget(self.input_panel)
        splitter.addWidget(self.terminal)

        self.spinner_label = QLabel()
        spinner_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gifs", "signal-2025-08-25-003555_006.png")
        print(spinner_path)
        self.spinner_movie = QMovie(spinner_path)
        self.spinner_label.setMovie(self.spinner_movie)
        self.spinner_movie.start()
        self.spinner_label.setFixedSize(100, 100)
        self.spinner_label.setVisible(True)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(splitter)
        layout.addWidget(self.output_browser)
        layout.addWidget(self.spinner_label)
        self.setLayout(layout)

        self.command_manager = CommandManager(
            terminal=self.terminal,
            button=self.input_panel.run_button,
            spinner=self.spinner_label
        )
        self.input_panel.run_button.clicked.connect(self.run_based_on_mode)

    def run_based_on_mode(self):
        mode = self.parent().current_mode if hasattr(self.parent(), "current_mode") else "main"
        if mode == "main":
            command = self.input_panel.build_command()
            if command:
                self.command_manager.execute_command(command)
        elif mode == "loop":
            folder_path = self.input_panel.get_folder_input()
            output_path = self.input_panel.output_folder_field.text()
            if folder_path and os.path.isdir(folder_path) and output_path and os.path.isdir(output_path):
                command = f"luvia loop --input_folder '{folder_path}' --output '{output_path}'"
                self.command_manager.execute_command(command)
            else:
                self.terminal.output.append("Please select valid input and output folders for Loop mode.\n")

class LoopLUVIAView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.terminal = Terminal()
        self.history_view = HistoryView(terminal=self.terminal)
        apply_dark_theme(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.addWidget(self.history_view)
        splitter.addWidget(self.terminal)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(splitter)
        self.setLayout(layout)

