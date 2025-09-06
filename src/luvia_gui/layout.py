
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QSplitter, QLabel
from PyQt6.QtGui import QMovie
from PyQt6.QtCore import Qt
from file_tree import FileTree
from input_panel import InputPanel
from terminal_panel import Terminal
from output_browser import OutputBrowser
from styles import apply_dark_theme
from command_management import CommandManager
import os



from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter, QLabel,
    QStackedWidget, QPushButton, QHBoxLayout,QComboBox,
)
from PyQt6.QtGui import QMovie
from PyQt6.QtCore import Qt
from file_tree import FileTree
from input_panel import InputPanel
from terminal_panel import Terminal
from output_browser import OutputBrowser
from styles import apply_dark_theme
from command_management import CommandManager
from history_view import HistoryView  # <-- Loop mode GUI
import os


class MainLUVIAView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.file_tree = FileTree()
        self.input_panel = InputPanel()
        self.terminal = Terminal()
        self.output_browser = OutputBrowser()

        self.file_tree.file_selected.connect(self.input_panel.set_input_file)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.file_tree)
        splitter.addWidget(self.input_panel)
        splitter.addWidget(self.terminal)

        self.spinner_label = QLabel()
        self.spinner_movie = QMovie("{}/data/GiFS/signal-2025-08-25-003555_006".format(os.path.dirname(os.path.realpath(__file__))))
        self.spinner_label.setMovie(self.spinner_movie)
        self.spinner_movie.start()
        self.spinner_label.setVisible(False)

        # Main layout
        layout = QVBoxLayout()
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LUVIA")
        self.setGeometry(100, 100, 1400, 900)
        apply_dark_theme(self)


        self.current_mode = "main"  # default mode

        # Mode selector buttons
        self.mode_selector = QWidget()
        selector_layout = QHBoxLayout()
        self.main_button = QPushButton("Main Mode")
        self.loop_button = QPushButton("Loop Mode")
        self.main_button.clicked.connect(lambda: self.set_mode("main"))
        self.loop_button.clicked.connect(lambda: self.set_mode("loop"))
        selector_layout.addWidget(self.main_button)
        selector_layout.addWidget(self.loop_button)
        self.mode_selector.setLayout(selector_layout)

        # Views
        self.stack = QStackedWidget()
        self.main_view = MainLUVIAView()
        self.loop_view = LoopLUVIAView()
        self.stack.addWidget(self.main_view)
        self.stack.addWidget(self.loop_view)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.mode_selector)
        layout.addWidget(self.stack)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)


    def show_main(self):
        self.stack.setCurrentWidget(self.main_view)

    def show_loop(self):
        self.stack.setCurrentWidget(self.loop_view)

    def set_mode(self, mode):
        self.current_mode = mode
        self.main_view.input_panel.set_mode(mode)  # Update input panel visibility

        # Show the correct view
        if mode == "main":
            self.show_main()
        elif mode == "loop":
            self.show_loop()

        # Send feedback to terminal (only if in main view)
        if hasattr(self.main_view, "terminal"):
            self.main_view.terminal.output.append(f"Switched to {mode} mode.\n")




class LoopLUVIAView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.file_tree = FileTree()
        self.terminal = Terminal()
        self.history_view = HistoryView(terminal=self.terminal)

        # Connect file selection to history view input
        self.file_tree.file_selected.connect(self.handle_file_selection)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.file_tree)
        splitter.addWidget(self.history_view)
        splitter.addWidget(self.terminal)

        layout = QVBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)

    def handle_file_selection(self, path: str):
        if os.path.isdir(path):
            self.history_view.set_input_folder(path)
        else:
            # Optional: handle file selection differently or ignore
            pass
