
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QSplitter
from PyQt6.QtCore import Qt
from file_tree import FileTree
from input_panel import InputPanel
from terminal_panel import Terminal
from output_browser import OutputBrowser
from styles import apply_dark_theme
from command_management import CommandManager

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LUVIA")
        self.setGeometry(100, 100, 1400, 900)
        apply_dark_theme(self)

        # Create widgets
        self.file_tree = FileTree()
        self.input_panel = InputPanel()
        self.terminal = Terminal()
        self.output_browser = OutputBrowser()

        # Connect drag-and-drop signal
        self.file_tree.file_selected.connect(self.input_panel.set_input_file)

        # Create command manager and wire it to input panel
        self.command_manager = CommandManager(self.terminal)
        self.input_panel.run_button.clicked.connect(
            lambda: self.command_manager.execute_command(self.input_panel.build_command())
        )

        # Layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.file_tree)
        splitter.addWidget(self.input_panel)
        splitter.addWidget(self.terminal)

        layout = QVBoxLayout()
        layout.addWidget(splitter)
        layout.addWidget(self.output_browser)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
