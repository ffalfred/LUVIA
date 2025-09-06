from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedLayout
from PyQt6.QtCore import QUrl
import os
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
from luvia_gui.components.input_panel import InputPanel
from luvia_gui.components.terminal_panel import Terminal
from luvia_gui.components.output_browser import OutputBrowser
from luvia_gui.components.file_tree import FileTree
from luvia_gui.windows.pdf_viewer_window import PDFViewerWindow
from luvia_gui.windows.image_viewer_window import ImageViewerWindow
from luvia_gui.backend.command_management import CommandManager
from luvia_gui.components.loop_mode_view import LoopModeView
from luvia_gui.windows.json_viewer_window import JsonViewerWindow
from app_state import AppState
from PyQt6.QtWidgets import QApplication

class MainControlWindow(QWidget):
    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        self.setWindowTitle("LUVIA Control Panel")
        self.setGeometry(100, 100, 1200, 800)
        
        # Top controls
        self.mode_label = QLabel("Mode: Main")
        self.mode_button = QPushButton("Switch to Loop Mode")
        self.mode_button.clicked.connect(self.toggle_mode)

        # Viewer windows
        self.pdf_viewer_window = PDFViewerWindow(app_state)
        self.image_viewer_window1 = ImageViewerWindow(app_state)
        self.image_viewer_window2 = ImageViewerWindow(app_state)

        # Core components
        self.input_panel = InputPanel()
        self.terminal = Terminal()
        self.output_browser = OutputBrowser()

        # File trees
        self.file_tree_main = FileTree()
        self.file_tree_image1 = FileTree()
        self.file_tree_image2 = FileTree()

        # Set initial root
        initial_output = app_state.get_output_folder()
        self.file_tree_main.set_root(initial_output)
        self.file_tree_image1.set_root(initial_output)
        self.file_tree_image2.set_root(initial_output)

        # Connect file trees to respective viewers
        self.file_tree_main.file_selected.connect(self.on_file_selected_pdf)
        self.file_tree_image1.file_selected.connect(self.on_file_selected_image1)
        self.file_tree_image2.file_selected.connect(self.on_file_selected_image2)

        # Connect output folder signal
        self.input_panel.output_folder_changed.connect(self.file_tree_main.set_root)
        self.input_panel.output_folder_changed.connect(self.output_browser.refresh_output)

        self.input_panel.output_folder_changed.connect(self.file_tree_image1.set_root)
        self.input_panel.output_folder_changed.connect(self.file_tree_image2.set_root)

        # Layouts
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.mode_label)
        top_layout.addWidget(self.mode_button)

        file_tree_layout = QVBoxLayout()
        file_tree_layout.addWidget(self.file_tree_main)
        file_tree_layout.addWidget(self.file_tree_image1)
        file_tree_layout.addWidget(self.file_tree_image2)

        main_layout = QHBoxLayout()
        main_layout.addLayout(file_tree_layout, 2)
        main_layout.addWidget(self.input_panel, 3)
        main_layout.addWidget(self.terminal, 3)

        self.loop_view = LoopModeView()

        self.stack_layout = QStackedLayout()
        self.stack_layout.addWidget(self.output_browser)  # Main mode view
        self.stack_layout.addWidget(self.loop_view)       # Loop mode view

        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addLayout(main_layout)
        layout.addLayout(self.stack_layout)
        self.setLayout(layout)

        # React to AppState changes
        self.app_state.mode_changed.connect(self.on_mode_changed)
        self.app_state.input_folder_changed.connect(self.input_panel.select_input_folder)
        self.app_state.output_folder_changed.connect(self.input_panel.select_output_folder)

        # Run button connection
        self.input_panel.run_button.clicked.connect(self.run_clicked)

        self.json_viewer_window = None
        self.backend_worker = None

        self.stop_loop_button = QPushButton("Stop")
        self.stop_loop_button.setVisible(False)
        self.stop_loop_button.clicked.connect(self.stop_loop_process)
        self.input_panel.layout().addWidget(self.stop_loop_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_command)
        self.input_panel.layout().addWidget(self.stop_button)

        # Add next to Run button
        self.input_panel.layout().addWidget(self.stop_loop_button)



        self.command_manager = CommandManager(
            terminal=self.terminal,
            button=self.input_panel.run_button,
            spinner=None,
            stop_button=self.stop_button  # new
        )


    def stop_command(self):
        self.command_manager.stop_command()
        self.terminal.output.append("Command stopped.")


    def run_clicked(self):
        command = self.input_panel.build_command()
        if not command:
            return  # Error already shown in InputPanel

        self.terminal.output.append(f"Running command: {command}")

        if self.app_state.get_mode() == "loop":
            input_folder = self.input_panel.input_folder_field.text()
            output_folder = self.input_panel.output_folder_field.text()
            self.command_manager.execute_command(command)


            history_path = os.path.join(output_folder, "LUVIA_history.jsonl")
            if os.path.isfile(history_path):
                self.json_viewer_window = JsonViewerWindow(history_path)
                screens = QApplication.screens()
                preferred_screen_index = 2
                if preferred_screen_index < len(screens):
                    self.json_viewer_window.move(screens[preferred_screen_index].geometry().topLeft())
                self.json_viewer_window.show()

                image_path = os.path.join(output_folder, "images/image-transformation.jpg")  # or whatever filename you expect
                if os.path.isfile(image_path):
                    self.image_viewer_window1.start_auto_refresh(image_path)
                    self.image_viewer_window2.start_auto_refresh(image_path)

                    screens = QApplication.screens()
                    if len(screens) > 2:
                        self.image_viewer_window1.move(screens[2].geometry().topLeft())
                    if len(screens) > 3:
                        self.image_viewer_window2.move(screens[3].geometry().topLeft())

                    self.image_viewer_window1.show()
                    self.image_viewer_window2.show()

        else:
            self.command_manager.execute_command(command)

        output_folder = self.input_panel.output_folder_field.text()
        if os.path.isdir(output_folder):
            self.file_tree_main.set_root(output_folder)
            self.file_tree_image1.set_root(output_folder)
            self.file_tree_image2.set_root(output_folder)
            self.output_browser.refresh_output()


    def on_loop_finished(self):
        self.stop_loop_button.setVisible(False)
        self.input_panel.run_button.setEnabled(True)


    def stop_loop_process(self):
        if self.backend_worker and self.backend_worker.isRunning():
            self.backend_worker.stop()
            self.terminal.output.append("Loop process terminated.")
            self.stop_loop_button.setVisible(False)
            self.image_viewer_window1.stop_auto_refresh()
            self.image_viewer_window2.stop_auto_refresh()



    def toggle_mode(self):
        new_mode = "loop" if self.app_state.get_mode() == "main" else "main"
        self.app_state.set_mode(new_mode)

    def on_mode_changed(self, mode: str):
        self.mode_label.setText(f"Mode: {mode.capitalize()}")
        self.mode_button.setText("Switch to Main Mode" if mode == "loop" else "Switch to Loop Mode")
        self.stack_layout.setCurrentIndex(1 if mode == "loop" else 0)
        self.input_panel.set_mode(mode)


    def on_file_selected_pdf(self, path: str):
        if path.lower().endswith(".pdf"):
            self.pdf_viewer_window.load_pdf(path)

            # Force it to open on screen 2 if available
            screens = QApplication.screens()
            preferred_screen_index = 2
            if preferred_screen_index < len(screens):
                self.pdf_viewer_window.move(screens[preferred_screen_index].geometry().topLeft())
            else:
                print(f"Preferred screen {preferred_screen_index} not available. Using default.")
            
            self.pdf_viewer_window.show()


    def on_file_selected_image1(self, path: str):
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            self.image_viewer_window1.load_image(path)

            # Move to screen 3 if available
            screens = QApplication.screens()
            preferred_screen_index = 2  # screen 3
            if preferred_screen_index < len(screens):
                self.image_viewer_window1.move(screens[preferred_screen_index].geometry().topLeft())
            else:
                print("Screen 3 not available. Using default.")

            self.image_viewer_window1.show()


    def on_file_selected_image2(self, path: str):
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            self.image_viewer_window2.load_image(path)

            # Move to screen 4 if available
            screens = QApplication.screens()
            preferred_screen_index = 3  # screen 4
            if preferred_screen_index < len(screens):
                self.image_viewer_window2.move(screens[preferred_screen_index].geometry().topLeft())
            else:
                print("Screen 4 not available. Using default.")

            self.image_viewer_window2.show()
