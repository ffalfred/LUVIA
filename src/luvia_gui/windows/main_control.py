from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedLayout
from PyQt6.QtCore import QUrl
import os
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
from luvia_gui.components.input_panel import InputPanel
from luvia_gui.components.terminal_panel import Terminal
from luvia_gui.components.output_browser import OutputBrowser
from luvia_gui.components.file_tree import FileTree
from luvia_gui.windows.pdf_viewer_window import PDFViewerWindow
from luvia_gui.windows.image_viewer_window import ImageViewerWindow, ImageView
from luvia_gui.backend.command_management import CommandManager
from luvia_gui.components.loop_mode_view import LoopModeView
from luvia_gui.windows.json_viewer_window import HistoryView
from app_state import AppState
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtGui import QPixmap, QTransform

from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtGui import QMovie, QPixmap
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
import time
from luvia_gui.windows.pdf_image_combined_window import PDFImageCombinedWindow


class MainControlWindow(QMainWindow):
    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        self.setWindowTitle("LUVIA Control Panel")
#        self.setGeometry(100, 100, 1200, 800)
#        self.resize(1200,800)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
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


        logo_label = QLabel()
        logo_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..","gifs", "signal-2025-08-25-003555_003.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            logo_label.setPixmap(pixmap.scaledToHeight(60, Qt.TransformationMode.SmoothTransformation))
            logo_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        else:
            logo_label.setText("LUVIA")
            logo_label.setStyleSheet("color: white; font-size: 20px;")

        # Layouts
        top_layout = QHBoxLayout()
        top_layout.addWidget(logo_label)
        top_layout.addWidget(self.mode_label)
        top_layout.addWidget(self.mode_button)

        file_tree_layout = QVBoxLayout()

        self.open_combined_view_manual_button = QPushButton("Select PDF and Image")
        self.open_combined_view_manual_button.clicked.connect(self.select_pdf_and_image)
        file_tree_layout.addWidget(self.open_combined_view_manual_button)

        file_tree_layout.addWidget(self.file_tree_main)
        file_tree_layout.addWidget(self.file_tree_image1)
        file_tree_layout.addWidget(self.file_tree_image2)

        main_layout = QHBoxLayout()
        main_layout.addLayout(file_tree_layout, 2)
        main_layout.addWidget(self.input_panel, 3)
        main_layout.addWidget(self.terminal, 3)

        self.loop_view = LoopModeView()

        self.stack_layout = QStackedLayout()
        #self.stack_layout.addWidget(self.output_browser)  # Main mode view
        #self.stack_layout.addWidget(self.loop_view)       # Loop mode view

        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addLayout(main_layout)
        #layout.addLayout(self.stack_layout)
  ##      self.setLayout(layout)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

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


        self.spinner_pixmap =QPixmap(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "gifs", "signal-2025-08-23-160817_003.png"))
        self.spinner_label = QLabel()
        self.spinner_label.setFixedSize(80, 80)
        self.update_spinner_pixmap()
        self.spinner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        spinner_container = QHBoxLayout()
        spinner_container.addStretch()
        spinner_container.addWidget(self.spinner_label)
        spinner_container.addStretch()
        self.input_panel.layout().addLayout(spinner_container)

        self.rotation_angle = 0
        self.spinner_timer = QTimer(self)
        self.spinner_timer.timeout.connect(self.rotate_spinner)
        self.spinner_timer.setInterval(100)

        self.pdf_image_combined_window = PDFImageCombinedWindow(app_state)
        self.open_combined_view_button = QPushButton("Open PDF + Image Viewer")
        self.open_combined_view_button.clicked.connect(self.open_combined_view)
        # Add next to Run button
        self.input_panel.layout().addWidget(self.stop_loop_button)

        self.history_check_timer = None
        self.history_check_start_time = None

        self.command_manager = CommandManager(
            terminal=self.terminal,
            button=self.input_panel.run_button,
            spinner=None,
            stop_button=self.stop_button  # new
        )
        self.command_manager.command_finished_callback = self.on_command_finished


    def update_spinner_pixmap(self):
        scaled_pixmap = self.spinner_pixmap.scaled(
            self.spinner_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.spinner_label.setPixmap(scaled_pixmap)


    def on_command_finished(self):
        self.spinner_timer.stop()
        self.update_spinner_pixmap()

    def rotate_spinner(self):
        transform = QTransform().rotate(self.rotation_angle)
        rotated_pixmap = self.spinner_pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)
        rotated_scaled_pixmap = rotated_pixmap.scaled(
            self.spinner_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.spinner_label.setPixmap(rotated_scaled_pixmap)
        self.rotation_angle = (self.rotation_angle + 10) % 360

    def select_pdf_and_image(self):
        pdf_path, _ = QFileDialog.getOpenFileName(self, "Select PDF File", "", "PDF Files (*.pdf)")
        if not pdf_path:
            self.terminal.output.append("PDF selection cancelled.")
            return

        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        if not image_path:
            self.terminal.output.append("Image selection cancelled.")
            return

        self.show_combined_pdf_image_viewer(pdf_path, image_path)


    def check_for_history_file(self, history_path):
        elapsed = time.time() - self.history_check_start_time
        if os.path.isfile(history_path):
            self.json_viewer_window = HistoryView(history_path, terminal=self.terminal)
            screens = QApplication.screens()
            preferred_screen_index = 2
            if preferred_screen_index < len(screens):
                self.json_viewer_window.move(screens[preferred_screen_index].geometry().topLeft())
            self.json_viewer_window.show()
            self.history_check_timer.stop()
            self.terminal.output.append("History viewer launched.")
        elif elapsed > 600:  # 10 minutes
            self.history_check_timer.stop()
            self.terminal.output.append("Timeout: History file not found after 10 minutes.")

    def check_for_image_file(self, image_path):
        elapsed = time.time() - self.image_check_start_time
        current_file_folder = os.path.dirname(os.path.abspath(__file__))
        reference_path = os.path.abspath("{}/../data/2023_84_40_2_0143_00113914_small.jpeg".format(current_file_folder))
        if os.path.isfile(image_path):
            self.image_viewer_window = ImageView(dynamic_image_path=image_path,
                                                    reference_image_path=reference_path)
            screens = QApplication.screens()
            preferred_screen_index = 2
            if preferred_screen_index < len(screens):
                self.image_viewer_window.move(screens[preferred_screen_index].geometry().topLeft())
            self.image_viewer_window.show()
            self.image_check_timer.stop()
            self.terminal.output.append("Image viewer launched.")
        elif elapsed > 600:  # 10 minutes
            self.image_check_timer.stop()
            self.terminal.output.append("Timeout: Image file not found after 10 minutes.")

    def open_combined_view(self):
        output_folder = self.input_panel.output_folder_field.text()
        if not os.path.isdir(output_folder):
            self.terminal.output.append("Invalid output folder.")
            return

        # Find latest PDF and image
        pdf_files = [f for f in os.listdir(output_folder) if f.lower().endswith('.pdf')]
        image_files = [f for f in os.listdir(output_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        if not pdf_files or not image_files:
            self.terminal.output.append("No PDF or image found in output folder.")
            return

        pdf_files.sort(key=lambda f: os.path.getmtime(os.path.join(output_folder, f)), reverse=True)
        image_files.sort(key=lambda f: os.path.getmtime(os.path.join(output_folder, f)), reverse=True)

        pdf_path = os.path.join(output_folder, pdf_files[0])
        image_path = os.path.join(output_folder, image_files[0])

        self.pdf_image_combined_window.load_pdf_and_image(pdf_path, image_path)
        screens = QApplication.screens()
        preferred_screen_index = 2
        if preferred_screen_index < len(screens):
            self.pdf_image_combined_window.move(screens[preferred_screen_index].geometry().topLeft())
        self.pdf_image_combined_window.show()


    def show_combined_pdf_image_viewer(self, pdf_path: str, image_path: str):
        self.pdf_image_combined_window.load_pdf_and_image(pdf_path, image_path)
        screens = QApplication.screens()
        preferred_screen_index = 2
        if preferred_screen_index < len(screens):
            self.pdf_image_combined_window.move(screens[preferred_screen_index].geometry().topLeft())
        self.pdf_image_combined_window.show()

    def stop_command(self):
        self.spinner_timer.stop()
        self.update_spinner_pixmap()
        self.command_manager.stop_command()
        self.terminal.output.append("Command stopped.")


    def run_clicked(self):
        command = self.input_panel.build_command()
        if not command:
            return  # Error already shown in InputPanel

        self.terminal.output.append(f"Running command: {command}")
        self.spinner_timer.start()

        if self.app_state.get_mode() == "loop":
            input_folder = self.input_panel.input_folder_field.text()
            output_folder = self.input_panel.output_folder_field.text()
            history_path = os.path.join(output_folder, "LUVIA_history.jsonl")
            image_path = os.path.join(output_folder, "images/image-transformation.jpg") 
            self.app_state.history_path = history_path
            self.app_state.output_folder = output_folder
            self.command_manager.execute_command(command)

            self.history_check_start_time = time.time()
            self.history_check_timer = QTimer(self)
            self.history_check_timer.timeout.connect(lambda: self.check_for_history_file(history_path))
            self.history_check_timer.start(2000)  # Check every 2 seconds

            self.image_check_start_time = time.time()
            self.image_check_timer = QTimer(self)
            self.image_check_timer.timeout.connect(lambda: self.check_for_image_file(image_path))
            self.image_check_timer.start(2000)  # Check every 2 seconds

        else:
            self.command_manager.execute_command(command)

        output_folder = self.input_panel.output_folder_field.text()
        if os.path.isdir(output_folder):
            self.file_tree_main.set_root(output_folder)
            self.file_tree_image1.set_root(output_folder)
            self.file_tree_image2.set_root(output_folder)
            self.output_browser.refresh_output()


    def on_loop_finished(self):
        self.spinner_timer.stop()
        self.update_spinner_pixmap()
        self.stop_loop_button.setVisible(False)
        self.input_panel.run_button.setEnabled(True)


    def stop_loop_process(self):
        if self.backend_worker and self.backend_worker.isRunning():
            self.backend_worker.stop()
            self.terminal.output.append("Loop process terminated.")
            self.stop_loop_button.setVisible(False)
            self.image_viewer_window1.stop_auto_refresh()
            self.image_viewer_window2.stop_auto_refresh()
            self.spinner_timer.stop()
            self.update_spinner_pixmap()



    def toggle_mode(self):
        new_mode = "loop" if self.app_state.get_mode() == "main" else "main"
        self.app_state.set_mode(new_mode)

    def on_mode_changed(self, mode: str):
        self.mode_label.setText(f"Mode: {mode.capitalize()}")
        self.mode_button.setText("Switch to Main Mode" if mode == "loop" else "Switch to Loop Mode")
        #self.stack_layout.setCurrentIndex(1 if mode == "loop" else 0)
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
