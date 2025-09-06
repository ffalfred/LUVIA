
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QCheckBox, QFormLayout, QMessageBox, QTabWidget,
    QHBoxLayout, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
import os
from luvia_gui.config.options_config import categories
from luvia_gui.backend.backend_worker import BackendWorker

class InputPanel(QWidget):
    output_folder_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.inputs = {}
        self.run_button = QPushButton("Run")
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.input_file_field = QLineEdit()
        self.input_file_field.setPlaceholderText("Select input file")
        self.input_file_field.setReadOnly(True)
        self.file_button = QPushButton("Browse")
        self.file_button.clicked.connect(self.select_input_file)

        file_layout = QHBoxLayout()
        file_layout.addWidget(self.input_file_field)
        file_layout.addWidget(self.file_button)

        layout.addWidget(QLabel("Input File"))
        layout.addLayout(file_layout)

        self.input_folder_field = QLineEdit()
        self.input_folder_field.setPlaceholderText("Select input folder")
        self.input_folder_field.setReadOnly(True)
        self.input_folder_field.setVisible(False)
        self.folder_button = QPushButton("Browse")
        self.folder_button.setVisible(False)
        self.folder_button.clicked.connect(self.select_input_folder)

        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.input_folder_field)
        folder_layout.addWidget(self.folder_button)

        layout.addWidget(QLabel("Input Folder"))
        layout.addLayout(folder_layout)

        output_layout = QHBoxLayout()
        self.output_folder_field = QLineEdit()
        self.output_folder_field.setPlaceholderText("Select output folder")
        output_button = QPushButton("Browse")
        output_button.clicked.connect(self.select_output_folder)
        output_layout.addWidget(self.output_folder_field)
        output_layout.addWidget(output_button)

        layout.addWidget(QLabel("Output Folder"))
        layout.addLayout(output_layout)

        tabs = QTabWidget()
        for category, options in categories.items():
            tab = QWidget()
            form_layout = QFormLayout()
            for label, (flag, widget_type, default, description) in options.items():
                if widget_type == "entry":
                    input_widget = QLineEdit()
                    input_widget.setText(default)
                elif widget_type == "dropdown":
                    input_widget = QComboBox()
                    input_widget.addItems(default)
                elif widget_type == "checkbox":
                    input_widget = QCheckBox()
                    input_widget.setChecked(False if default in ["", None, "False"] else True)
                else:
                    continue
                form_layout.addRow(QLabel(f"{label} ({flag})"), input_widget)
                self.inputs[flag] = input_widget
            tab.setLayout(form_layout)
            tabs.addTab(tab, category)

        layout.addWidget(tabs)
        layout.addWidget(self.run_button)
        self.setLayout(layout)

        #self.run_button.clicked.connect(self.run_backend)

    def select_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input File", "", "All Files (*)")
        if file_path:
            self.input_file_field.setText(file_path)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_field.setText(folder)
            self.output_folder_changed.emit(folder)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_folder_field.setText(folder)

    def get_folder_input(self):
        return self.input_folder_field.text()

    def set_mode(self, mode: str):
        is_loop = mode.lower() == "loop"
        self.input_file_field.setVisible(not is_loop)
        self.file_button.setVisible(not is_loop)
        self.input_folder_field.setVisible(is_loop)
        self.folder_button.setVisible(is_loop)
        self.findChild(QTabWidget).setVisible(not is_loop)
        self.run_button.setText("Run Loop" if is_loop else "Run Main")

    def run_backend(self):
        input_folder = self.input_folder_field.text()
        output_folder = self.output_folder_field.text()

        if not os.path.isdir(input_folder):
            QMessageBox.warning(self, "Missing Input Folder", "Please select a valid input folder.")
            return
        if not os.path.isdir(output_folder):
            QMessageBox.warning(self, "Missing Output Folder", "Please select a valid output folder.")
            return

        self.worker = BackendWorker(input_folder, output_folder)
        self.worker.output_signal.connect(lambda msg: print(msg))
        self.worker.error_signal.connect(lambda msg: print(msg))
        self.worker.finished_signal.connect(self.on_worker_finished)
        self.worker.start()

        self.run_button.setEnabled(False)

    def stop_backend(self):
        if self.worker:
            self.worker.stop()
            self.run_button.setEnabled(True)

    def on_worker_finished(self):
        self.run_button.setEnabled(True)

    def build_command(self):
        args = []
        if self.input_file_field.isVisible():  # Main mode
            input_file = self.input_file_field.text()
            output_folder = self.output_folder_field.text()
            if input_file and os.path.isfile(input_file):
                args.append(f"--input '{input_file}'")
            else:
                QMessageBox.warning(self, "Missing Input File", "Please select a valid input file.")
                return ""
            if output_folder and os.path.isdir(output_folder):
                args.append(f"--output '{output_folder}'")
            else:
                QMessageBox.warning(self, "Missing Output Folder", "Please select a valid output folder.")
                return ""
            for flag, widget in self.inputs.items():
                if isinstance(widget, QLineEdit):
                    value = widget.text()
                elif isinstance(widget, QComboBox):
                    value = widget.currentText()
                elif isinstance(widget, QCheckBox):
                    value = widget.isChecked()
                    if value:
                        args.append(flag)
                        continue
                    else:
                        continue
                else:
                    continue
                if value != "" and value is not None:
                    args.append(f"{flag} {value}")
            command = f"luvia main {' '.join(args)}"
        else:  # Loop mode
            input_folder = self.input_folder_field.text()
            if input_folder and os.path.isdir(input_folder):
                args.append(f"--folder_streets '{input_folder}'")
            else:
                QMessageBox.warning(self, "Missing Input Folder", "Please select a valid input folder.")
                return ""
            command = f"luvia horde {' '.join(args)}"
            output_folder = self.output_folder_field.text()
            if output_folder and os.path.isdir(output_folder):
                command += f" --output {output_folder}"
            else:
                QMessageBox.warning(self, "Missing Output Folder", "Please select a valid output folder.")
                return ""
        return command
