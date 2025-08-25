
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QCheckBox, QFormLayout, QMessageBox, QTabWidget,
    QHBoxLayout, QFileDialog
)
from PyQt6.QtCore import Qt
from options_config import categories
import os

class InputPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.inputs = {}
        self.run_button = QPushButton("Run")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Input File field with drag-and-drop
        self.input_file_field = QLineEdit()
        self.input_file_field.setPlaceholderText("Drop input file here")
        self.input_file_field.setReadOnly(True)
        self.input_file_field.setAcceptDrops(True)
        self.input_file_field.installEventFilter(self)

        layout.addWidget(QLabel("Input File"))
        layout.addWidget(self.input_file_field)

        # Output Folder field with folder picker
        output_layout = QHBoxLayout()
        self.output_folder_field = QLineEdit()
        self.output_folder_field.setPlaceholderText("Select output folder")
        output_button = QPushButton("Browse")
        output_button.clicked.connect(self.select_output_folder)
        output_layout.addWidget(self.output_folder_field)
        output_layout.addWidget(output_button)

        layout.addWidget(QLabel("Output Folder"))
        layout.addLayout(output_layout)


        # Tabs for advanced options
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


    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_field.setText(folder)


    def set_input_file(self, path: str):
        if os.path.isfile(path):
            self.input_file_field.setText(path)
        else:
            QMessageBox.warning(self, "Invalid File", f"The selected path is not a valid file:\n{path}")

    def eventFilter(self, source, event):
        if source == self.input_file_field and event.type() == event.Type.DragEnter:
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                return True
        elif source == self.input_file_field and event.type() == event.Type.Drop:
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                self.set_input_file(file_path)
            return True
        return super().eventFilter(source, event)


    def build_command(self):
        args = []

        input_file = self.input_file_field.text()
        if input_file and os.path.isfile(input_file):
            args.append(f"--input '{input_file}'")
        else:
            QMessageBox.warning(self, "Missing Input File", "Please select a valid input file.")
            return ""

        output_folder = self.output_folder_field.text()
        if output_folder and os.path.isdir(output_folder):
            args.append(f"--output {output_folder}")
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

            if value != "" and value is not None:
                args.append(f"{flag} {value}")

        return f"luvia main {' '.join(args)}"

