
from PyQt6.QtWidgets import QMainWindow, QTreeView, QVBoxLayout, QWidget, QLabel, QHeaderView, QTextEdit
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QStandardItemModel, QStandardItem
import json
import os



class JsonViewerWindow(QMainWindow):
    def __init__(self, jsonl_path):
        super().__init__()
        self.setWindowTitle("Loop History Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.jsonl_path = jsonl_path

        # Central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout()

        # File label
        self.label = QLabel(f"Viewing: {os.path.basename(jsonl_path)}")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        # Sentence viewer
        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.text_view.setStyleSheet("""
            QTextEdit {
                background-color: #FFFEFD;
                color: #231F20;
                font-family: 'Courier Neue', 'Roboto', monospace;
                font-size: 14pt;
                padding: 20px;
                border: none;
            }
        """)
        layout.addWidget(self.text_view)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Timer for auto-refresh
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.load_jsonl)
        self.timer.start(3000)

        self.load_jsonl()


    def load_jsonl(self):
        if not os.path.exists(self.jsonl_path):
            return
        try:
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                sentences = []
                for line in lines:
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict):
                            for val in data.values():
                                if isinstance(val, str):
                                    sentences.append(val)
                        elif isinstance(data, str):
                            sentences.append(data)
                    except json.JSONDecodeError:
                        continue

                # Preserve scroll position
                scrollbar = self.text_view.verticalScrollBar()
                scroll_pos = scrollbar.value()

                self.text_view.setPlainText("\n".join(sentences))

                # Restore scroll position
                scrollbar.setValue(scroll_pos)

        except Exception as e:
            self.label.setText(f"Error loading JSONL: {str(e)}")



    def add_items(self, parent, value):
        if isinstance(value, dict):
            for key, val in value.items():
                key_item = QStandardItem(str(key))
                val_item = QStandardItem("")
                parent.appendRow([key_item, val_item])
                self.add_items(key_item, val)
        elif isinstance(value, list):
            for i, val in enumerate(value):
                key_item = QStandardItem(f"[{i}]")
                val_item = QStandardItem("")
                parent.appendRow([key_item, val_item])
                self.add_items(key_item, val)
        else:
            key_item = QStandardItem("")
            val_item = QStandardItem(str(value))
            parent.appendRow([key_item, val_item])
