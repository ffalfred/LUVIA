
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QFrame, QHBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QPushButton, QFileDialog
from PyQt6.QtGui import QPdfWriter, QPainter, QFont, QPixmap
from PyQt6.QtCore import QRectF, QRect

from PyQt6.QtGui import QPageSize, QTextDocument

from PyQt6.QtWidgets import QFileDialog



from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtGui import QPageSize

from PyQt6.QtWidgets import QWidget

from PyQt6.QtGui import QPdfWriter, QPainter, QPixmap
from PyQt6.QtCore import QRect, QSize, Qt, QSizeF
from PyQt6.QtGui import QPageSize

import math
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

from PyQt6.QtWidgets import QFileDialog

from PyQt6.QtPrintSupport import QPrinter


import json
import os

class HistoryView(QWidget):
    def __init__(self, jsonl_path: str, terminal=None):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.terminal = terminal
        self.init_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.load_jsonl)
        self.timer.start(3000)
        self.load_jsonl()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.download_button = QPushButton("Download PDF")
        self.download_button.clicked.connect(self.export_to_pdf)
        layout.addWidget(self.download_button)

        self.title = QLabel("Loop History Report")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet("font-family: Times-Roman; font-size: 18pt; color: black;")
        layout.addWidget(self.title)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: white;")
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)

    def load_jsonl(self):
        if not os.path.exists(self.jsonl_path):
            return

        try:
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            scroll_bar = self.scroll_area.verticalScrollBar()

            while self.scroll_layout.count():
                item = self.scroll_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()

            for idx, line in enumerate(lines):
                try:
                    data = json.loads(line)
                    entry_widget = self.create_entry_widget(idx, data)
                    self.scroll_layout.addWidget(entry_widget)
                except json.JSONDecodeError:
                    continue

            QTimer.singleShot(0, lambda: scroll_bar.setValue(scroll_bar.maximum()))

        except Exception as e:
            if self.terminal is not None:
                self.terminal.output.append(f"[HistoryView] Error loading history: {str(e)}")

    def create_divider_line(self, color, height=4):
        line = QFrame()
        line.setFixedHeight(height)
        line.setStyleSheet(f"background-color: {color};")
        return line


    def create_entry_widget(self, idx, data):
        sentence1 = data.get("sentence0", {}).get("sentence", "")
        sentence2 = data.get("sentence1", {}).get("sentence", "")
        sentence3 = data.get("sentence2", {}).get("sentence", "")
        probability1 = data.get("sentence0", {}).get("probability", 0.)
        probability2 = data.get("sentence1", {}).get("probability", 0.)
        probability3 = data.get("sentence2", {}).get("probability", 0.)

        image_paths = data.get("image", [])
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        location = data.get("location", "")
        time = data.get("time", "")
        entry_id = data.get("id", "")

        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(10, 10, 10, 10)
        container_layout.setSpacing(6)

        container_layout.addWidget(self.create_divider_line("#162F48"))  # Deep Blue
        container_layout.addWidget(self.create_divider_line("#DCB68A"))  # Golden

        entry_frame = QFrame()
        entry_layout = QHBoxLayout()
        entry_layout.setContentsMargins(0, 0, 0, 0)
        entry_layout.setSpacing(20)

        # Left: Text
        text_column = QVBoxLayout()
        header = QLabel(f"Sentence #{idx + 1}")
        header.setStyleSheet("font-family: Times-Bold; font-size: 14pt; color: black;")
        text_column.addWidget(header)

        def format_sentence(sentence: str, probability: float) -> str:
            return f"<b>{sentence}</b> <i><span style='font-size:8pt;'>(probability: {probability:.2f})</span></i>"

        # Formatted sentence label
        formatted_text = f"<b>{sentence1}</b> <i><span style='font-size:8pt;'>(probability: {probability1:.2f})</span></i>"
        sentence_label = QLabel(formatted_text)
        sentence_label.setWordWrap(True)
        sentence_label.setTextFormat(Qt.TextFormat.RichText)
        sentence_label.setStyleSheet("font-family: 'Courier New'; font-size: 13pt; color: black;")
        text_column.addWidget(sentence_label)

        # Formatted sentence label
        formatted_text = f"{sentence2} <i><span style='font-size:8pt;'>(probability: {probability2:.2f})</span></i>"
        sentence_label = QLabel(formatted_text)
        sentence_label.setWordWrap(True)
        sentence_label.setTextFormat(Qt.TextFormat.RichText)
        sentence_label.setStyleSheet("font-family: 'Courier New'; font-size: 13pt; color: black;")
        text_column.addWidget(sentence_label)
    
        # Formatted sentence label
        formatted_text = f"{sentence3} <i><span style='font-size:8pt;'>(probability: {probability3:.2f})</span></i>"
        sentence_label = QLabel(formatted_text)
        sentence_label.setWordWrap(True)
        sentence_label.setTextFormat(Qt.TextFormat.RichText)
        sentence_label.setStyleSheet("font-family: 'Courier New'; font-size: 13pt; color: black;")
        text_column.addWidget(sentence_label)


        metadata_label = QLabel()
        metadata_label.setText(
            f"<span style='font-weight:bold; color:black;'>Location:</span> {location}<br>"
            f"<span style='font-weight:bold; color:black;'>Time:</span> {time}<br>"
            f"<span style='font-weight:bold; color:black;'>ID:</span> {entry_id}"
        )
        metadata_label.setStyleSheet("font-family: Times-Roman; font-size: 10pt; color: black;")
        metadata_label.setTextFormat(Qt.TextFormat.RichText)
        text_column.addWidget(metadata_label)

        entry_layout.addLayout(text_column)

        # Right: Image
        if image_paths and os.path.exists(image_paths[0]):
            image_label = QLabel()
            pixmap = QPixmap(image_paths[0])
            pixmap = pixmap.scaledToWidth(400, Qt.TransformationMode.SmoothTransformation)
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            entry_layout.addWidget(image_label)

        entry_frame.setLayout(entry_layout)
        container_layout.addWidget(entry_frame)

        container_layout.addWidget(self.create_divider_line("#DCB68A"))  # Golden
        container_layout.addWidget(self.create_divider_line("#B8374A"))  # Red

        container.setLayout(container_layout)
        return container
    


    def export_to_pdf_high_quality(self, jsonl_path, pdf_writer, terminal=None):
        doc = QTextDocument()
        html = "<h2>Loop History Report</h2>"

        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            if terminal:
                terminal.output.append(f"[PDF Export] Error reading file: {str(e)}")
            return

        for line in lines:
            try:
                data = json.loads(line)
                html += f"""
                <p><b>Sentence 1:</b> {data.get('sentence0', {}).get('sentence', '')} 
                <i>(probability: {data.get('sentence0', {}).get('probability', 0.):.2f})</i></p>
                <p><b>Sentence 2:</b> {data.get('sentence1', {}).get('sentence', '')} 
                <i>(probability: {data.get('sentence1', {}).get('probability', 0.):.2f})</i></p>
                <p><b>Sentence 3:</b> {data.get('sentence2', {}).get('sentence', '')} 
                <i>(probability: {data.get('sentence2', {}).get('probability', 0.):.2f})</i></p>
                <p><b>Location:</b> {data.get('location', '')}<br>
                <b>Time:</b> {data.get('time', '')}<br>
                <b>ID:</b> {data.get('id', '')}</p>
                <hr>
                """
            except json.JSONDecodeError:
                continue

        doc.setHtml(html)
        doc.print(pdf_writer)

        if terminal:
            terminal.output.append("[PDF Export] High-quality PDF export completed.")

    def export_to_pdf(self):


        file_path, _ = QFileDialog.getSaveFileName(self, "Save PDF", "", "PDF Files (*.pdf)")
        if not file_path:
            return

        scale_factor = 3
        original_size = self.scroll_content.size()
        high_res_width = original_size.width() * scale_factor
        high_res_height = original_size.height() * scale_factor

        # Render the scroll_content to a high-resolution image
        full_image = QPixmap(high_res_width, high_res_height)
        full_image.fill(Qt.GlobalColor.white)

        painter = QPainter(full_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.scale(scale_factor, scale_factor)
        self.scroll_content.render(painter)
        painter.end()

        # Determine the actual content width by checking the rightmost edge of all direct child widgets
        entry_widgets = [w for w in self.scroll_content.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly)]
        max_right_edge = max((w.x() + w.width()) * scale_factor for w in entry_widgets) if entry_widgets else high_res_width

        # Set the page size tightly to the content width and full height
        content_size = QSizeF(int(max_right_edge/4), int(high_res_height/2))
        pdf_writer = QPdfWriter(file_path)
        pdf_writer.setResolution(900)
        pdf_writer.setPageSize(QPageSize(content_size, QPageSize.Unit.Point))

        # Write the image to the PDF
        pdf_painter = QPainter(pdf_writer)
        pdf_painter.drawPixmap(0, 0, full_image)
        pdf_painter.end()

        if self.terminal:
            self.terminal.output.append("[PDF Export] PDF saved with tight content width and no entry cuts.")
