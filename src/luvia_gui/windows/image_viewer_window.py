
import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QScrollArea,
    QGridLayout, QPushButton
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap
from app_state import AppState

from PIL import Image
import os
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtGui import QPixmap, QWheelEvent, QMouseEvent
from PyQt6.QtCore import Qt
import os
from PyQt6.QtCore import QPoint

class ImageViewerWindow(QMainWindow):
    def __init__(self, app_state, title="Image Viewer"):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 800, 600)
        self.app_state = app_state
        self.zoom_factor = 1.0
        self.current_image_path = ""

        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # Zoom buttons
        button_layout = QHBoxLayout()
        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_out_button = QPushButton("Zoom Out")
        self.reset_zoom_button = QPushButton("Reset Zoom")
        button_layout.addWidget(self.zoom_in_button)
        button_layout.addWidget(self.zoom_out_button)
        button_layout.addWidget(self.reset_zoom_button)
        self.main_layout.addLayout(button_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setStyleSheet("background-color: black;")
        self.image_label.setStyleSheet("background-color: black;")
        self.main_layout.addWidget(self.scroll_area)

        # Connect buttons
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.reset_zoom_button.clicked.connect(self.reset_zoom)

        self.dragging = False
        self.drag_start_position = QPoint()

        self.image_label.setScaledContents(False)
        self.image_label.setMouseTracking(True)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_image)
        self.refresh_interval_ms = 10000  # 10 seconds


    def start_auto_refresh(self, image_path: str):
        self.current_image_path = image_path
        self.refresh_timer.start(self.refresh_interval_ms)
        self.update_image_display()

    def stop_auto_refresh(self):
        self.refresh_timer.stop()

    def refresh_image(self):
        if os.path.exists(self.current_image_path):
            self.update_image_display()

    def load_image(self, image_path: str):
        if os.path.exists(image_path) and image_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            self.current_image_path = image_path
            self.zoom_factor = 1.0
            self.update_image_display()

    def update_image_display(self):
        pixmap = QPixmap(self.current_image_path)
        if not pixmap.isNull():
            scroll_area_size = self.scroll_area.viewport().size() * self.zoom_factor
            scaled_pixmap = pixmap.scaled(
                scroll_area_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setText("")
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText("Failed to load image. {}".format(self.current_image_path))

    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.update_image_display()

    def zoom_out(self):
        self.zoom_factor /= 1.2
        self.update_image_display()

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.update_image_display()

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()


    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False


    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dragging:
            delta = event.pos() - self.drag_start_position
            self.drag_start_position = event.pos()
            self.image_label.move(self.image_label.pos() + delta)


    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.drag_start_position = event.pos()



class ImageView(QWidget):
    def __init__(self, reference_image_path: str, dynamic_image_path: str):
        super().__init__()
        self.reference_image_path = reference_image_path
        self.dynamic_image_path = dynamic_image_path
        self.last_modified = None

        self.reference_label = QLabel()
        self.dynamic_label = QLabel()

        self.zoom_factor = 1.0

        self.init_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_for_update)
        self.timer.start(2000)  # Check every 2 seconds

        self.dragging = False
        self.drag_start_position = None

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.title = QLabel("Image Comparison Viewer")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet("font-family: Times-Roman; font-size: 18pt; color: black;")
        layout.addWidget(self.title)


        # Zoom buttons
        button_layout = QHBoxLayout()
        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        button_layout.addWidget(self.zoom_in_button)
        button_layout.addWidget(self.zoom_out_button)
        layout.addLayout(button_layout)


        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: black;")
        layout.addWidget(self.scroll_area)

        self.scroll_content = QWidget()
        self.scroll_layout = QHBoxLayout()  # Horizontal layout for side-by-side images
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_content)


        self.scroll_area.setMouseTracking(True)
        self.scroll_area.viewport().setMouseTracking(True)


        self.reference_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dynamic_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reference_label.setStyleSheet("background-color: black;")
        self.dynamic_label.setStyleSheet("background-color: black;")

        self.scroll_layout.addWidget(self.reference_label)
        self.scroll_layout.addWidget(self.dynamic_label)


        self.load_images()


    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.load_images()

    def zoom_out(self):
        self.zoom_factor /= 1.2
        self.load_images()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.drag_start_position = event.pos()

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = event.pos() - self.drag_start_position
            self.drag_start_position = event.pos()
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() - delta.x()
            )
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - delta.y()
            )

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False

    def load_images(self):
        # Load reference image
        if os.path.exists(self.reference_image_path):
            ref_pixmap = QPixmap(self.reference_image_path)
            scaled_ref = ref_pixmap.scaled(
                ref_pixmap.size() * self.zoom_factor,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.reference_label.setPixmap(scaled_ref)
        else:
            self.reference_label.setText(f"Reference image not found: {self.reference_image_path}")

        # Load dynamic image
        if os.path.exists(self.dynamic_image_path):
            dyn_pixmap = QPixmap(self.dynamic_image_path)
            scaled_dyn = dyn_pixmap.scaled(
                dyn_pixmap.size() * self.zoom_factor,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.dynamic_label.setPixmap(scaled_dyn)
            self.last_modified = os.path.getmtime(self.dynamic_image_path)
        else:
            self.dynamic_label.setText(f"Dynamic image not found: {self.dynamic_image_path}")


    def check_for_update(self):
        if not os.path.exists(self.dynamic_image_path):
            return

        current_modified = os.path.getmtime(self.dynamic_image_path)
        if self.last_modified is None or current_modified > self.last_modified:
            self.load_images()

    def stop_auto_refresh(self):
        self.timer.stop()


