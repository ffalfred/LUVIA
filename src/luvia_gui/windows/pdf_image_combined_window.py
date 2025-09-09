
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog
)
from PyQt6.QtCore import Qt
from luvia_gui.windows.pdf_viewer_window import PDFViewerWindow
from luvia_gui.windows.image_viewer_window import ImageViewerWindow
from app_state import AppState

class PDFImageCombinedWindow(QMainWindow):
    def __init__(self, app_state: AppState):
        super().__init__()
        self.setWindowTitle("PDF + Image Viewer")
        self.setGeometry(100, 100, 1600, 800)

        self.app_state = app_state
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()

        # Top buttons
        button_layout = QHBoxLayout()
        self.open_pdf_button = QPushButton("Open PDF")
        self.open_image_button = QPushButton("Open Image")
        button_layout.addWidget(self.open_pdf_button)
        button_layout.addWidget(self.open_image_button)
        layout.addLayout(button_layout)

        self.open_pdf_button.clicked.connect(self.select_pdf)
        self.open_image_button.clicked.connect(self.select_image)

        # Splitter view
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.pdf_viewer = PDFViewerWindow(app_state)
        self.image_viewer = ImageViewerWindow(app_state)
        self.splitter.addWidget(self.pdf_viewer)
        self.splitter.addWidget(self.image_viewer)
        self.splitter.setSizes([800, 800])
        layout.addWidget(self.splitter)

        self.central_widget.setLayout(layout)

    def load_pdf_and_image(self, pdf_path: str, image_path: str):
        self.pdf_viewer.load_pdf(pdf_path)
        self.image_viewer.load_image(image_path)

    def select_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PDF", "", "PDF Files (*.pdf)")
        if file_path:
            self.pdf_viewer.load_pdf(file_path)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_path:
            self.image_viewer.load_image(file_path)
