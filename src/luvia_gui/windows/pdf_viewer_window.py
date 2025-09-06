import os
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt6.QtPdfWidgets import QPdfView
from PyQt6.QtPdf import QPdfDocument
from PyQt6.QtCore import QPointF
from app_state import AppState

class PDFViewerWindow(QMainWindow):
    def __init__(self, app_state: AppState):
        super().__init__()
        self.setWindowTitle("PDF Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.app_state = app_state

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        self.pdf_view = QPdfView(self)
        self.pdf_doc = QPdfDocument(self)
        self.pdf_view.setDocument(self.pdf_doc)
        self.main_layout.addWidget(self.pdf_view)

        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous Page")
        self.next_button = QPushButton("Next Page")
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        self.main_layout.addLayout(nav_layout)

        self.prev_button.clicked.connect(self.show_previous_page)
        self.next_button.clicked.connect(self.show_next_page)

        self.app_state.output_folder_changed.connect(self.on_output_folder_changed)

        self.current_folder = self.app_state.get_output_folder()
        self.current_page = 0
        self.load_latest_pdf()

    def on_output_folder_changed(self, folder: str):
        self.current_folder = folder
        self.load_latest_pdf()

    def load_pdf(self, path: str):
        if os.path.exists(path) and path.lower().endswith(".pdf"):
            self.pdf_doc.load(path)
            self.current_page = 0
            self.pdf_view.pageNavigator().jump(
                self.current_page, QPointF(), self.pdf_view.pageNavigator().currentZoom()
            )

    def load_latest_pdf(self):
        if not self.current_folder or not os.path.exists(self.current_folder):
            return

        pdf_files = [f for f in os.listdir(self.current_folder) if f.lower().endswith('.pdf')]
        if not pdf_files:
            return

        pdf_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.current_folder, f)), reverse=True)
        latest_pdf = os.path.join(self.current_folder, pdf_files[0])

        self.pdf_doc.load(latest_pdf)
        self.current_page = 0
        self.pdf_view.pageNavigator().jump(self.current_page, QPointF(), self.pdf_view.pageNavigator().currentZoom())

    def show_previous_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.pdf_view.pageNavigator().jump(self.current_page, QPointF(), self.pdf_view.pageNavigator().currentZoom())

    def show_next_page(self):
        if self.current_page < self.pdf_doc.pageCount() - 1:
            self.current_page += 1
            self.pdf_view.pageNavigator().jump(self.current_page, QPointF(), self.pdf_view.pageNavigator().currentZoom())