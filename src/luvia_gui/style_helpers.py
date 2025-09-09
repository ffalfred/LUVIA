
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QPushButton, QTabWidget

# Color constants
GOLDEN = "#DCB68A"
DEEP_BLUE = "#162F48"
RED = "#B8374A"
OFF_WHITE = "#FFFEFD"
OFF_BLACK = "#231F20"

def style_button(button: QPushButton):
    button.setStyleSheet(f"""
        QPushButton {{
            background-color: {DEEP_BLUE};
            color: {OFF_WHITE};
            border: 1px solid {GOLDEN};
            padding: 6px 12px;
            font-family: Roboto;
        }}
        QPushButton:hover {{
            background-color: {GOLDEN};
            color: {OFF_BLACK};
        }}
    """)

def style_tabs(tabs: QTabWidget):
    tabs.setStyleSheet(f"""
        QTabWidget::pane {{
            border: 1px solid {GOLDEN};
        }}
        QTabBar::tab {{
            background: {DEEP_BLUE};
            color: {OFF_WHITE};
            padding: 6px;
        }}
        QTabBar::tab:selected {{
            background: {GOLDEN};
            color: {OFF_BLACK};
        }}
    """)

def apply_fonts(widget):
    widget.setFont(QFont("Roboto", 10))
