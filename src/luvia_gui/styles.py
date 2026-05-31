
# styles.py
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtWidgets import QWidget, QApplication

# Color constants
GOLDEN = "#DCB68A"
DEEP_BLUE = "#162F48"
RED = "#B8374A"
OFF_WHITE = "#FFFEFD"
OFF_BLACK = "#231F20"

def apply_dark_theme(widget: QWidget):
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(OFF_BLACK))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(OFF_WHITE))
    palette.setColor(QPalette.ColorRole.Base, QColor(DEEP_BLUE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(GOLDEN))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(OFF_WHITE))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(OFF_BLACK))
    palette.setColor(QPalette.ColorRole.Text, QColor(OFF_WHITE))
    palette.setColor(QPalette.ColorRole.Button, QColor(DEEP_BLUE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(OFF_WHITE))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(GOLDEN))
    widget.setPalette(palette)
def apply_dark_theme_main(app: QApplication):
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(OFF_BLACK))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(OFF_WHITE))
    palette.setColor(QPalette.ColorRole.Base, QColor(DEEP_BLUE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(GOLDEN))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(OFF_WHITE))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(OFF_BLACK))
    palette.setColor(QPalette.ColorRole.Text, QColor(OFF_WHITE))
    palette.setColor(QPalette.ColorRole.Button, QColor(DEEP_BLUE))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(OFF_WHITE))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(GOLDEN))
    
    app.setStyle("Fusion")
    app.setPalette(palette)
