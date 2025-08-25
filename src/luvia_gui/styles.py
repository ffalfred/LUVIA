
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtWidgets import QWidget

def apply_dark_theme(widget: QWidget):
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#231F20"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#FFFEFD"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#162F48"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#DCB68A"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#FFFEFD"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#231F20"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#FFFEFD"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#B8374A"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#FFFEFD"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#DCB68A"))
    widget.setPalette(palette)
