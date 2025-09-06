
from PyQt6.QtWidgets import QApplication
from app_state import AppState
from luvia_gui.windows.main_control import MainControlWindow
from luvia_gui.windows_manager import WindowManager

def main():
    import sys
    app = QApplication(sys.argv)

    # Create shared AppState
    app_state = AppState()

    # Create and show the fixed MainControlWindow
    main_control_window = MainControlWindow(app_state)
    screens = app.screens()
    if screens:
        main_control_window.move(screens[0].geometry().topLeft())
    main_control_window.show()

    # Create and launch dynamic windows
    window_manager = WindowManager(app_state)
    window_manager.launch_windows(screens[1:])  # Use remaining screens for dynamic windows

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
