"""
DAMOS – Dental Analysis and Morphometry Open Suite
Entry point
"""
import sys
import os

# Ensure the project root is in sys.path so all sub-packages resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from gui.main_window import MainWindow


def resource_path(relative_path: str) -> str:
    """
    Resolve resource path, compatible with PyInstaller bundles.
    In dev: uses current dir. In PyInstaller: uses sys._MEIPASS.
    """
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("DAMOS")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("Dental Anthropology Lab")

    # Icona de la finestra i de la barra de tasques
    app.setWindowIcon(QIcon(resource_path("icons/damos_logo_exe.png")))

    from gui.style import DAMOS_STYLESHEET
    app.setStyleSheet(DAMOS_STYLESHEET)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()