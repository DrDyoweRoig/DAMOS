"""
DAMOS global stylesheet – slate blue, modern, scientific.
"""

DAMOS_STYLESHEET = """
/* ── Base ─────────────────────────────────────────────── */
* {
    font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    color: #F0F4FF;
}

QMainWindow, QDialog, QWidget {
    background-color: #1E2433;
}

QFrame#TopBar {
    background-color: #1A2030;
    border-bottom: 1px solid #323C52;
}

QFrame#CardWidget {
    background-color: #252D3D;
    border: 1px solid #323C52;
    border-radius: 8px;
}

QGroupBox {
    background-color: #252D3D;
    border: 1px solid #323C52;
    border-radius: 6px;
    margin-top: 18px;
    padding: 8px 6px 6px 6px;
    font-weight: 600;
    color: #BDD0E8;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    color: #BDD0E8;
}

QPushButton {
    background-color: #2D3F5E;
    color: #E2E8F0;
    border: 1px solid #3D5478;
    border-radius: 5px;
    padding: 6px 14px;
    min-height: 28px;
}
QPushButton:hover { background-color: #3A5080; border-color: #5A78A8; }
QPushButton:pressed { background-color: #233050; }
QPushButton:disabled { background-color: #252D3D; color: #5A6A82; border-color: #323C52; }

QPushButton#PrimaryButton {
    background-color: #1D6FA4;
    border-color: #2D8FC4;
    font-weight: 600;
    min-height: 32px;
    color: #FFFFFF;
}
QPushButton#PrimaryButton:hover { background-color: #2D8FC4; }

QPushButton#DangerButton {
    background-color: #7B2D2D;
    border-color: #A03A3A;
    color: #FFD0D0;
}
QPushButton#DangerButton:hover { background-color: #A03A3A; }

QPushButton#NavButton {
    background-color: #2A3348;
    border: 1px solid #323C52;
    border-radius: 4px;
    padding: 4px 10px;
    min-height: 26px;
}
QPushButton#NavButton:hover { background-color: #2D3F5E; border-color: #3D5478; }

QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #1A2030;
    border: 1px solid #323C52;
    border-radius: 4px;
    padding: 4px 8px;
    color: #F0F4FF;
}
QLineEdit:focus, QPlainTextEdit:focus { border-color: #2D8FC4; }

QSpinBox, QDoubleSpinBox {
    background-color: #1A2030;
    border: 1px solid #323C52;
    border-radius: 4px;
    padding: 3px 6px;
    color: #F0F4FF;
    min-width: 80px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #252D3D; border: none; width: 16px;
}

QComboBox {
    background-color: #1A2030;
    border: 1px solid #323C52;
    border-radius: 4px;
    padding: 4px 8px;
    color: #F0F4FF;
}
QComboBox QAbstractItemView {
    background-color: #252D3D;
    border: 1px solid #323C52;
    selection-background-color: #1D6FA4;
    color: #F0F4FF;
}

QCheckBox { spacing: 6px; color: #E8F0FF; }
QCheckBox::indicator {
    width: 15px; height: 15px;
    border: 1px solid #3D5478; border-radius: 3px;
    background-color: #1A2030;
}
QCheckBox::indicator:checked { background-color: #1D6FA4; border-color: #2D8FC4; }

QRadioButton { spacing: 6px; color: #E8F0FF; }
QRadioButton::indicator {
    width: 14px; height: 14px; border-radius: 7px;
    border: 1px solid #3D5478; background-color: #1A2030;
}
QRadioButton::indicator:checked { background-color: #1D6FA4; border-color: #2D8FC4; }

QListWidget, QTreeWidget, QTableWidget {
    background-color: #1A2030;
    border: 1px solid #323C52;
    border-radius: 4px;
    alternate-background-color: #1E2840;
    outline: none;
}
QListWidget::item { padding: 4px 6px; border: none; }
QListWidget::item:selected { background-color: #2D3F5E; color: #D8ECFF; }
QListWidget::item:hover { background-color: #232E42; }

QHeaderView::section {
    background-color: #252D3D;
    border: none; border-right: 1px solid #323C52; border-bottom: 1px solid #323C52;
    padding: 4px 8px; color: #94A3B8; font-weight: 600;
}

QProgressBar {
    background-color: #1A2030;
    border: 1px solid #323C52;
    border-radius: 4px;
    text-align: center;
    color: #94A3B8;
    height: 16px;
}
QProgressBar::chunk { background-color: #1D6FA4; border-radius: 3px; }

QScrollBar:vertical { background: #1A2030; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #3A4A64; min-height: 20px; border-radius: 4px; }
QScrollBar::handle:vertical:hover { background: #4A6080; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #1A2030; height: 8px; }
QScrollBar::handle:horizontal { background: #3A4A64; min-width: 20px; border-radius: 4px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

QSplitter::handle { background-color: #323C52; width: 2px; height: 2px; }

QLabel { color: #F0F4FF; background: transparent; }
QLabel#TitleLabel { font-size: 32px; font-weight: 700; color: #FFFFFF; letter-spacing: 2px; }
QLabel#SubtitleLabel { font-size: 13px; color: #9DB4CC; }
QLabel#SectionLabel { font-size: 11px; font-weight: 700; color: #9DB4CC; letter-spacing: 1.5px; }
QLabel#StatusLabel { color: #9DB4CC; font-size: 12px; }
QLabel#ValueLabel { color: #58B4FF; font-family: "Consolas", monospace; font-size: 12px; }

QToolTip {
    background-color: #2A3A54; color: #E2E8F0;
    border: 1px solid #3D5478; padding: 4px 8px; border-radius: 4px;
}

QMenuBar { background-color: #1A2030; border-bottom: 1px solid #323C52; color: #C8D8EC; }
QMenuBar::item:selected { background-color: #2D3F5E; }
QMenu { background-color: #252D3D; border: 1px solid #323C52; }
QMenu::item { padding: 5px 20px; color: #C8D8EC; }
QMenu::item:selected { background-color: #2D3F5E; }
QMenu::separator { height: 1px; background-color: #323C52; margin: 2px 10px; }
"""
