"""
DAMOS – Main Window
Assembles all panels: Home, FileList, Viewer3D, and the six module panels.
"""

import os

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QSplitter, QLabel, QFrame,
    QStatusBar, QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QKeySequence, QShortcut

from gui.app_state import AppState
from gui.home_screen import HomeScreen
from gui.file_list_panel import FileListPanel
from gui.viewer3d import Viewer3D

from modules.meshorient_panel import MeshOrientPanel
from modules.autoplancut_panel import AutoPlaneCutPanel
from modules.polytrim_panel import PolyTrimPanel
from modules.autolmk_panel import AutoLMKPanel
from modules.automorph_panel import AutoMorphPanel
from modules.ffei_panel import FFEIPanel


# ── Module registry ──────────────────────────────────────
_MODULES = {
    "meshorient":  ("MeshOrient",     MeshOrientPanel),
    "autoplancut": ("AutoPlaneCut",   AutoPlaneCutPanel),
    "polytrim":    ("PolyTrim",       PolyTrimPanel),
    "autolmk":     ("AutoLMK",        AutoLMKPanel),
    "automorph":   ("AutoMorph",      AutoMorphPanel),
    "ffei":        ("FFEI",           FFEIPanel),
}


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAMOS  –  Dental Analysis and Morphometry Open Suite")
        self.resize(1440, 860)
        self.setMinimumSize(1000, 620)

        self._state = AppState(self)
        self._viewer = Viewer3D()
        self._module_panels: dict[str, QWidget] = {}

        self._build_ui()
        self._setup_menu()
        self._setup_shortcuts()
        self._connect_global_signals()

    # ── Build UI ──────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top bar (module tabs) ──────────────────────────
        topbar = QFrame()
        topbar.setObjectName("TopBar")
        topbar.setFixedHeight(46)
        tb_layout = QHBoxLayout(topbar)
        tb_layout.setContentsMargins(12, 4, 12, 4)
        tb_layout.setSpacing(2)

        self._tab_buttons: dict[str, QPushButton] = {}

        # Home button
        btn_home = QPushButton("Home")
        btn_home.setObjectName("NavButton")
        btn_home.setFixedHeight(32)
        btn_home.clicked.connect(lambda: self._switch_module("home"))
        tb_layout.addWidget(btn_home)
        self._tab_buttons["home"] = btn_home

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #21262D;")
        tb_layout.addWidget(sep)

        for key, (label, _) in _MODULES.items():
            btn = QPushButton(label)
            btn.setObjectName("NavButton")
            btn.setFixedHeight(32)
            btn.clicked.connect(lambda _=False, k=key: self._switch_module(k))
            tb_layout.addWidget(btn)
            self._tab_buttons[key] = btn

        tb_layout.addStretch(1)

        # File info label (top right)
        self._file_label = QLabel("")
        self._file_label.setObjectName("StatusLabel")
        self._file_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        tb_layout.addWidget(self._file_label)

        root.addWidget(topbar)

        # ── Body splitter ──────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # Left: file list
        self._file_panel = FileListPanel(self._state)
        splitter.addWidget(self._file_panel)

        # Centre: stacked pages (home + module panels)
        self._page_stack = QStackedWidget()

        # Home
        self._home = HomeScreen()
        self._home.navigate.connect(self._switch_module)
        self._page_stack.addWidget(self._home)

        # Module panels (viewer is shared)
        viewer_wrap = QWidget()
        vw_layout = QHBoxLayout(viewer_wrap)
        vw_layout.setContentsMargins(0, 0, 0, 0)
        vw_layout.setSpacing(0)

        # Inner splitter: viewer (centre) + control panel (right)
        self._inner_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._inner_splitter.setHandleWidth(4)

        # Viewer takes most space
        self._viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._inner_splitter.addWidget(self._viewer)

        # Right: control panels (stacked)
        # No fixed max width – the splitter lets the user resize freely
        self._ctrl_stack = QStackedWidget()
        self._ctrl_stack.setMinimumWidth(260)

        for key, (_, PanelClass) in _MODULES.items():
            panel = PanelClass(self._state, self._viewer)
            self._module_panels[key] = panel
            self._ctrl_stack.addWidget(panel)

        self._inner_splitter.addWidget(self._ctrl_stack)
        self._inner_splitter.setSizes([860, 340])

        vw_layout.addWidget(self._inner_splitter)
        self._page_stack.addWidget(viewer_wrap)

        splitter.addWidget(self._page_stack)
        splitter.setSizes([220, 1220])

        root.addWidget(splitter, stretch=1)

        # ── Status bar ─────────────────────────────────────
        self._status_bar = QStatusBar()
        self._status_bar.setStyleSheet(
            "QStatusBar { background: #161B22; border-top: 1px solid #21262D; "
            "color: #8B9DC3; font-size: 12px; }"
        )
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("DAMOS ready.")

        # Start on Home
        self._switch_module("home")

    # ── Menu ──────────────────────────────────────────────

    def _setup_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        file_menu.addAction("Load Files…",   self._file_panel._on_load_files,    "Ctrl+O")
        file_menu.addAction("Load Folder…",  self._file_panel._on_load_folder,   "Ctrl+Shift+O")
        file_menu.addSeparator()
        file_menu.addAction("Quit",          self.close,                          "Ctrl+Q")

        view_menu = menubar.addMenu("View")
        view_menu.addAction("Occlusal view",    self._viewer.view_occlusal,   "Ctrl+1")
        view_menu.addAction("Reset camera",     self._viewer.reset_camera,    "Ctrl+R")

        nav_menu = menubar.addMenu("Navigate")
        nav_menu.addAction("Previous mesh",     self._state.go_previous,  "Left")
        nav_menu.addAction("Next mesh",         self._state.go_next,      "Right")

    # ── Keyboard shortcuts ────────────────────────────────

    def _setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Left),  self, self._state.go_previous)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._state.go_next)

    # ── Global signal connections ─────────────────────────

    def _connect_global_signals(self):
        self._state.status_message.connect(self._status_bar.showMessage)
        self._state.current_index_changed.connect(self._on_index_changed)

    # ── Module switching ──────────────────────────────────

    def _switch_module(self, key: str):
        # Highlight active tab
        for k, btn in self._tab_buttons.items():
            btn.setStyleSheet(
                "QPushButton#NavButton { background: #1D6FA4; color: #E6EDF3; font-weight: 600; }"
                if k == key else ""
            )

        if key == "home":
            self._file_panel.setVisible(False)
            self.menuBar().setVisible(False)
            self._page_stack.setCurrentWidget(self._home)
        else:
            self._file_panel.setVisible(True)
            self.menuBar().setVisible(True)
            viewer_page = self._page_stack.widget(1)
            self._page_stack.setCurrentWidget(viewer_page)
            panel = self._module_panels.get(key)
            if panel:
                self._ctrl_stack.setCurrentWidget(panel)

    # ── File navigation ────────────────────────────────────

    def _on_index_changed(self, idx: int):
        files = self._state.files
        if 0 <= idx < len(files):
            fname = os.path.basename(files[idx])
            total = len(files)
            self._file_label.setText(f"{fname}   [{idx + 1} / {total}]")
        else:
            self._file_label.setText("")

    # ── Close ─────────────────────────────────────────────

    def closeEvent(self, event):
        try:
            self._viewer.closeEvent(event)
        except Exception:
            pass
        super().closeEvent(event)