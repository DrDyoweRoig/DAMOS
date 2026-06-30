"""
DAMOS – File List Panel
Shows loaded mesh files; supports single-click selection and Prev/Next navigation.
"""

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QListWidgetItem, QPushButton, QLabel, QFileDialog,
    QAbstractItemView, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal

from gui.app_state import AppState


SUPPORTED_EXTENSIONS = (".ply", ".obj")


class FileListPanel(QWidget):
    """
    Left-side panel that shows the loaded file list and navigation controls.
    """

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self._state = state
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        self.setMinimumWidth(220)
        self.setMaximumWidth(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Header ────────────────────────────────────────
        hdr = QLabel("MESH FILES")
        hdr.setObjectName("SectionLabel")
        layout.addWidget(hdr)

        # ── Load buttons ──────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self.btn_load_files = QPushButton("Load files")
        self.btn_load_files.setObjectName("NavButton")
        self.btn_load_files.setToolTip("Load individual PLY/OBJ files")

        self.btn_load_folder = QPushButton("Load folder")
        self.btn_load_folder.setObjectName("NavButton")
        self.btn_load_folder.setToolTip("Load all PLY/OBJ files in a folder")

        btn_row.addWidget(self.btn_load_files)
        btn_row.addWidget(self.btn_load_folder)
        layout.addLayout(btn_row)

        # ── File count label ──────────────────────────────
        self.count_label = QLabel("No files loaded")
        self.count_label.setObjectName("StatusLabel")
        layout.addWidget(self.count_label)

        # ── List widget ───────────────────────────────────
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.list_widget, stretch=1)

        # ── Navigation row ────────────────────────────────
        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)

        self.btn_prev = QPushButton("◀ Prev")
        self.btn_prev.setObjectName("NavButton")
        self.btn_prev.setToolTip("Previous mesh  [Left Arrow]")
        self.btn_prev.setEnabled(False)

        self.pos_label = QLabel("—")
        self.pos_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.pos_label.setObjectName("StatusLabel")
        self.pos_label.setMinimumWidth(50)

        self.btn_next = QPushButton("Next ▶")
        self.btn_next.setObjectName("NavButton")
        self.btn_next.setToolTip("Next mesh  [Right Arrow]")
        self.btn_next.setEnabled(False)

        nav_row.addWidget(self.btn_prev)
        nav_row.addWidget(self.pos_label, stretch=1)
        nav_row.addWidget(self.btn_next)
        layout.addLayout(nav_row)

        # ── Clear button ──────────────────────────────────
        self.btn_clear = QPushButton("Clear list")
        self.btn_clear.setObjectName("DangerButton")
        self.btn_clear.setToolTip("Remove all loaded files")
        layout.addWidget(self.btn_clear)

    def _connect_signals(self):
        # UI → state
        self.btn_load_files.clicked.connect(self._on_load_files)
        self.btn_load_folder.clicked.connect(self._on_load_folder)
        self.btn_prev.clicked.connect(self._state.go_previous)
        self.btn_next.clicked.connect(self._state.go_next)
        self.btn_clear.clicked.connect(self._on_clear)
        self.list_widget.currentRowChanged.connect(self._on_list_row_changed)

        # State → UI
        self._state.files_changed.connect(self._on_files_changed)
        self._state.current_index_changed.connect(self._on_index_changed)

    # ── Handlers ─────────────────────────────────────────

    def _on_load_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load mesh files", "",
            "3D Meshes (*.ply *.obj);;PLY files (*.ply);;OBJ files (*.obj)"
        )
        if paths:
            existing = set(self._state.files)
            new_paths = [p for p in paths if p not in existing]
            self._state.set_files(self._state.files + new_paths)

    def _on_load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with meshes")
        if not folder:
            return
        paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ])
        if paths:
            self._state.set_files(paths)
            self._state.post_status(f"Loaded {len(paths)} files from {folder}")
        else:
            self._state.post_status("No PLY/OBJ files found in selected folder.")

    def _on_clear(self):
        self._state.set_files([])
        self._state.set_mesh(None)
        self._state.clear_landmarks()

    def _on_list_row_changed(self, row: int):
        if row != self._state.current_index:
            self._state.set_current_index(row)

    # ── State → UI updates ───────────────────────────────

    def _on_files_changed(self, files: list[str]):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for path in files:
            item = QListWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

        n = len(files)
        self.count_label.setText(f"{n} file{'s' if n != 1 else ''} loaded" if n else "No files loaded")
        self._update_nav_buttons()

    def _on_index_changed(self, idx: int):
        # Sync list selection without re-triggering row-changed
        self.list_widget.blockSignals(True)
        self.list_widget.setCurrentRow(idx)
        self.list_widget.blockSignals(False)

        n = len(self._state.files)
        if n > 0 and 0 <= idx < n:
            self.pos_label.setText(f"{idx + 1} / {n}")
        else:
            self.pos_label.setText("—")

        self._update_nav_buttons()

    def _update_nav_buttons(self):
        n = len(self._state.files)
        idx = self._state.current_index
        self.btn_prev.setEnabled(idx > 0)
        self.btn_next.setEnabled(0 <= idx < n - 1)
