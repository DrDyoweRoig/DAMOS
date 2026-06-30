"""
DAMOS – PolyTrim module panel
Batch mesh decimation using PyVista (replaces the Slicer extension).
"""

import os

import numpy as np
import pyvista as pv

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QPushButton, QComboBox, QSpinBox,
    QCheckBox, QLineEdit, QPlainTextEdit, QProgressBar,
    QFileDialog, QScrollArea, QFrame, QLabel
)
from PySide6.QtCore import QThread, Signal, QObject

from gui.app_state import AppState
from gui.viewer3d import Viewer3D


# ─────────────────────────────────────────────────────────────────────────────
# Logic
# ─────────────────────────────────────────────────────────────────────────────

class PolyTrimLogic:
    def decimate_to_target(self, mesh: pv.PolyData, target_polys: int,
                           preserve_topology: bool = True) -> pv.PolyData:
        mesh = mesh.triangulate()
        n0 = mesh.n_cells

        if n0 == 0:
            raise ValueError("Input mesh has 0 polygons.")
        if n0 <= target_polys:
            return mesh

        reduction = max(0.0, min(0.99, 1.0 - target_polys / float(n0)))

        # decimate_pro = vtkDecimatePro (same as original Slicer PolyTrim)
        decimated = mesh.decimate_pro(
            reduction=reduction,
            preserve_topology=preserve_topology,
            boundary_vertex_deletion=False,
            splitting=False,
            inplace=False,
        )
        return decimated.triangulate().clean()


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

class _TrimWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal()

    def __init__(self, files: list[str], target: int, preserve: bool,
                 output_dir: str):
        super().__init__()
        self._files = files
        self._target = target
        self._preserve = preserve
        self._output_dir = output_dir
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        logic = PolyTrimLogic()
        n = len(self._files)
        os.makedirs(self._output_dir, exist_ok=True)

        for i, path in enumerate(self._files, 1):
            if self._abort:
                self.progress.emit(i, n, "Aborted.")
                break
            fname = os.path.basename(path)
            try:
                mesh = pv.read(path)
                n0 = mesh.n_cells
                out = logic.decimate_to_target(mesh, self._target, self._preserve)
                n1 = out.n_cells
                base = os.path.splitext(fname)[0]
                out_path = os.path.join(self._output_dir, f"{base}_{self._target}.ply")
                out.save(out_path)
                self.progress.emit(i, n, f"OK  {fname}: {n0} → {n1}")
            except Exception as e:
                self.progress.emit(i, n, f"FAIL  {fname}: {e}")

        self.finished.emit()


# ─────────────────────────────────────────────────────────────────────────────
# Panel
# ─────────────────────────────────────────────────────────────────────────────

class PolyTrimPanel(QWidget):

    def __init__(self, state: AppState, viewer: Viewer3D, parent=None):
        super().__init__(parent)
        self._state = state
        self._viewer = viewer
        self._logic = PolyTrimLogic()
        self._worker_thread: QThread | None = None
        self._worker: _TrimWorker | None = None
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)

        # Input
        grp_in = QGroupBox("Input")
        f_in = QFormLayout(grp_in)

        self.cmb_format = QComboBox()
        self.cmb_format.addItems(["PLY", "OBJ"])
        f_in.addRow("Input format:", self.cmb_format)

        row_in = QHBoxLayout()
        self.le_input = QLineEdit()
        self.le_input.setPlaceholderText("Input folder…")
        self.le_input.setReadOnly(True)
        btn_in = QPushButton("Browse…")
        btn_in.setObjectName("NavButton")
        btn_in.clicked.connect(self._browse_input)
        row_in.addWidget(self.le_input)
        row_in.addWidget(btn_in)
        f_in.addRow("Input folder:", row_in)

        layout.addWidget(grp_in)

        # Decimation
        grp_dec = QGroupBox("Decimation")
        f_dec = QFormLayout(grp_dec)

        self.spn_target = QSpinBox()
        self.spn_target.setMinimum(1000)
        self.spn_target.setMaximum(2_000_000)
        self.spn_target.setSingleStep(1000)
        self.spn_target.setValue(20_000)
        f_dec.addRow("Target polygons:", self.spn_target)

        self.chk_preserve = QCheckBox()
        self.chk_preserve.setChecked(True)
        f_dec.addRow("Preserve topology:", self.chk_preserve)

        layout.addWidget(grp_dec)

        # Preview
        grp_prev = QGroupBox("Preview (current mesh)")
        f_prev = QFormLayout(grp_prev)
        self.btn_preview = QPushButton("Preview decimation")
        self.btn_preview.setObjectName("PrimaryButton")
        f_prev.addRow(self.btn_preview)

        self.lbl_before = QLabel("—")
        self.lbl_before.setObjectName("ValueLabel")
        f_prev.addRow("Before:", self.lbl_before)

        self.lbl_after = QLabel("—")
        self.lbl_after.setObjectName("ValueLabel")
        f_prev.addRow("After:", self.lbl_after)

        layout.addWidget(grp_prev)

        # Output
        grp_out = QGroupBox("Batch Output")
        f_out = QFormLayout(grp_out)

        row_out = QHBoxLayout()
        self.le_output = QLineEdit()
        self.le_output.setPlaceholderText("Output folder (default: input/output)")
        btn_out = QPushButton("Browse…")
        btn_out.setObjectName("NavButton")
        btn_out.clicked.connect(self._browse_output)
        row_out.addWidget(self.le_output)
        row_out.addWidget(btn_out)
        f_out.addRow("Output folder:", row_out)

        layout.addWidget(grp_out)

        # Log
        grp_log = QGroupBox("Log")
        f_log = QVBoxLayout(grp_log)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        f_log.addWidget(self.progress_bar)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(100)
        f_log.addWidget(self.log_box)
        layout.addWidget(grp_log)

        layout.addStretch(1)

        # Run / Abort
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Batch")
        self.btn_run.setObjectName("PrimaryButton")
        self.btn_run.setMinimumHeight(36)
        self.btn_abort = QPushButton("Abort")
        self.btn_abort.setObjectName("DangerButton")
        self.btn_abort.setEnabled(False)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_abort)
        layout.addLayout(btn_row)

        scroll.setWidget(inner)
        root.addWidget(scroll)

    def _connect_signals(self):
        self.btn_preview.clicked.connect(self._on_preview)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_abort.clicked.connect(self._on_abort)
        self._state.current_index_changed.connect(self._on_index_changed)

    def _browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select input folder")
        if folder:
            self.le_input.setText(folder)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self.le_output.setText(folder)

    def _on_index_changed(self, idx: int):
        path = self._state.current_path
        if path is None:
            return
        try:
            mesh = pv.read(path)
            self._state.set_mesh(mesh)
            self._viewer.display_mesh(mesh)
            self.lbl_before.setText(str(mesh.n_cells))
            self.lbl_after.setText("—")
            self._state.post_status(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            self._state.post_status(f"Load error: {e}")

    def _on_preview(self):
        mesh = self._state.current_mesh
        if mesh is None:
            self._log("No mesh loaded.")
            return
        target = self.spn_target.value()
        preserve = self.chk_preserve.isChecked()
        try:
            before = mesh.n_cells
            out = self._logic.decimate_to_target(mesh, target, preserve)
            after = out.n_cells
            self.lbl_before.setText(str(before))
            self.lbl_after.setText(str(after))
            self._viewer.display_mesh(out)
            self._log(f"Preview: {before} → {after} polygons")
        except Exception as e:
            self._log(f"Preview error: {e}")

    def _on_run(self):
        input_dir = self.le_input.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            self._log("Please select a valid input folder.")
            return
        ext = f".{self.cmb_format.currentText().lower()}"
        files = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(ext)
        ])
        if not files:
            self._log(f"No {ext} files found.")
            return
        output_dir = self.le_output.text().strip() or os.path.join(input_dir, "output")

        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_abort.setEnabled(True)

        self._worker = _TrimWorker(
            files, self.spn_target.value(),
            self.chk_preserve.isChecked(), output_dir
        )
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_done)
        self._worker_thread.start()

    def _on_abort(self):
        if self._worker:
            self._worker.abort()

    def _on_worker_progress(self, current: int, total: int, msg: str):
        self.progress_bar.setValue(int(current / total * 100))
        self._log(f"[{current}/{total}] {msg}")

    def _on_worker_done(self):
        self.btn_run.setEnabled(True)
        self.btn_abort.setEnabled(False)
        self.progress_bar.setValue(100)
        self._log("Done.")
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread.wait()
        self._state.post_status("PolyTrim batch complete.")

    def _log(self, msg: str):
        self.log_box.appendPlainText(msg)
        self.log_box.ensureCursorVisible()
