"""
DAMOS – AutoLMK module panel
Engine ported 1:1 from dental_landmarks_gui.py (MorphologyEngine + LandmarkingTab).
Full landmark visualisation, batch processing with preview, and manual editing.
"""

import os
import math
from typing import Optional, Dict, List

import numpy as np
import pyvista as pv

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar,
    QFileDialog, QScrollArea, QFrame, QLabel,
    QListWidget, QListWidgetItem, QAbstractItemView,
    QSizePolicy, QComboBox, QLineEdit, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer

from gui.app_state import AppState
from gui.viewer3d import Viewer3D
from morphology_engine import DetectionResult, MorphologyEngine

# ─────────────────────────────────────────────────────────────────────────────
# Batch worker
# ─────────────────────────────────────────────────────────────────────────────

class _LMKWorker(QObject):
    progress = Signal(int, int, str, object)   # current, total, msg, result|None
    finished = Signal()

    def __init__(self, files: list, params: dict):
        super().__init__()
        self._files = files
        self._params = params
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        engine = MorphologyEngine()
        n = len(self._files)
        results = {}

        for i, path in enumerate(self._files, 1):
            if self._abort:
                self.progress.emit(i, n, "Aborted.", None)
                break
            fname = os.path.basename(path)
            try:
                engine.load_mesh(path)
                result = engine.detect_all(**self._params)
                results[path] = result
                msg = f"OK  {fname}  ({len(result.cusps)} cusps)"
                if engine._ordering_warnings:
                    msg += f"  [{len(engine._ordering_warnings)} warning(s)]"
                    for w in engine._ordering_warnings:
                        self.progress.emit(i, n, f"  {w}", None)
                self.progress.emit(i, n, msg, result)
            except Exception as e:
                self.progress.emit(i, n, f"FAIL  {fname}: {e}", None)

        self.finished.emit()


# ─────────────────────────────────────────────────────────────────────────────
# Panel
# ─────────────────────────────────────────────────────────────────────────────

class AutoLMKPanel(QWidget):

    def __init__(self, state: AppState, viewer: Viewer3D, parent=None):
        super().__init__(parent)
        self._state  = state
        self._viewer = viewer
        self._engine = MorphologyEngine()

        # Per-tooth state
        self._current_mesh: Optional[pv.PolyData] = None
        self._current_result: Optional[DetectionResult] = None
        self._original_result: Optional[DetectionResult] = None
        self._picked_point: Optional[np.ndarray] = None

        # Batch state
        self._batch_results: Dict[str, DetectionResult] = {}
        self._batch_original: Dict[str, DetectionResult] = {}
        self._batch_status: Dict[str, str] = {}

        self._worker: Optional[_LMKWorker] = None
        self._worker_thread: Optional[QThread] = None

        # Manual placement state
        self._manual_placing = False
        self._manual_cusps: List[np.ndarray] = []
        self._manual_n_cusps = 0

        self._pending_reload = False

        self._build_ui()
        self._connect_signals()
        self._enable_picking()

    # ── Build UI ──────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── Detection settings ────────────────────────────
        grp_det = QGroupBox("Detection Settings")
        f_det = QFormLayout(grp_det)

        self.cmb_tooth = QComboBox()
        self.cmb_tooth.addItems([
            "Auto (generic)",
            "M1 lower – Sapiens (5 cusps)",
            "M1 upper – Sapiens (4 cusps)",
            "M2 lower – Sapiens (4 cusps)",
            "M2 upper – Sapiens (4 cusps)",
            "Pm – Sapiens (2 cusps)",
            "Bilophodont (4 cusps)",
        ])
        f_det.addRow("Tooth type:", self.cmb_tooth)

        self.spn_cusps = QSpinBox()
        self.spn_cusps.setRange(2, 10); self.spn_cusps.setValue(5)
        f_det.addRow("Expected cusps:", self.spn_cusps)

        self.spn_border = QSpinBox()
        self.spn_border.setRange(10, 500); self.spn_border.setValue(50)
        f_det.addRow("N semilandmarks:", self.spn_border)

        def _toggle_cusps_row(text: str):
            f_det.setRowVisible(1, text == "Auto (generic)")
        self.cmb_tooth.currentTextChanged.connect(_toggle_cusps_row)

        layout.addWidget(grp_det)

        # ── Visibility toggles (2-column grid) ────────────
        grp_vis = QGroupBox("Visibility")
        from PySide6.QtWidgets import QGridLayout
        f_vis = QGridLayout(grp_vis)
        f_vis.setSpacing(4)
        self.chk_cusps   = QCheckBox("Cusps");         self.chk_cusps.setChecked(True)
        self.chk_inter   = QCheckBox("Intercuspal");   self.chk_inter.setChecked(True)
        self.chk_center  = QCheckBox("Center");        self.chk_center.setChecked(True)
        self.chk_border  = QCheckBox("Semilandmarks"); self.chk_border.setChecked(True)
        self.chk_curve   = QCheckBox("Border curve");  self.chk_curve.setChecked(False)
        self.chk_labels  = QCheckBox("Labels");        self.chk_labels.setChecked(True)
        chks = [self.chk_cusps, self.chk_inter, self.chk_center,
                self.chk_border, self.chk_curve, self.chk_labels]
        for i, chk in enumerate(chks):
            f_vis.addWidget(chk, i // 2, i % 2)
        layout.addWidget(grp_vis)

        # ── Current mesh ───────────────────────────────────
        grp_cur = QGroupBox("Current Mesh")
        f_cur = QVBoxLayout(grp_cur)

        self.btn_detect   = QPushButton("Detect landmarks")
        self.btn_detect.setObjectName("PrimaryButton")
        self.btn_detect.setMinimumHeight(34)
        self.btn_recompute = QPushButton("Recompute dependent points")
        self.btn_recompute.setObjectName("NavButton")
        f_cur.addWidget(self.btn_detect)
        f_cur.addWidget(self.btn_recompute)

        # Status labels
        self.lbl_status = QLabel("No result")
        self.lbl_status.setObjectName("StatusLabel")
        f_cur.addWidget(self.lbl_status)

        layout.addWidget(grp_cur)

        # ── Manual placement ──────────────────────────────
        grp_manual = QGroupBox("Manual Landmark Placement")
        f_man = QVBoxLayout(grp_manual)
        f_man.addWidget(QLabel(
            "Place cusps manually by clicking on the mesh.\n"
            "Intercuspal, center and border semilandmarks\n"
            "will be computed automatically."
        ))

        man_row = QHBoxLayout()
        man_row.addWidget(QLabel("Number of cusps:"))
        self.spn_manual_cusps = QSpinBox()
        self.spn_manual_cusps.setRange(2, 10)
        self.spn_manual_cusps.setValue(4)
        man_row.addWidget(self.spn_manual_cusps)
        f_man.addLayout(man_row)

        self.btn_start_manual = QPushButton("Start manual placement")
        self.btn_start_manual.setObjectName("PrimaryButton")
        self.btn_start_manual.setMinimumHeight(34)
        f_man.addWidget(self.btn_start_manual)

        self.lbl_manual_status = QLabel("")
        self.lbl_manual_status.setObjectName("StatusLabel")
        self.lbl_manual_status.setWordWrap(True)
        f_man.addWidget(self.lbl_manual_status)

        self.btn_undo_manual = QPushButton("Undo last cusp")
        self.btn_undo_manual.setObjectName("NavButton")
        self.btn_undo_manual.setEnabled(False)
        f_man.addWidget(self.btn_undo_manual)

        self.btn_cancel_manual = QPushButton("Cancel placement")
        self.btn_cancel_manual.setObjectName("DangerButton")
        self.btn_cancel_manual.setEnabled(False)
        f_man.addWidget(self.btn_cancel_manual)

        layout.addWidget(grp_manual)

        # ── Edit mode ──────────────────────────────────────
        grp_edit = QGroupBox("Manual Editing")
        f_edit = QFormLayout(grp_edit)

        self.chk_edit = QCheckBox("Edit mode (click mesh to pick)")
        f_edit.addRow(self.chk_edit)

        self.cmb_edit_group = QComboBox()
        self.cmb_edit_group.addItems(["Cusps", "Intercuspal minima", "Center"])
        f_edit.addRow("Landmark group:", self.cmb_edit_group)

        self.spn_edit_idx = QSpinBox()
        self.spn_edit_idx.setRange(1, 10); self.spn_edit_idx.setValue(1)
        f_edit.addRow("Index:", self.spn_edit_idx)

        self.lbl_picked = QLabel("Picked point: none")
        self.lbl_picked.setObjectName("StatusLabel")
        f_edit.addRow(self.lbl_picked)

        self.btn_apply = QPushButton("Apply picked position")
        self.btn_apply.setObjectName("PrimaryButton")
        f_edit.addRow(self.btn_apply)
        self.btn_reset_auto = QPushButton("Reset to auto")
        self.btn_reset_auto.setObjectName("NavButton")
        f_edit.addRow(self.btn_reset_auto)

        layout.addWidget(grp_edit)

        # ── Export ────────────────────────────────────────
        grp_exp = QGroupBox("Export")
        f_exp = QVBoxLayout(grp_exp)
        f_exp.setSpacing(4)

        from PySide6.QtWidgets import QGridLayout as _QGrid
        fmt_row = QFormLayout()
        self.cmb_export_fmt = QComboBox()
        self.cmb_export_fmt.addItems([
            "CSV (DAMOS)",
            "CSV wide (geomorph/R)",
            "Morphologika",
            "TPS",
            "NTS (MorphoJ)",
            "JSON",
            "FCSV (3D Slicer)",
        ])
        self.cmb_export_fmt.setToolTip(
            "CSV (DAMOS): one row per landmark, columns = type,x,y,z\n"
            "CSV wide: one row per specimen, columns = X1,Y1,Z1,X2,…\n"
            "Morphologika: standard .txt for GM software\n"
            "TPS: Thin Plate Spline format (tpsDig/tpsRelw)\n"
            "NTS: MorphoJ format\n"
            "JSON: structured, machine-readable\n"
            "FCSV: 3D Slicer / Checkpoint markup"
        )
        fmt_row.addRow("Format:", self.cmb_export_fmt)
        f_exp.addLayout(fmt_row)

        self.btn_export_csv = QPushButton("Save current")
        self.btn_export_csv.setObjectName("NavButton")
        self.btn_export_all = QPushButton("Save all batch")
        self.btn_export_all.setObjectName("NavButton")
        f_exp.addWidget(self.btn_export_csv)
        f_exp.addWidget(self.btn_export_all)
        layout.addWidget(grp_exp)

        # ── Batch ─────────────────────────────────────────
        grp_batch = QGroupBox("Batch Processing")
        f_batch = QVBoxLayout(grp_batch)

        row_in = QHBoxLayout()
        self.le_batch = QLineEdit()
        self.le_batch.setPlaceholderText("Folder with PLY/OBJ meshes…")
        self.le_batch.setReadOnly(True)
        btn_bi = QPushButton("Browse…")
        btn_bi.setObjectName("NavButton")
        btn_bi.clicked.connect(self._browse_batch)
        row_in.addWidget(self.le_batch); row_in.addWidget(btn_bi)
        f_batch.addLayout(row_in)

        self.progress_bar = QProgressBar(); self.progress_bar.setValue(0)
        f_batch.addWidget(self.progress_bar)

        self.btn_batch_run = QPushButton("Run Batch")
        self.btn_batch_run.setObjectName("PrimaryButton")
        self.btn_batch_run.setMinimumHeight(34)
        f_batch.addWidget(self.btn_batch_run)
        self.btn_batch_abort = QPushButton("Abort")
        self.btn_batch_abort.setObjectName("DangerButton")
        self.btn_batch_abort.setEnabled(False)
        f_batch.addWidget(self.btn_batch_abort)

        layout.addWidget(grp_batch)

        # ── Log ───────────────────────────────────────────
        grp_log = QGroupBox("Log")
        f_log = QVBoxLayout(grp_log)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(80)
        f_log.addWidget(self.log_box)
        layout.addWidget(grp_log)
        layout.addStretch(1)

        scroll.setWidget(inner)
        root.addWidget(scroll)

    # ── Signals ───────────────────────────────────────────

    def _connect_signals(self):
        self.btn_detect.clicked.connect(self._on_detect)
        self.btn_recompute.clicked.connect(self._on_recompute)
        self.btn_apply.clicked.connect(self._on_apply_pick)
        self.btn_reset_auto.clicked.connect(self._on_reset_auto)
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        self.btn_export_all.clicked.connect(self._on_export_all)
        self.btn_batch_run.clicked.connect(self._on_batch_run)
        self.btn_batch_abort.clicked.connect(self._on_batch_abort)

        self.btn_start_manual.clicked.connect(self._on_start_manual)
        self.btn_undo_manual.clicked.connect(self._on_undo_manual)
        self.btn_cancel_manual.clicked.connect(self._on_cancel_manual)

        self.chk_edit.toggled.connect(self._on_edit_mode_toggled)
        self.cmb_edit_group.currentIndexChanged.connect(self._update_edit_limits)
        self.cmb_edit_group.currentIndexChanged.connect(self._refresh_scene)
        self.spn_edit_idx.valueChanged.connect(self._refresh_scene)

        for chk in (self.chk_cusps, self.chk_inter, self.chk_center,
                    self.chk_border, self.chk_curve, self.chk_labels):
            chk.toggled.connect(self._refresh_scene)

        self._state.current_index_changed.connect(self._on_index_changed)
        self._viewer.landmark_picked.connect(self._on_viewer_pick)

    def _enable_picking(self):
        self._viewer.set_click_handler(self._on_mesh_clicked)

    def showEvent(self, event):
        super().showEvent(event)
        if self._pending_reload:
            self._on_index_changed(self._state.current_index)

    # ── Index change ──────────────────────────────────────

    def _on_index_changed(self, idx: int):
        if not self.isVisible():
            self._pending_reload = True
            return
        self._pending_reload = False
        path = self._state.current_path
        if path is None:
            return
        try:
            mesh = pv.read(path)
            self._engine.set_mesh(mesh)
            self._current_mesh = mesh
            self._state.set_mesh(mesh)

            # Restore cached result if batch was run
            if path in self._batch_results:
                self._current_result  = self._batch_results[path]
                self._original_result = self._batch_original.get(path)
                status = self._batch_status.get(path, "auto")
            else:
                self._current_result  = None
                self._original_result = None
                status = "loaded"

            self._picked_point = None
            self.lbl_picked.setText("Picked point: none")
            self.lbl_status.setText(f"Status: {status}")
            self._update_edit_limits()
            self._refresh_scene()
            self._state.post_status(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            self._state.post_status(f"Load error: {e}")

    # ── Detection ─────────────────────────────────────────

    def _get_params(self) -> dict:
        return {
            "n_cusps":               self.spn_cusps.value(),
            "tooth_type":            self.cmb_tooth.currentText(),
            "n_border_semilandmarks": self.spn_border.value(),
        }

    def _on_detect(self):
        if self._current_mesh is None:
            self._log("No mesh loaded.")
            return
        try:
            result = self._engine.detect_all(**self._get_params())
            self._current_result  = result
            self._original_result = result.copy()

            path = self._state.current_path
            if path:
                self._batch_results[path]  = result.copy()
                self._batch_original[path] = result.copy()
                self._batch_status[path]   = "auto"

            self._picked_point = None
            self.lbl_picked.setText("Picked point: none")
            self.lbl_status.setText(
                f"Status: auto  |  {len(result.cusps)} cusps  "
                f"{len(result.intercuspal)} inter  "
                f"{len(result.border_semilandmarks)} semi"
            )
            self._update_edit_limits()
            self._refresh_scene()
            self._log(f"Detection OK: {len(result.cusps)} cusps detected.")

            # Show ordering warnings if any
            for w in self._engine._ordering_warnings:
                self._log(w)

        except Exception as e:
            self._log(f"Detection error: {e}")

    def _on_recompute(self):
        if self._current_result is None or self._current_mesh is None:
            return
        try:
            pts = self._current_mesh.points
            cusps = self._engine.order_points_consistently(self._current_result.cusps, tooth_type=self.cmb_tooth.currentText())
            inter = self._engine.detect_intercuspal_points(
                pts, cusps, tooth_type=self.cmb_tooth.currentText())
            semis, curve = self._engine.detect_semilandmarks_on_lower_border(
                self._current_mesh, cusps[0], self.spn_border.value())
            center = self._engine.detect_central_point(pts, cusps=cusps, border_points=semis)

            self._current_result.cusps = cusps
            self._current_result.intercuspal = inter
            self._current_result.border_semilandmarks = semis
            self._current_result.border_curve = curve
            self._current_result.center = center

            path = self._state.current_path
            if path:
                self._batch_results[path] = self._current_result.copy()
                self._batch_status[path]  = "edited"

            self.lbl_status.setText("Status: recomputed")
            self._refresh_scene()
            self._log("Dependent points recomputed.")
        except Exception as e:
            self._log(f"Recompute error: {e}")

    # ── Scene refresh ─────────────────────────────────────

    def _refresh_scene(self):
        plotter = self._viewer._plotter
        plotter.clear()

        # Mesh
        if self._current_mesh is not None:
            mesh = self._current_mesh.copy(deep=False)
            mesh["elevation"] = mesh.points[:, 2]
            plotter.add_mesh(
                mesh, scalars="elevation", cmap="gist_earth",
                smooth_shading=True, opacity=1.0,
                show_edges=False, name="mesh",
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": "Z (mm)", "title_font_size": 11,
                    "label_font_size": 10, "n_labels": 4,
                    "fmt": "%.1f", "width": 0.07, "height": 0.30,
                    "position_x": 0.92, "position_y": 0.06,
                    "vertical": True, "color": "white",
                    "shadow": True,
                },
            )
            plotter.enable_eye_dome_lighting()
        # Show in-progress manual cusps during placement
        if self._manual_placing and self._manual_cusps:
            pts = np.array(self._manual_cusps)
            plotter.add_mesh(pv.PolyData(pts),
                color="#FF3A1A", point_size=22,
                render_points_as_spheres=True, name="manual_cusps")
            self._add_number_labels(plotter, pts, "manual_cusps")

        r = self._current_result
        if r is None:
            plotter.view_xy(); plotter.reset_camera(); plotter.render()
            QTimer.singleShot(50, self._post_render)
            return

        # Cusps
        if self.chk_cusps.isChecked() and len(r.cusps) > 0:
            plotter.add_mesh(pv.PolyData(r.cusps),
                color="#FF3A1A", point_size=22, render_points_as_spheres=True, name="cusps")
            if self.chk_labels.isChecked():
                self._add_number_labels(plotter, r.cusps, "cusps")

        # Intercuspal
        if self.chk_inter.isChecked() and len(r.intercuspal) > 0:
            plotter.add_mesh(pv.PolyData(r.intercuspal),
                color="#1E90FF", point_size=16, render_points_as_spheres=True, name="inter")
            if self.chk_labels.isChecked():
                self._add_number_labels(plotter, r.intercuspal, "inter")

        # Center
        if self.chk_center.isChecked():
            plotter.add_mesh(pv.PolyData(np.array([r.center])),
                color="#00E5FF", point_size=26, render_points_as_spheres=True, name="center")

        # Border semilandmarks
        if self.chk_border.isChecked() and len(r.border_semilandmarks) > 0:
            plotter.add_mesh(pv.PolyData(r.border_semilandmarks),
                color="limegreen", point_size=12, render_points_as_spheres=True, name="border")

        # Border curve
        if self.chk_curve.isChecked() and len(r.border_curve) > 0:
            bc = np.vstack([r.border_curve, r.border_curve[0]])
            lines = []
            for i in range(len(bc) - 1):
                lines.extend([2, i, i + 1])
            poly = pv.PolyData()
            poly.points = bc
            poly.lines = np.array(lines)
            plotter.add_mesh(poly, color="yellow", line_width=2, name="border_curve")

        # Selected landmark (edit mode)
        if self.chk_edit.isChecked():
            sel = self._get_selected_landmark()
            if sel is not None:
                plotter.add_mesh(pv.PolyData(np.array([sel])),
                    color="magenta", point_size=28, render_points_as_spheres=True,
                    name="selected_lmk")
            if self._picked_point is not None:
                plotter.add_mesh(pv.PolyData(np.array([self._picked_point])),
                    color="orange", point_size=24, render_points_as_spheres=True,
                    name="picked_pt")

        plotter.view_xy()
        plotter.reset_camera()
        plotter.render()
        # Post-render: axes and scalar bar need a second render cycle
        QTimer.singleShot(50, self._post_render)


    def _post_render(self):
        """Second render pass: axes and scalar bar need this to appear on first load."""
        try:
            p = self._viewer._plotter
            p.show_axes()
            p.render()
        except Exception:
            pass

    def _add_number_labels(self, plotter, points: np.ndarray, prefix: str,
                           labels: list = None):
        if points is None or len(points) == 0:
            return
        center = np.mean(points[:, :2], axis=0)
        if self._current_mesh is not None:
            b = self._current_mesh.bounds
            diag = math.sqrt((b[1]-b[0])**2 + (b[3]-b[2])**2 + (b[5]-b[4])**2)
        else:
            diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        offset = max(1.5, 0.055 * diag)
        lbl_pts = points.copy()
        for i in range(len(points)):
            vec = np.array([points[i,0]-center[0], points[i,1]-center[1], 0.0])
            norm = np.linalg.norm(vec[:2])
            direction = vec / norm if norm > 1e-9 else np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
            lbl_pts[i, 0] += direction[0] * offset
            lbl_pts[i, 1] += direction[1] * offset
            lbl_pts[i, 2] += 0.02 * diag

        # Use anatomical names if provided, otherwise numbered
        if labels is not None and len(labels) >= len(points):
            display_labels = [labels[i] for i in range(len(points))]
        else:
            display_labels = [str(i+1) for i in range(len(points))]

        plotter.add_point_labels(
            lbl_pts, display_labels,
            font_size=22, point_size=1, shape_opacity=0.75,
            shape_color="white", text_color="black",
            always_visible=True, name=f"labels_{prefix}"
        )

    # ── Picking / editing ─────────────────────────────────

    def _on_mesh_clicked(self, point, *args):
        """Called by PyVista when user clicks in the viewer."""
        if point is None or self._current_mesh is None:
            return

        pt = np.asarray(point)
        idx = int(np.argmin(np.linalg.norm(self._current_mesh.points - pt, axis=1)))
        snapped = self._current_mesh.points[idx].copy()

        if self._manual_placing:
            self._on_manual_click(snapped)
            return

        if not self.chk_edit.isChecked():
            return
        self._picked_point = snapped
        self.lbl_picked.setText(f"Picked: {snapped[0]:.2f}, {snapped[1]:.2f}, {snapped[2]:.2f}")
        self._log("Picked new position on mesh.")
        self._refresh_scene()

    def _on_viewer_pick(self, label: str, xyz: np.ndarray):
        """Called by Viewer3D's own pick mode (if used)."""
        pass  # handled via _on_mesh_clicked above

    def _on_edit_mode_toggled(self, checked: bool):
        if checked and self._manual_placing:
            self._on_cancel_manual()
        self._log("Edit mode ON – click mesh to pick position." if checked else "Edit mode OFF.")
        self._refresh_scene()

    # ── Manual placement ─────────────────────────────────

    def _on_start_manual(self):
        if self._current_mesh is None:
            self._log("No mesh loaded.")
            return
        self.chk_edit.setChecked(False)
        n = self.spn_manual_cusps.value()
        self._manual_n_cusps = n
        self._manual_cusps.clear()
        self._manual_placing = True
        self.btn_start_manual.setEnabled(False)
        self.btn_undo_manual.setEnabled(True)
        self.btn_cancel_manual.setEnabled(True)

        tooth_type = self.cmb_tooth.currentText()
        names = self._engine.get_cusp_names(tooth_type, n)
        first_name = names[0] if names else "cusp 1"
        self.lbl_manual_status.setText(
            f"Click on mesh to place {first_name} (1/{n})"
        )
        self._log(f"Manual placement started: {n} cusps to place.")
        self._refresh_scene()

    def _on_manual_click(self, snapped: np.ndarray):
        n = self._manual_n_cusps
        self._manual_cusps.append(snapped.copy())
        placed = len(self._manual_cusps)

        tooth_type = self.cmb_tooth.currentText()
        names = self._engine.get_cusp_names(tooth_type, n)
        placed_name = names[placed - 1] if placed - 1 < len(names) else f"cusp {placed}"
        self._log(f"Placed {placed_name} at "
                  f"({snapped[0]:.2f}, {snapped[1]:.2f}, {snapped[2]:.2f})")

        if placed >= n:
            self._finalize_manual_placement()
        else:
            next_name = names[placed] if placed < len(names) else f"cusp {placed+1}"
            self.lbl_manual_status.setText(
                f"Click on mesh to place {next_name} ({placed+1}/{n})"
            )
            self._refresh_scene()

    def _finalize_manual_placement(self):
        self._manual_placing = False
        self.btn_start_manual.setEnabled(True)
        self.btn_undo_manual.setEnabled(False)
        self.btn_cancel_manual.setEnabled(False)

        cusps = np.array(self._manual_cusps)
        tooth_type = self.cmb_tooth.currentText()
        n_border = self.spn_border.value()
        pts = self._current_mesh.points

        try:
            self._engine.set_mesh(self._current_mesh)
            intercuspal = self._engine.detect_intercuspal_points(
                pts, cusps, tooth_type=tooth_type)
            border_semis, border_curve = self._engine.detect_semilandmarks_on_lower_border(
                self._current_mesh, cusps[0], n_samples=n_border)
            center = self._engine.detect_central_point(
                pts, cusps=cusps, border_points=border_semis)

            result = DetectionResult(
                cusps=cusps,
                intercuspal=intercuspal,
                center=center,
                border_semilandmarks=border_semis,
                border_curve=border_curve,
            )
            self._current_result = result
            self._original_result = result.copy()

            path = self._state.current_path
            if path:
                self._batch_results[path] = result.copy()
                self._batch_original[path] = result.copy()
                self._batch_status[path] = "manual"

            self.lbl_manual_status.setText(
                f"Done: {len(cusps)} cusps placed manually.\n"
                f"Intercuspal, center and semilandmarks computed.\n"
                f"Use Edit mode to refine individual points."
            )
            self.lbl_status.setText(
                f"Status: manual  |  {len(cusps)} cusps  "
                f"{len(intercuspal)} inter  "
                f"{len(border_semis)} semi"
            )
            self._update_edit_limits()
            self._refresh_scene()
            self._log(f"Manual placement complete. "
                      f"Dependent landmarks auto-computed.")
        except Exception as e:
            self.lbl_manual_status.setText(f"Error computing dependent landmarks: {e}")
            self._log(f"Manual placement error: {e}")

    def _on_undo_manual(self):
        if not self._manual_placing or not self._manual_cusps:
            return
        removed = self._manual_cusps.pop()
        remaining = len(self._manual_cusps)
        n = self._manual_n_cusps
        tooth_type = self.cmb_tooth.currentText()
        names = self._engine.get_cusp_names(tooth_type, n)
        next_name = names[remaining] if remaining < len(names) else f"cusp {remaining+1}"
        self.lbl_manual_status.setText(
            f"Undone. Click to place {next_name} ({remaining+1}/{n})"
        )
        self._log(f"Undid last cusp. {remaining}/{n} placed.")
        self._refresh_scene()

    def _on_cancel_manual(self):
        self._manual_placing = False
        self._manual_cusps.clear()
        self.btn_start_manual.setEnabled(True)
        self.btn_undo_manual.setEnabled(False)
        self.btn_cancel_manual.setEnabled(False)
        self.lbl_manual_status.setText("")
        self._log("Manual placement cancelled.")
        self._refresh_scene()

    def _update_edit_limits(self):
        if self._current_result is None:
            self.spn_edit_idx.setRange(1, 1)
            return
        group = self.cmb_edit_group.currentText()
        if group == "Cusps":
            n = len(self._current_result.cusps)
        elif group == "Intercuspal minima":
            n = len(self._current_result.intercuspal)
        else:
            n = 1
        self.spn_edit_idx.setRange(1, max(1, n))

    def _get_selected_landmark(self) -> Optional[np.ndarray]:
        if self._current_result is None:
            return None
        group = self.cmb_edit_group.currentText()
        idx = self.spn_edit_idx.value() - 1
        if group == "Cusps" and 0 <= idx < len(self._current_result.cusps):
            return self._current_result.cusps[idx]
        if group == "Intercuspal minima" and 0 <= idx < len(self._current_result.intercuspal):
            return self._current_result.intercuspal[idx]
        if group == "Center":
            return self._current_result.center
        return None

    def _on_apply_pick(self):
        if self._current_result is None or self._picked_point is None:
            self._log("No picked point. Click on the mesh first.")
            return
        group = self.cmb_edit_group.currentText()
        idx = self.spn_edit_idx.value() - 1
        if group == "Cusps":
            self._current_result.cusps[idx] = self._picked_point.copy()
            self._log(f"Updated cusp {idx+1}.")
        elif group == "Intercuspal minima":
            self._current_result.intercuspal[idx] = self._picked_point.copy()
            self._log(f"Updated intercuspal {idx+1}.")
        elif group == "Center":
            self._current_result.center = self._picked_point.copy()
            self._log("Updated center.")
        path = self._state.current_path
        if path:
            self._batch_results[path] = self._current_result.copy()
            self._batch_status[path] = "edited"
        self.lbl_status.setText("Status: edited")
        self._refresh_scene()

    def _on_reset_auto(self):
        if self._original_result is None:
            return
        self._current_result = self._original_result.copy()
        path = self._state.current_path
        if path:
            self._batch_results[path] = self._current_result.copy()
            self._batch_status[path] = "auto"
        self._picked_point = None
        self.lbl_picked.setText("Picked point: none")
        self.lbl_status.setText("Status: auto (restored)")
        self._update_edit_limits()
        self._refresh_scene()
        self._log("Restored automatic landmarks.")

    # ── Export ────────────────────────────────────────────

    _FMT_EXTENSIONS = {
        "CSV (DAMOS)":          ".csv",
        "CSV wide (geomorph/R)":".csv",
        "Morphologika":         ".txt",
        "TPS":                  ".tps",
        "NTS (MorphoJ)":        ".nts",
        "JSON":                 ".json",
        "FCSV (3D Slicer)":     ".fcsv",
    }

    def _on_export_csv(self):
        if self._current_result is None:
            self._log("No result to export.")
            return
        fmt = self.cmb_export_fmt.currentText()
        ext = self._FMT_EXTENSIONS.get(fmt, ".csv")
        path, _ = QFileDialog.getSaveFileName(
            self, f"Save {fmt}", f"landmarks{ext}",
            f"Landmark files (*{ext});;All files (*)"
        )
        if not path:
            return
        tooth_type = self.cmb_tooth.currentText()
        specimen = os.path.splitext(os.path.basename(
            self._state.current_path or "specimen"))[0]
        self._write_single(path, self._current_result, tooth_type, specimen, fmt)
        self._log(f"Saved ({fmt}): {path}")

    def _on_export_all(self):
        if not self._batch_results:
            self._log("No batch results to export.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select export folder")
        if not folder:
            return
        fmt = self.cmb_export_fmt.currentText()
        ext = self._FMT_EXTENSIONS.get(fmt, ".csv")
        tooth_type = self.cmb_tooth.currentText()

        # Per-specimen files
        for fpath, result in self._batch_results.items():
            base = os.path.splitext(os.path.basename(fpath))[0]
            out = os.path.join(folder, f"{base}_landmarks{ext}")
            self._write_single(out, result, tooth_type, base, fmt)

        # Combined file (formats that support multi-specimen)
        if fmt == "CSV (DAMOS)":
            self._write_all_csv_damos(os.path.join(folder, "ALL_Landmarks.csv"), tooth_type)
        elif fmt == "CSV wide (geomorph/R)":
            self._write_all_csv_wide(os.path.join(folder, "ALL_Landmarks_wide.csv"), tooth_type)
        elif fmt == "Morphologika":
            self._write_all_morphologika(os.path.join(folder, "ALL_Landmarks.txt"), tooth_type)
        elif fmt == "TPS":
            self._write_all_tps(os.path.join(folder, "ALL_Landmarks.tps"), tooth_type)
        elif fmt == "NTS (MorphoJ)":
            self._write_all_nts(os.path.join(folder, "ALL_Landmarks.nts"), tooth_type)
        elif fmt == "JSON":
            self._write_all_json(os.path.join(folder, "ALL_Landmarks.json"), tooth_type)
        elif fmt == "FCSV (3D Slicer)":
            pass  # FCSV is per-specimen only

        self._log(f"Exported {len(self._batch_results)} files ({fmt}) to {folder}")

    # ── Format writers ────────────────────────────────────

    def _get_landmark_rows(self, result: DetectionResult, tooth_type: str):
        """Build ordered list of (label, x, y, z) for a result."""
        cusp_names = self._engine.get_cusp_names(tooth_type, len(result.cusps))
        rows = []
        for i, p in enumerate(result.cusps):
            name = cusp_names[i] if i < len(cusp_names) else f"cusp{i+1}"
            rows.append((name, p[0], p[1], p[2]))
        for i, p in enumerate(result.intercuspal, 1):
            rows.append((f"inter{i}", p[0], p[1], p[2]))
        rows.append(("center", result.center[0], result.center[1], result.center[2]))
        for i, p in enumerate(result.border_semilandmarks, 1):
            rows.append((f"semi{i}", p[0], p[1], p[2]))
        return rows

    def _write_single(self, path, result, tooth_type, specimen, fmt):
        """Dispatch to the appropriate single-specimen writer."""
        if fmt == "CSV (DAMOS)":
            self._write_csv_damos(path, result, tooth_type)
        elif fmt == "CSV wide (geomorph/R)":
            self._write_csv_wide(path, result, tooth_type, specimen)
        elif fmt == "Morphologika":
            self._write_morphologika(path, result, tooth_type, specimen)
        elif fmt == "TPS":
            self._write_tps(path, result, tooth_type, specimen)
        elif fmt == "NTS (MorphoJ)":
            self._write_nts(path, result, tooth_type, specimen)
        elif fmt == "JSON":
            self._write_json(path, result, tooth_type, specimen)
        elif fmt == "FCSV (3D Slicer)":
            self._write_fcsv(path, result, tooth_type, specimen)

    # ── CSV DAMOS (original) ──

    def _write_csv_damos(self, filepath, result, tooth_type):
        rows = self._get_landmark_rows(result, tooth_type)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("type,x,y,z\n")
            for name, x, y, z in rows:
                f.write(f"{name},{x},{y},{z}\n")

    def _write_all_csv_damos(self, filepath, tooth_type):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("file,type,x,y,z\n")
            for fpath, result in self._batch_results.items():
                fname = os.path.basename(fpath)
                for name, x, y, z in self._get_landmark_rows(result, tooth_type):
                    f.write(f"{fname},{name},{x},{y},{z}\n")

    # ── CSV wide (geomorph/R) ──

    def _write_csv_wide(self, filepath, result, tooth_type, specimen):
        rows = self._get_landmark_rows(result, tooth_type)
        header_parts = []
        val_parts = []
        for name, x, y, z in rows:
            header_parts.extend([f"{name}_X", f"{name}_Y", f"{name}_Z"])
            val_parts.extend([str(x), str(y), str(z)])
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("specimen," + ",".join(header_parts) + "\n")
            f.write(specimen + "," + ",".join(val_parts) + "\n")

    def _write_all_csv_wide(self, filepath, tooth_type):
        # Build header from first specimen
        first_result = next(iter(self._batch_results.values()))
        first_rows = self._get_landmark_rows(first_result, tooth_type)
        header_parts = []
        for name, _, _, _ in first_rows:
            header_parts.extend([f"{name}_X", f"{name}_Y", f"{name}_Z"])

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("specimen," + ",".join(header_parts) + "\n")
            for fpath, result in self._batch_results.items():
                specimen = os.path.splitext(os.path.basename(fpath))[0]
                rows = self._get_landmark_rows(result, tooth_type)
                vals = []
                for _, x, y, z in rows:
                    vals.extend([str(x), str(y), str(z)])
                f.write(specimen + "," + ",".join(vals) + "\n")

    # ── Morphologika ──

    def _write_morphologika(self, filepath, result, tooth_type, specimen):
        rows = self._get_landmark_rows(result, tooth_type)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"[individuals]\n1\n")
            f.write(f"[landmarks]\n{len(rows)}\n")
            f.write(f"[dimensions]\n3\n")
            f.write(f"[names]\n{specimen}\n")
            f.write(f"[labels]\n")
            for name, _, _, _ in rows:
                f.write(f"{name}\n")
            f.write(f"[rawpoints]\n")
            f.write(f"'{specimen}\n")
            for _, x, y, z in rows:
                f.write(f"{x} {y} {z}\n")

    def _write_all_morphologika(self, filepath, tooth_type):
        n_spec = len(self._batch_results)
        first_result = next(iter(self._batch_results.values()))
        first_rows = self._get_landmark_rows(first_result, tooth_type)
        n_lm = len(first_rows)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"[individuals]\n{n_spec}\n")
            f.write(f"[landmarks]\n{n_lm}\n")
            f.write(f"[dimensions]\n3\n")
            f.write(f"[names]\n")
            for fpath in self._batch_results:
                f.write(f"{os.path.splitext(os.path.basename(fpath))[0]}\n")
            f.write(f"[labels]\n")
            for name, _, _, _ in first_rows:
                f.write(f"{name}\n")
            f.write(f"[rawpoints]\n")
            for fpath, result in self._batch_results.items():
                specimen = os.path.splitext(os.path.basename(fpath))[0]
                f.write(f"'{specimen}\n")
                for _, x, y, z in self._get_landmark_rows(result, tooth_type):
                    f.write(f"{x} {y} {z}\n")

    # ── TPS ──

    def _write_tps(self, filepath, result, tooth_type, specimen):
        rows = self._get_landmark_rows(result, tooth_type)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"LM3={len(rows)}\n")
            for _, x, y, z in rows:
                f.write(f"{x} {y} {z}\n")
            f.write(f"ID={specimen}\n")

    def _write_all_tps(self, filepath, tooth_type):
        with open(filepath, "w", encoding="utf-8") as f:
            for fpath, result in self._batch_results.items():
                specimen = os.path.splitext(os.path.basename(fpath))[0]
                rows = self._get_landmark_rows(result, tooth_type)
                f.write(f"LM3={len(rows)}\n")
                for _, x, y, z in rows:
                    f.write(f"{x} {y} {z}\n")
                f.write(f"ID={specimen}\n")

    # ── NTS (MorphoJ) ──

    def _write_nts(self, filepath, result, tooth_type, specimen):
        rows = self._get_landmark_rows(result, tooth_type)
        n_lm = len(rows)
        with open(filepath, "w", encoding="utf-8") as f:
            # Header: 1 individuals, n_lm landmarks, 3 dimensions, 0=rectangular
            f.write(f"1 {n_lm} 3 0\n")
            for _, x, y, z in rows:
                f.write(f"{x} {y} {z}\n")

    def _write_all_nts(self, filepath, tooth_type):
        n_spec = len(self._batch_results)
        first_result = next(iter(self._batch_results.values()))
        n_lm = len(self._get_landmark_rows(first_result, tooth_type))
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"{n_spec} {n_lm} 3 0\n")
            for fpath, result in self._batch_results.items():
                for _, x, y, z in self._get_landmark_rows(result, tooth_type):
                    f.write(f"{x} {y} {z}\n")

    # ── JSON ──

    def _write_json(self, filepath, result, tooth_type, specimen):
        import json
        rows = self._get_landmark_rows(result, tooth_type)
        cusp_names = self._engine.get_cusp_names(tooth_type, len(result.cusps))

        data = {
            "format": "DAMOS_landmarks_v1",
            "specimen": specimen,
            "tooth_type": tooth_type,
            "n_landmarks": len(rows),
            "landmark_labels": [r[0] for r in rows],
            "cusps": {
                cusp_names[i] if i < len(cusp_names) else f"cusp{i+1}": {
                    "x": float(p[0]), "y": float(p[1]), "z": float(p[2])
                } for i, p in enumerate(result.cusps)
            },
            "intercuspal": {
                f"inter{i+1}": {
                    "x": float(p[0]), "y": float(p[1]), "z": float(p[2])
                } for i, p in enumerate(result.intercuspal)
            },
            "center": {
                "x": float(result.center[0]),
                "y": float(result.center[1]),
                "z": float(result.center[2]),
            },
            "semilandmarks": [
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                for p in result.border_semilandmarks
            ],
            "coordinates_matrix": [[float(x), float(y), float(z)]
                                   for _, x, y, z in rows],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _write_all_json(self, filepath, tooth_type):
        import json
        specimens = []
        for fpath, result in self._batch_results.items():
            specimen = os.path.splitext(os.path.basename(fpath))[0]
            rows = self._get_landmark_rows(result, tooth_type)
            cusp_names = self._engine.get_cusp_names(tooth_type, len(result.cusps))
            specimens.append({
                "specimen": specimen,
                "tooth_type": tooth_type,
                "n_landmarks": len(rows),
                "landmark_labels": [r[0] for r in rows],
                "coordinates_matrix": [[float(x), float(y), float(z)]
                                       for _, x, y, z in rows],
            })
        data = {
            "format": "DAMOS_landmarks_v1",
            "n_specimens": len(specimens),
            "specimens": specimens,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ── FCSV (3D Slicer) ──

    def _write_fcsv(self, filepath, result, tooth_type, specimen):
        rows = self._get_landmark_rows(result, tooth_type)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# Markups fiducial file version = 4.11\n")
            f.write("# CoordinateSystem = LPS\n")
            f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,"
                    "label,desc,associatedNodeID\n")
            for i, (name, x, y, z) in enumerate(rows):
                f.write(f"vtkMRMLMarkupsFiducialNode_{i},"
                        f"{x},{y},{z},"
                        f"0,0,0,1,1,1,0,"
                        f"{name},,\n")

    # ── Batch ─────────────────────────────────────────────

    def _browse_batch(self):
        folder = QFileDialog.getExistingDirectory(self, "Select input folder")
        if folder:
            self.le_batch.setText(folder)

    def _on_batch_run(self):
        input_dir = self.le_batch.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            # Use file list from state if no folder selected
            files = self._state.files
            if not files:
                self._log("No folder selected and no files loaded.")
                return
        else:
            files = sorted([
                os.path.join(input_dir, f)
                for f in os.listdir(input_dir)
                if f.lower().endswith((".ply", ".obj"))
            ])
        if not files:
            self._log("No PLY/OBJ files found.")
            return

        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.btn_batch_run.setEnabled(False)
        self.btn_batch_abort.setEnabled(True)

        params = self._get_params()
        self._worker = _LMKWorker(files, params)
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_batch_progress)
        self._worker.finished.connect(self._on_batch_done)
        self._worker_thread.start()

    def _on_batch_abort(self):
        if self._worker:
            self._worker.abort()

    def _on_batch_progress(self, current: int, total: int, msg: str, result):
        self.progress_bar.setValue(int(current / total * 100))
        self._log(f"[{current}/{total}] {msg}")
        # Cache result using path stored in worker
        if result is not None and hasattr(self._worker, '_files'):
            idx = current - 1
            if 0 <= idx < len(self._worker._files):
                path = self._worker._files[idx]
                self._batch_results[path]  = result
                self._batch_original[path] = result.copy()
                self._batch_status[path]   = "auto"
                # Update viewer if this is the currently displayed tooth
                if self._state.current_path == path:
                    self._current_result  = result
                    self._original_result = result.copy()
                    self.lbl_status.setText(
                        f"Status: auto  |  {len(result.cusps)} cusps")
                    self._update_edit_limits()
                    QTimer.singleShot(10, self._refresh_scene)

    def _on_batch_done(self):
        self.btn_batch_run.setEnabled(True)
        self.btn_batch_abort.setEnabled(False)
        self.progress_bar.setValue(100)
        self._log(f"Batch done. {len(self._batch_results)} meshes processed.")
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread.wait()
        self._state.post_status("AutoLMK batch complete.")

    def _log(self, msg: str):
        self.log_box.appendPlainText(msg)
        self.log_box.ensureCursorVisible()