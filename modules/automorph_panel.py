"""
DAMOS – AutoMorph / Topography panel
Uses TopographyEngine from ofc_engine.py (ofc.py original).

Metrics computed:
  mean_slope    – mean face slope angle (degrees)
  slope_sd      – standard deviation of slope
  simple_relief – max Z - min Z (mm)
  area_3d       – real 3D surface area (mm²)
  area_2d       – projected 2D area (mm²)
  relief        – 100 × (area_3d / area_2d)
  rfi_1         – area_3d / area_2d  (Relief Feature Index)
  rfi_2         – ln(√(area_3d/area_2d))
  dne           – Dirichlet Normal Energy (Bunn et al. 2011)
  opcr          – Orientation Patch Count Restricted (Boyer et al. 2010)
  md            – mesiodistal bounding-box extent (mm)  ⚠ requires oriented mesh
  bl            – buccolingual bounding-box extent (mm) ⚠ requires oriented mesh
"""

import os
import sys
from typing import Optional, Dict

import numpy as np
import pyvista as pv

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QPushButton, QSpinBox,
    QPlainTextEdit, QProgressBar, QFileDialog,
    QScrollArea, QFrame, QLabel, QLineEdit, QComboBox
)
from PySide6.QtCore import QThread, Signal, QObject, QTimer

from gui.app_state import AppState
from gui.viewer3d import Viewer3D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from ofc_engine import TopographyEngine, TopographyResult
    _ENGINE_OK = True
except ImportError:
    _ENGINE_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# PCV — Projected Cavity Visibility
# ─────────────────────────────────────────────────────────────────────────────

def _compute_pcv(mesh: pv.PolyData,
                 search_radius_frac: float = 0.20,
                 max_faces: int = 5000) -> np.ndarray:
    """Per-vertex Projected Cavity Visibility.

    For each vertex, measures its relative elevation within a local XY disk:
      1.0 → cusp / ridge  (fully exposed to sky)
      0.0 → fovea floor   (fully occluded)

    Parameters
    ----------
    mesh              : pv.PolyData — input surface
    search_radius_frac: disk radius as fraction of mean XY extent (default 0.20)
    max_faces         : if mesh has more faces, decimate before computing

    Returns
    -------
    pcv : (N,) float64 array in [0, 1], one value per vertex of *mesh*
    """
    from scipy.spatial import cKDTree

    original_pts = np.asarray(mesh.points, dtype=np.float64)

    # Optional internal decimation for speed
    work_mesh = mesh
    decimated = False
    n_cells = mesh.n_cells
    if n_cells > max_faces:
        try:
            ratio = 1.0 - max_faces / n_cells
            work_mesh = mesh.decimate(ratio)
            decimated = True
        except Exception:
            work_mesh = mesh

    pts = np.asarray(work_mesh.points, dtype=np.float64)
    z   = pts[:, 2]

    xy_extent = np.ptp(pts[:, :2], axis=0)
    radius    = float(np.mean(xy_extent)) * search_radius_frac

    tree  = cKDTree(pts[:, :2])
    pairs = tree.query_pairs(radius, output_type='ndarray')  # (M, 2)

    local_max = z.copy()
    local_min = z.copy()

    if len(pairs):
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]
        np.maximum.at(local_max, i_idx, z[j_idx])
        np.maximum.at(local_max, j_idx, z[i_idx])
        np.minimum.at(local_min, i_idx, z[j_idx])
        np.minimum.at(local_min, j_idx, z[i_idx])

    span = local_max - local_min
    span[span < 1e-10] = 1.0
    pcv_work = (z - local_min) / span

    if not decimated:
        return pcv_work

    # Nearest-neighbor interpolation back to original resolution
    tree_work = cKDTree(pts)
    _, idx    = tree_work.query(original_pts)
    return pcv_work[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

class _TopoWorker(QObject):
    progress = Signal(int, int, str, object)
    finished = Signal()

    def __init__(self, files: list, params: dict, output_dir: str):
        super().__init__()
        self._files      = files
        self._params     = params
        self._output_dir = output_dir
        self._abort      = False

    def abort(self):
        self._abort = True

    def run(self):
        if not _ENGINE_OK:
            self.progress.emit(0, 1, "ERROR: ofc_engine.py not found.", None)
            self.finished.emit()
            return

        engine = TopographyEngine()
        n      = len(self._files)
        batch_results: Dict[str, TopographyResult] = {}
        os.makedirs(self._output_dir, exist_ok=True)

        for i, path in enumerate(self._files, 1):
            if self._abort:
                self.progress.emit(i, n, "Aborted.", None)
                break
            fname = os.path.basename(path)
            try:
                engine.load_mesh(path)
                result = engine.compute_metrics(
                    smooth_iterations  = self._params.get("smooth_iter", 0),
                    target_faces       = self._params.get("target_faces", 0),
                    set_origin_to_lowest = True,
                )
                batch_results[path] = result

                base     = os.path.splitext(fname)[0]
                csv_path = os.path.join(self._output_dir, f"{base}_topo.csv")
                engine.export_single_csv(csv_path, result)

                self.progress.emit(
                    i, n,
                    f"OK  {fname}   slope={result.mean_slope:.2f}°  "
                    f"RFI={result.rfi_1:.4f}  relief={result.simple_relief:.3f}",
                    result
                )
            except Exception as e:
                self.progress.emit(i, n, f"FAIL  {fname}: {e}", None)

        if batch_results:
            combined = os.path.join(self._output_dir, "ALL_Topography.csv")
            try:
                engine.export_batch_csv(combined, batch_results, self._files)
            except Exception:
                pass

        self.finished.emit()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: label that shows its tooltip immediately on hover
# ─────────────────────────────────────────────────────────────────────────────

class _HintLabel(QLabel):
    def enterEvent(self, event):
        from PySide6.QtWidgets import QToolTip
        if self.toolTip():
            QToolTip.showText(
                self.mapToGlobal(self.rect().center()),
                self.toolTip(), self
            )
        super().enterEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Panel
# ─────────────────────────────────────────────────────────────────────────────

class AutoMorphPanel(QWidget):
    """AutoMorph / Topography panel – slope, relief, RFI, DNE, OPCR, MD, BL metrics."""

    def __init__(self, state: AppState, viewer: Viewer3D, parent=None):
        super().__init__(parent)
        self._state   = state
        self._viewer  = viewer
        self._engine  = TopographyEngine() if _ENGINE_OK else None
        self._current_result: Optional[TopographyResult] = None
        self._pcv_scalars: Optional[np.ndarray] = None
        self._worker: Optional[_TopoWorker] = None
        self._worker_thread: Optional[QThread] = None

        self._build_ui()
        self._connect_signals()

    # ── UI ─────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        if not _ENGINE_OK:
            warn = QLabel("ofc_engine.py not found.\nPlace ofc.py in the DAMOS root folder and restart.")
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #F0A030; padding: 12px;")
            root.addWidget(warn)
            root.addStretch(1)
            return

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner  = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── Prep ───────────────────────────────────────────
        grp_prep = QGroupBox("Mesh Preparation")
        f_prep   = QFormLayout(grp_prep)

        self.spn_smooth = QSpinBox()
        self.spn_smooth.setRange(0, 100); self.spn_smooth.setValue(0)
        f_prep.addRow("Smooth iterations:", self.spn_smooth)

        self.spn_faces = QSpinBox()
        self.spn_faces.setRange(0, 200_000)
        self.spn_faces.setSingleStep(1000); self.spn_faces.setValue(0)
        self.spn_faces.setToolTip("0 = no decimation")
        f_prep.addRow("Target faces (0=off):", self.spn_faces)

        layout.addWidget(grp_prep)

        # ── View style ──────────────────────────────────────
        grp_view = QGroupBox("Display Style")
        f_view   = QFormLayout(grp_view)
        self.cmb_style = QComboBox()
        self.cmb_style.addItems(["Elevation", "Slope", "Shaded", "DNE", "Orientation", "PCV Skyline"])
        f_view.addRow("Colour map:", self.cmb_style)
        layout.addWidget(grp_view)

        # ── Current mesh ───────────────────────────────────
        grp_cur = QGroupBox("Current Mesh")
        f_cur   = QVBoxLayout(grp_cur)

        self.btn_compute = QPushButton("Compute metrics")
        self.btn_compute.setObjectName("PrimaryButton")
        self.btn_compute.setMinimumHeight(36)
        f_cur.addWidget(self.btn_compute)

        # Results table
        metrics = [
            ("Mean slope (°):", "lbl_slope"),
            ("Slope SD (°):",   "lbl_slope_sd"),
            ("Simple relief:",  "lbl_relief_s"),
            ("Area 3D (mm²):",  "lbl_area3d"),
            ("Area 2D (mm²):",  "lbl_area2d"),
            ("Relief (%):",     "lbl_relief"),
            ("RFI₁:",           "lbl_rfi1"),
            ("RFI₂:",           "lbl_rfi2"),
            ("DNE:",            "lbl_dne"),
            ("OPCR:",           "lbl_opcr"),
            ("Mesiodist. (mm):","lbl_md"),
            ("Bucoling. (mm):", "lbl_bl"),
            ("PCV Mean:",       "lbl_pcv_mean"),
            ("PCV Min:",        "lbl_pcv_min"),
        ]
        _md_tooltip = (
            "⚠ Bounding-box extent along the X axis.\n"
            "Anatomically meaningful only if the mesh has been\n"
            "oriented with the occlusal plane ≈ XY (use MeshOrient)."
        )
        _bl_tooltip = (
            "⚠ Bounding-box extent along the Y axis.\n"
            "Anatomically meaningful only if the mesh has been\n"
            "oriented with the occlusal plane ≈ XY (use MeshOrient)."
        )

        f_res = QFormLayout()
        for label, attr in metrics:
            tip = (
                _md_tooltip if attr == "lbl_md" else
                _bl_tooltip if attr == "lbl_bl" else
                None
            )
            if tip:
                lbl     = _HintLabel("—"); lbl.setToolTip(tip)
                row_lbl = _HintLabel(label); row_lbl.setToolTip(tip)
            else:
                lbl     = QLabel("—")
                row_lbl = QLabel(label)
            lbl.setObjectName("ValueLabel")
            setattr(self, attr, lbl)
            f_res.addRow(row_lbl, lbl)
        f_cur.addLayout(f_res)

        layout.addWidget(grp_cur)

        # ── Export ─────────────────────────────────────────
        grp_exp = QGroupBox("Export")
        f_exp   = QVBoxLayout(grp_exp)
        self.btn_export_csv = QPushButton("Save current CSV")
        self.btn_export_csv.setObjectName("NavButton")
        f_exp.addWidget(self.btn_export_csv)
        layout.addWidget(grp_exp)

        # ── Batch ──────────────────────────────────────────
        grp_batch = QGroupBox("Batch")
        f_batch   = QVBoxLayout(grp_batch)

        row_in = QHBoxLayout()
        self.le_input = QLineEdit()
        self.le_input.setPlaceholderText("Input folder with PLY/OBJ…")
        self.le_input.setReadOnly(True)
        self.btn_browse_in = QPushButton("…")
        self.btn_browse_in.setFixedWidth(32)
        row_in.addWidget(self.le_input)
        row_in.addWidget(self.btn_browse_in)
        f_batch.addLayout(row_in)

        row_out = QHBoxLayout()
        self.le_output = QLineEdit()
        self.le_output.setPlaceholderText("Output folder (auto if empty)")
        self.le_output.setReadOnly(True)
        self.btn_browse_out = QPushButton("…")
        self.btn_browse_out.setFixedWidth(32)
        row_out.addWidget(self.le_output)
        row_out.addWidget(self.btn_browse_out)
        f_batch.addLayout(row_out)

        row_btns = QHBoxLayout()
        self.btn_batch_run = QPushButton("Run Batch")
        self.btn_batch_run.setObjectName("PrimaryButton")
        self.btn_batch_abort = QPushButton("Abort")
        self.btn_batch_abort.setEnabled(False)
        row_btns.addWidget(self.btn_batch_run)
        row_btns.addWidget(self.btn_batch_abort)
        f_batch.addLayout(row_btns)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        f_batch.addWidget(self.progress_bar)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(160)
        self.log_box.setStyleSheet(
            "QPlainTextEdit { font-family: 'Consolas','Menlo',monospace; "
            "font-size: 11px; background: #0D1117; color: #8B9DC3; }"
        )
        f_batch.addWidget(self.log_box)

        layout.addWidget(grp_batch)
        layout.addStretch(1)

        scroll.setWidget(inner)
        root.addWidget(scroll)

    # ── Connections ─────────────────────────────────────────

    def _connect_signals(self):
        if not _ENGINE_OK:
            return
        self.btn_compute.clicked.connect(self._on_compute)
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        self.btn_browse_in.clicked.connect(self._browse_input)
        self.btn_browse_out.clicked.connect(self._browse_output)
        self.btn_batch_run.clicked.connect(self._on_batch_run)
        self.btn_batch_abort.clicked.connect(self._on_batch_abort)
        self.cmb_style.currentIndexChanged.connect(self._refresh_scene)
        self._state.current_index_changed.connect(self._on_index_changed)

    # ── Index change ───────────────────────────────────────

    def _on_index_changed(self, idx: int):
        path = self._state.current_path
        if path is None:
            return
        try:
            mesh = pv.read(path)
            self._engine.load_mesh(path)
            self._state.set_mesh(mesh)
            self._current_result = None
            self._pcv_scalars    = None
            self._reset_labels()
            self._viewer.display_mesh(mesh)
            self._state.post_status(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            self._state.post_status(f"Load error: {e}")

    # ── Compute ────────────────────────────────────────────

    def _on_compute(self):
        path = self._state.current_path
        if path is None:
            self._log("No mesh selected.")
            return
        try:
            self._engine.load_mesh(path)
            result = self._engine.compute_metrics(
                smooth_iterations=self.spn_smooth.value(),
                target_faces=self.spn_faces.value(),
                set_origin_to_lowest=True,
            )
            self._current_result = result
            # PCV computed together with the other metrics
            try:
                pcv = _compute_pcv(result.mesh_for_display)
                self._pcv_scalars = pcv
            except Exception as e:
                self._pcv_scalars = None
                self._log(f"PCV error: {e}")
            self._update_labels(result)
            self._refresh_scene()
            pcv_mean = float(np.mean(self._pcv_scalars)) if self._pcv_scalars is not None else float('nan')
            pcv_min  = float(np.min(self._pcv_scalars))  if self._pcv_scalars is not None else float('nan')
            self._log(
                f"slope={result.mean_slope:.2f}°±{result.slope_sd:.2f}  "
                f"RFI₁={result.rfi_1:.4f}  RFI₂={result.rfi_2:.4f}  "
                f"relief={result.simple_relief:.3f}  "
                f"DNE={result.dne:.4f}  OPCR={result.opcr:.3f}  "
                f"MD={result.md:.3f} mm  BL={result.bl:.3f} mm  "
                f"PCV_mean={pcv_mean:.4f}  PCV_min={pcv_min:.4f}"
            )
        except Exception as e:
            self._log(f"Compute error: {e}")

    # ── Scene ──────────────────────────────────────────────

    def _refresh_scene(self):
        plotter = self._viewer._plotter
        plotter.clear()

        style = self.cmb_style.currentText() if _ENGINE_OK else "Elevation"

        if self._current_result is not None:
            mesh = self._current_result.mesh_for_display.copy(deep=False)
        elif self._state.current_mesh is not None:
            mesh = self._state.current_mesh.copy(deep=False)
        else:
            plotter.render()
            return

        _sbar = dict(
            title_font_size=11, label_font_size=10,
            width=0.07, height=0.35,
            position_x=0.92, position_y=0.05,
            vertical=True, color="white", shadow=True,
        )

        if style == "Slope" and self._current_result is not None:
            slopes = self._engine.compute_triangle_slopes(mesh)
            mesh.cell_data["SlopeDeg"] = slopes
            plotter.add_mesh(
                mesh, scalars="SlopeDeg", cmap="turbo",
                smooth_shading=False, opacity=1.0,
                show_edges=False, name="mesh",
                show_scalar_bar=True,
                scalar_bar_args={**_sbar, "title": "Slope (°)",
                                 "n_labels": 5, "fmt": "%.1f"},
            )
        elif (style == "DNE"
              and self._current_result is not None
              and "DNE_face" in mesh.cell_data.keys()):
            plotter.add_mesh(
                mesh, scalars="DNE_face", cmap="inferno",
                smooth_shading=False, opacity=1.0,
                show_edges=False, name="mesh",
                show_scalar_bar=True,
                scalar_bar_args={**_sbar, "title": "DNE / face",
                                 "n_labels": 4, "fmt": "%.5f"},
            )
        elif (style == "Orientation"
              and self._current_result is not None
              and "OrientBin" in mesh.cell_data.keys()):
            plotter.add_mesh(
                mesh, scalars="OrientBin", cmap="tab10",
                clim=[0, 7],
                smooth_shading=False, opacity=1.0,
                show_edges=False, name="mesh",
                show_scalar_bar=True,
                scalar_bar_args={**_sbar, "title": "Orient. bin (0–7)",
                                 "n_labels": 8, "fmt": "%.0f"},
            )
        elif (style == "PCV Skyline"
              and self._pcv_scalars is not None
              and len(self._pcv_scalars) == mesh.n_points):
            mesh.point_data["PCV"] = self._pcv_scalars
            plotter.add_mesh(
                mesh, scalars="PCV", cmap="RdYlGn",
                clim=[0.0, 1.0],
                smooth_shading=True, opacity=1.0,
                show_edges=False, name="mesh",
                show_scalar_bar=True,
                scalar_bar_args={**_sbar, "title": "PCV (0=fovea, 1=cusp)",
                                 "n_labels": 5, "fmt": "%.2f"},
            )
        elif style == "Shaded":
            plotter.add_mesh(
                mesh, color="#D8C8B0",
                smooth_shading=True, specular=0.3, ambient=0.25, diffuse=0.8,
                opacity=1.0, show_edges=False, name="mesh",
            )
        else:  # Elevation (default o si DNE/Orientation no computat encara)
            mesh["elevation"] = mesh.points[:, 2]
            plotter.add_mesh(
                mesh, scalars="elevation", cmap="gist_earth",
                smooth_shading=True, opacity=1.0,
                show_edges=False, name="mesh",
                show_scalar_bar=True,
                scalar_bar_args={**_sbar, "title": "Z (mm)",
                                 "n_labels": 4, "fmt": "%.1f",
                                 "height": 0.30, "position_y": 0.06},
            )

        plotter.enable_eye_dome_lighting()
        plotter.view_xy()
        plotter.reset_camera()
        plotter.render()
        QTimer.singleShot(50, self._post_render)

    def _post_render(self):
        try:
            self._viewer._plotter.show_axes()
            self._viewer._plotter.render()
        except Exception:
            pass

    # ── Labels ─────────────────────────────────────────────

    def _update_labels(self, r: TopographyResult):
        self.lbl_slope.setText(f"{r.mean_slope:.3f}")
        self.lbl_slope_sd.setText(f"{r.slope_sd:.3f}")
        self.lbl_relief_s.setText(f"{r.simple_relief:.4f}")
        self.lbl_area3d.setText(f"{r.area_3d:.4f}")
        self.lbl_area2d.setText(f"{r.area_2d:.4f}")
        self.lbl_relief.setText(f"{r.relief:.4f}")
        self.lbl_rfi1.setText(f"{r.rfi_1:.6f}")
        self.lbl_rfi2.setText(f"{r.rfi_2:.6f}")
        self.lbl_dne.setText(f"{r.dne:.4f}")
        self.lbl_opcr.setText(f"{r.opcr:.3f}")
        self.lbl_md.setText(f"{r.md:.3f}")
        self.lbl_bl.setText(f"{r.bl:.3f}")
        if self._pcv_scalars is not None:
            self.lbl_pcv_mean.setText(f"{float(np.mean(self._pcv_scalars)):.4f}")
            self.lbl_pcv_min.setText(f"{float(np.min(self._pcv_scalars)):.4f}")

    def _reset_labels(self):
        for attr in ("lbl_slope","lbl_slope_sd","lbl_relief_s",
                     "lbl_area3d","lbl_area2d","lbl_relief","lbl_rfi1","lbl_rfi2",
                     "lbl_dne","lbl_opcr","lbl_md","lbl_bl",
                     "lbl_pcv_mean","lbl_pcv_min"):
            getattr(self, attr).setText("—")

    # ── Export ─────────────────────────────────────────────

    def _on_export_csv(self):
        if self._current_result is None:
            self._log("No result to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "topography.csv", "CSV (*.csv)")
        if not path:
            return
        self._engine.export_single_csv(path, self._current_result)
        self._log(f"Saved: {path}")

    # ── Batch ──────────────────────────────────────────────

    def _browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select input folder")
        if folder: self.le_input.setText(folder)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder: self.le_output.setText(folder)

    def _on_batch_run(self):
        input_dir = self.le_input.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            files = self._state.files
            if not files:
                self._log("No input folder selected and no files loaded.")
                return
            input_dir = os.path.dirname(files[0])
        else:
            files = sorted([
                os.path.join(input_dir, f)
                for f in os.listdir(input_dir)
                if f.lower().endswith((".ply", ".obj"))
            ])

        if not files:
            self._log("No PLY/OBJ files found.")
            return

        output_dir = self.le_output.text().strip() or \
                     os.path.join(input_dir, "topo_results")
        params = {
            "smooth_iter":  self.spn_smooth.value(),
            "target_faces": self.spn_faces.value(),
        }

        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.btn_batch_run.setEnabled(False)
        self.btn_batch_abort.setEnabled(True)

        self._worker        = _TopoWorker(files, params, output_dir)
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_done)
        self._worker_thread.start()

    def _on_batch_abort(self):
        if self._worker: self._worker.abort()

    def _on_worker_progress(self, current: int, total: int, msg: str, result):
        self.progress_bar.setValue(int(current / total * 100))
        self._log(f"[{current}/{total}] {msg}")
        if result is not None:
            self._current_result = result
            self._update_labels(result)
            QTimer.singleShot(10, self._refresh_scene)

    def _on_worker_done(self):
        self.btn_batch_run.setEnabled(True)
        self.btn_batch_abort.setEnabled(False)
        self.progress_bar.setValue(100)
        self._log("Batch done.")
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread.wait()
        self._state.post_status("AutoMorph batch complete.")

    def _log(self, msg: str):
        self.log_box.appendPlainText(msg)
        self.log_box.ensureCursorVisible()
