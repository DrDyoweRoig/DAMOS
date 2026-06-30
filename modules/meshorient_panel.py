"""
DAMOS – MeshOrient panel (v3)
==============================
Workflow:
  Manual     – rotate with sliders, then "Set as reference"
  Semi-auto  – PCA+flip+ICP to reference + fine-tune with sliders
  Automatic  – PCA+flip+ICP to reference (batch-ready)

Flagged items are highlighted in the file list panel.
"""

import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QGroupBox, QDoubleSpinBox,
    QFileDialog, QProgressBar, QSizePolicy, QListWidgetItem, QLineEdit,
)
from PySide6.QtCore import Qt, Signal, QThread, QObject
from PySide6.QtGui import QColor

import pyvista as pv

from orientation_engine import (
    OrientationMethod,
    OrientationResult,
    compute_reference_alignment,
    compute_manual_rotation,
    compute_three_point_orientation,
    apply_orientation,
)
from mesh_exporter import export_oriented_mesh, SUFFIX


# ── Background worker for batch ──────────────────────────

class _BatchWorker(QObject):
    progress = Signal(int, int, str)
    item_done = Signal(int, str, float)   # idx, "done"/"flagged"/"error", error_mm
    finished = Signal(int, int)           # done, flagged

    def __init__(self, files, ref_verts, threshold, output_dir=None, parent=None):
        super().__init__(parent)
        self._files = files
        self._ref_verts = ref_verts
        self._threshold = threshold
        self._output_dir = output_dir
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        done = flagged = 0
        total = len(self._files)

        for idx, fpath in enumerate(self._files):
            if self._cancelled:
                break

            self.progress.emit(idx, total, Path(fpath).name)

            try:
                mesh = pv.read(fpath)
                verts = np.array(mesh.points)
                verts = verts - np.mean(verts, axis=0)

                result = compute_reference_alignment(verts, self._ref_verts)
                oriented = apply_orientation(verts, result, center=False)
                oriented = oriented - np.mean(oriented, axis=0)

                try:
                    faces = mesh.regular_faces
                except Exception:
                    faces = np.zeros((0, 3), dtype=int)

                export_oriented_mesh(
                    original_path=Path(fpath),
                    oriented_vertices=oriented,
                    faces=faces,
                    output_dir=self._output_dir if self._output_dir else Path(fpath).parent,
                )

                err = result.icp_mean_error
                if err > self._threshold:
                    self.item_done.emit(idx, "flagged", err)
                    flagged += 1
                else:
                    self.item_done.emit(idx, "done", err)
                    done += 1

            except Exception as e:
                self.item_done.emit(idx, "error", 0.0)

        self.finished.emit(done, flagged)


class MeshOrientPanel(QWidget):
    """Right-side control panel for the MeshOrient module."""

    def __init__(self, state, viewer, parent=None):
        super().__init__(parent)
        self._state = state
        self._viewer = viewer

        self._current_mode = "Manual"
        self._current_result = None
        self._original_verts = None
        self._original_faces = None
        self._original_mesh = None
        self._reference_verts = None
        self._reference_name = None
        self._batch_thread = None
        self._flagged_indices = set()

        self._three_points: list = []
        self._picking_active = False
        self._pending_reload = False

        self.setMinimumWidth(280)
        self._build_ui()
        self._connect_signals()

    # ══════════════════════════════════════════════════════
    # UI
    # ══════════════════════════════════════════════════════

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("MeshOrient")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #E8F0FF;")
        layout.addWidget(title)
        subtitle = QLabel("Occlusal plane alignment to +Z")
        subtitle.setStyleSheet("font-size: 11px; color: #7A9AB8;")
        layout.addWidget(subtitle)
        layout.addSpacing(4)

        # ── Reference ────────────────────────────────────
        ref_group = QGroupBox("Reference mesh")
        rl = QVBoxLayout(ref_group)

        self._ref_label = QLabel("⚠ No reference set")
        self._ref_label.setWordWrap(True)
        self._ref_label.setStyleSheet("font-size: 11px; color: #F0A030;")
        rl.addWidget(self._ref_label)

        ref_btns = QHBoxLayout()
        self._btn_set_ref = QPushButton("Set current as reference")
        self._btn_set_ref.clicked.connect(self._on_set_reference)
        ref_btns.addWidget(self._btn_set_ref)
        self._btn_load_ref = QPushButton("Load .ply")
        self._btn_load_ref.clicked.connect(self._on_load_reference)
        ref_btns.addWidget(self._btn_load_ref)
        rl.addLayout(ref_btns)
        layout.addWidget(ref_group)

        # ── Mode ─────────────────────────────────────────
        mode_group = QGroupBox("Mode")
        mg = QHBoxLayout(mode_group)
        self._mode_btns = {}
        for mode in ("Manual", "Semi-auto", "Automatic", "3 Points"):
            btn = QPushButton(mode)
            btn.setCheckable(True)
            btn.setChecked(mode == "Manual")
            btn.clicked.connect(lambda _, m=mode: self._on_mode(m))
            mg.addWidget(btn)
            self._mode_btns[mode] = btn
        layout.addWidget(mode_group)

        # ── 3-point picking ──────────────────────────────
        self._three_pt_group = QGroupBox("3-Point plane orientation")
        tpl = QVBoxLayout(self._three_pt_group)
        tpl.addWidget(QLabel(
            "Pick 3 points on the occlusal surface.\n"
            "They define a plane that will be aligned to Z+."
        ))
        self._tp_labels = []
        for i in range(3):
            lbl = QLabel(f"  Point {i+1}: —")
            lbl.setStyleSheet("font-size: 11px; color: #8B9DC3; font-family: monospace;")
            tpl.addWidget(lbl)
            self._tp_labels.append(lbl)

        tp_btns = QHBoxLayout()
        self._btn_start_pick = QPushButton("Start picking")
        self._btn_start_pick.setStyleSheet(
            "background: #1D6FA4; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 6px 16px;"
        )
        self._btn_start_pick.clicked.connect(self._on_start_pick_3pt)
        tp_btns.addWidget(self._btn_start_pick)

        self._btn_clear_pick = QPushButton("Clear points")
        self._btn_clear_pick.clicked.connect(self._on_clear_3pt)
        tp_btns.addWidget(self._btn_clear_pick)
        tpl.addLayout(tp_btns)

        self._btn_apply_3pt = QPushButton("Apply 3-point orientation")
        self._btn_apply_3pt.setStyleSheet(
            "background: #2A7A5A; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 6px 16px;"
        )
        self._btn_apply_3pt.clicked.connect(self._on_apply_3pt)
        tpl.addWidget(self._btn_apply_3pt)
        self._three_pt_group.setVisible(False)
        layout.addWidget(self._three_pt_group)

        # ── Sliders ──────────────────────────────────────
        self._adj_group = QGroupBox("Fine adjustment")
        al = QVBoxLayout(self._adj_group)
        self._pitch_slider = self._make_slider("Pitch (X)", al, 180)
        self._roll_slider = self._make_slider("Roll (Y)", al, 180)
        self._yaw_slider = self._make_slider("Yaw (Z)", al, 180)
        layout.addWidget(self._adj_group)

        # ── Actions ──────────────────────────────────────
        act_row = QHBoxLayout()
        self._btn_orient = QPushButton("Apply rotation")
        self._btn_orient.setStyleSheet(
            "background: #1D6FA4; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 6px 16px;"
        )
        self._btn_orient.clicked.connect(self._on_orient)
        self._btn_reset = QPushButton("Reset")
        self._btn_reset.clicked.connect(self._on_reset)
        self._btn_export = QPushButton(f"Export ({SUFFIX})")
        self._btn_export.clicked.connect(self._on_export)
        act_row.addWidget(self._btn_orient)

        self._btn_apply = QPushButton("Apply ✓")
        self._btn_apply.setStyleSheet(
            "background: #2A7A5A; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 6px 16px;"
        )
        self._btn_apply.setToolTip("Lock current orientation as the new base")
        self._btn_apply.clicked.connect(self._on_apply)
        act_row.addWidget(self._btn_apply)

        act_row.addWidget(self._btn_reset)
        act_row.addWidget(self._btn_export)
        layout.addLayout(act_row)

        # ── Batch ────────────────────────────────────────
        batch_group = QGroupBox("Batch processing")
        bl = QVBoxLayout(batch_group)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Error threshold:"))
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.01, 10.0)
        self._threshold_spin.setValue(0.50)
        self._threshold_spin.setSuffix(" mm")
        self._threshold_spin.setDecimals(2)
        thresh_row.addWidget(self._threshold_spin)
        bl.addLayout(thresh_row)

        outdir_row = QHBoxLayout()
        outdir_row.addWidget(QLabel("Output dir:"))
        self._out_dir_edit = QLineEdit()
        self._out_dir_edit.setPlaceholderText("Same folder as source (default)")
        self._out_dir_edit.setReadOnly(True)
        outdir_row.addWidget(self._out_dir_edit, 1)
        self._btn_out_dir = QPushButton("…")
        self._btn_out_dir.setFixedWidth(28)
        self._btn_out_dir.clicked.connect(self._on_pick_out_dir)
        outdir_row.addWidget(self._btn_out_dir)
        bl.addLayout(outdir_row)

        self._btn_batch = QPushButton("Orient all loaded meshes")
        self._btn_batch.clicked.connect(self._on_batch)
        bl.addWidget(self._btn_batch)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        bl.addWidget(self._progress)

        self._batch_summary = QLabel("")
        self._batch_summary.setWordWrap(True)
        self._batch_summary.setStyleSheet("font-size: 11px; color: #7A9AB8;")
        bl.addWidget(self._batch_summary)

        layout.addWidget(batch_group)

        # ── Status ───────────────────────────────────────
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("font-size: 11px; color: #7A9AB8;")
        layout.addWidget(self._status)

        layout.addStretch()
        self._update_ui_for_mode()

    def _make_slider(self, label, parent_layout, range_deg=45):
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setFixedWidth(65)
        lbl.setStyleSheet("color: #8B9DC3; font-size: 12px;")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(-range_deg * 10, range_deg * 10)
        slider.setValue(0)
        val_lbl = QLabel("0.0°")
        val_lbl.setFixedWidth(45)
        val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        val_lbl.setStyleSheet("font-family: monospace; font-size: 11px; color: #8B9DC3;")

        def on_change(v):
            val_lbl.setText(f"{v / 10:.1f}°")
            self._on_slider_changed()

        slider.valueChanged.connect(on_change)
        row.addWidget(lbl)
        row.addWidget(slider, stretch=1)
        row.addWidget(val_lbl)
        parent_layout.addLayout(row)
        return slider

    # ══════════════════════════════════════════════════════
    # Connections
    # ══════════════════════════════════════════════════════

    def _connect_signals(self):
        self._state.current_index_changed.connect(self._on_mesh_loaded)

    # ══════════════════════════════════════════════════════
    # Mode
    # ══════════════════════════════════════════════════════

    def _on_mode(self, mode):
        self._current_mode = mode
        for m, btn in self._mode_btns.items():
            btn.setChecked(m == mode)
        self._update_ui_for_mode()

    def _update_ui_for_mode(self):
        mode = self._current_mode
        self._adj_group.setVisible(mode in ("Manual", "Semi-auto"))
        self._three_pt_group.setVisible(mode == "3 Points")
        labels = {
            "Manual": "Apply rotation",
            "Semi-auto": "ICP + fine-tune",
            "Automatic": "ICP orient",
        }
        self._btn_orient.setText(labels.get(mode, "Orient"))
        self._btn_orient.setVisible(mode not in ("Manual", "3 Points"))
        if mode != "3 Points" and self._picking_active:
            self._picking_active = False
            self._viewer.set_click_handler(getattr(self, '_prev_click_handler', None))

    # ══════════════════════════════════════════════════════
    # Reference
    # ══════════════════════════════════════════════════════

    def _on_set_reference(self):
        viewer_mesh = self._viewer._mesh
        if viewer_mesh is None:
            self._status.setText("No mesh displayed.")
            return
        verts = np.array(viewer_mesh.points).copy()
        verts -= np.mean(verts, axis=0)
        self._reference_verts = verts
        files = self._state.files
        idx = self._state.current_index
        self._reference_name = Path(files[idx]).name if 0 <= idx < len(files) else "current"
        self._ref_label.setText(f"✓ Reference: {self._reference_name}")
        self._ref_label.setStyleSheet("font-size: 11px; color: #4ECDC4;")
        self._status.setText(f"Reference set: {self._reference_name}")

    def _on_load_reference(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load reference mesh", "",
            "Mesh files (*.ply *.stl *.obj *.off);;All (*)"
        )
        if not filepath:
            return
        try:
            mesh = pv.read(filepath)
            verts = np.array(mesh.points)
            verts -= np.mean(verts, axis=0)
            self._reference_verts = verts
            self._reference_name = Path(filepath).name
            self._ref_label.setText(f"✓ Reference: {self._reference_name}")
            self._ref_label.setStyleSheet("font-size: 11px; color: #4ECDC4;")
            self._status.setText(f"Reference loaded: {self._reference_name}")
        except Exception as e:
            self._status.setText(f"Error: {e}")

    def _require_reference(self) -> bool:
        if self._reference_verts is None:
            self._status.setText(
                "No reference set. Orient one mesh manually, then "
                "click 'Set current as reference'."
            )
            return False
        return True

    # ══════════════════════════════════════════════════════
    # Mesh loading
    # ══════════════════════════════════════════════════════

    def showEvent(self, event):
        super().showEvent(event)
        if self._pending_reload:
            self._on_mesh_loaded(self._state.current_index)

    def _on_mesh_loaded(self, idx):
        if not self.isVisible():
            self._pending_reload = True
            return
        self._pending_reload = False
        files = self._state.files
        if 0 <= idx < len(files):
            try:
                mesh = pv.read(files[idx])
                mesh.points = mesh.points - np.mean(mesh.points, axis=0)
                self._original_mesh = mesh.copy(deep=True)
                self._original_verts = np.array(mesh.points).copy()
                try:
                    self._original_faces = mesh.regular_faces
                except Exception:
                    self._original_faces = None
                self._current_result = None
                self._reset_sliders()
                self._three_points.clear()
                self._picking_active = False
                self._viewer.display_mesh(mesh.copy(deep=True))
                self._status.setText(f"Loaded: {Path(files[idx]).name}")
            except Exception as e:
                self._status.setText(f"Error: {e}")

    # ══════════════════════════════════════════════════════
    # Display
    # ══════════════════════════════════════════════════════

    def _show_oriented(self, vertices):
        if self._original_mesh is None:
            return
        centered = vertices - np.mean(vertices, axis=0)
        mesh = self._original_mesh.copy(deep=True)
        mesh.points = centered
        self._viewer.display_mesh(mesh)
        self._viewer.reset_camera()

    # ══════════════════════════════════════════════════════
    # Orient
    # ══════════════════════════════════════════════════════

    def _on_orient(self):
        if self._original_verts is None:
            self._status.setText("No mesh loaded.")
            return

        mode = self._current_mode

        if mode == "Manual":
            self._on_slider_changed()
            return

        if not self._require_reference():
            return

        verts = self._original_verts.copy()
        result = compute_reference_alignment(verts, self._reference_verts)
        self._current_result = result

        if not result.success:
            self._status.setText(f"Alignment failed: {result.message}")
            return

        oriented = apply_orientation(verts, result, center=False)
        self._show_oriented(oriented)
        self._status.setText(result.message)

        if mode == "Semi-auto":
            self._status.setText(f"{result.message} — adjust with sliders if needed")

    def _on_slider_changed(self):
        if self._original_verts is None:
            return

        if self._current_result and self._current_result.success:
            base = apply_orientation(self._original_verts, self._current_result, center=False)
        else:
            base = self._original_verts.copy()

        p = self._pitch_slider.value() / 10.0
        r = self._roll_slider.value() / 10.0
        y = self._yaw_slider.value() / 10.0

        if p == 0 and r == 0 and y == 0:
            self._show_oriented(base)
            return

        manual = compute_manual_rotation(p, r, y)
        self._show_oriented((manual.rotation_matrix @ base.T).T)
        self._status.setText(f"Fine-tune: P={p:.1f}° R={r:.1f}° Y={y:.1f}°")

    def _on_reset(self):
        if self._original_mesh is not None:
            self._viewer.display_mesh(self._original_mesh.copy(deep=True))
            self._viewer.reset_camera()
            self._current_result = None
            self._reset_sliders()
            self._status.setText("Reset to original.")

    def _on_apply(self):
        """Lock the current orientation as the new base."""
        viewer_mesh = self._viewer._mesh
        if viewer_mesh is None:
            self._status.setText("Nothing to apply.")
            return

        # The viewer's current points become the new original
        new_mesh = viewer_mesh.copy(deep=True)
        self._original_mesh = new_mesh
        self._original_verts = np.array(new_mesh.points).copy()
        self._current_result = None
        self._reset_sliders()
        self._status.setText("✓ Orientation applied and locked.")

    def _reset_sliders(self):
        for s in (self._pitch_slider, self._roll_slider, self._yaw_slider):
            s.blockSignals(True)
            s.setValue(0)
            s.blockSignals(False)

    # ══════════════════════════════════════════════════════
    # 3-Point orientation
    # ══════════════════════════════════════════════════════

    def _on_start_pick_3pt(self):
        if self._original_verts is None:
            self._status.setText("No mesh loaded.")
            return
        self._three_points.clear()
        self._picking_active = True
        self._prev_click_handler = self._viewer._click_handler
        self._viewer.set_click_handler(self._on_orient_mesh_clicked)
        for lbl in self._tp_labels:
            lbl.setText(lbl.text().split(":")[0] + ": —")
            lbl.setStyleSheet("font-size: 11px; color: #8B9DC3; font-family: monospace;")
        self._btn_start_pick.setText("Picking… (click mesh)")
        self._status.setText("Click on the mesh to place point 1 of 3.")

    def _on_clear_3pt(self):
        self._three_points.clear()
        self._picking_active = False
        self._viewer.set_click_handler(getattr(self, '_prev_click_handler', None))
        for lbl in self._tp_labels:
            lbl.setText(lbl.text().split(":")[0] + ": —")
            lbl.setStyleSheet("font-size: 11px; color: #8B9DC3; font-family: monospace;")
        self._btn_start_pick.setText("Start picking")
        self._status.setText("Points cleared.")
        self._refresh_3pt_display()

    def _on_orient_mesh_clicked(self, point, *args):
        if not self._picking_active or point is None:
            return
        if self._original_mesh is None:
            return
        pt = np.asarray(point)
        viewer_mesh = self._viewer._mesh
        if viewer_mesh is None:
            return
        idx = int(np.argmin(np.linalg.norm(viewer_mesh.points - pt, axis=1)))
        snapped = viewer_mesh.points[idx].copy()

        if len(self._three_points) < 3:
            self._three_points.append(snapped)
            i = len(self._three_points) - 1
            self._tp_labels[i].setText(
                f"  Point {i+1}: {snapped[0]:.2f}, {snapped[1]:.2f}, {snapped[2]:.2f}"
            )
            self._tp_labels[i].setStyleSheet(
                "font-size: 11px; color: #4ECDC4; font-family: monospace;"
            )
            self._refresh_3pt_display()

        if len(self._three_points) >= 3:
            self._picking_active = False
            self._btn_start_pick.setText("Start picking")
            self._status.setText(
                "3 points set. Click 'Apply 3-point orientation'."
            )
        else:
            remaining = 3 - len(self._three_points)
            self._status.setText(
                f"Click on the mesh to place point "
                f"{len(self._three_points)+1} of 3."
            )

    def _refresh_3pt_display(self):
        plotter = self._viewer._plotter
        for name in ("3pt_pts", "3pt_labels", "3pt_plane"):
            try:
                plotter.remove_actor(name)
            except Exception:
                pass
        if not self._three_points:
            plotter.render()
            return
        pts = np.array(self._three_points)
        plotter.add_mesh(
            pv.PolyData(pts), color="#FF6B35", point_size=22,
            render_points_as_spheres=True, name="3pt_pts",
        )
        labels = [f"P{i+1}" for i in range(len(pts))]
        plotter.add_point_labels(
            pts, labels, font_size=20, point_size=1,
            shape_opacity=0.8, shape_color="white",
            text_color="black", always_visible=True, name="3pt_labels",
        )
        if len(self._three_points) == 3:
            p1, p2, p3 = self._three_points
            tri_pts = np.array([p1, p2, p3])
            faces = np.array([[3, 0, 1, 2]])
            plane_mesh = pv.PolyData(tri_pts, faces)
            plotter.add_mesh(
                plane_mesh, color="#FF6B35", opacity=0.25,
                show_edges=True, edge_color="#FF6B35", name="3pt_plane",
            )
        plotter.render()

    def _on_apply_3pt(self):
        if len(self._three_points) < 3:
            self._status.setText("Pick 3 points first.")
            return
        if self._original_verts is None:
            self._status.setText("No mesh loaded.")
            return

        viewer_mesh = self._viewer._mesh
        if viewer_mesh is None:
            return
        verts = np.array(viewer_mesh.points)

        result = compute_three_point_orientation(
            verts,
            self._three_points[0],
            self._three_points[1],
            self._three_points[2],
        )
        if not result.success:
            self._status.setText(result.message)
            return

        oriented = apply_orientation(verts, result, center=False)
        oriented = oriented - np.mean(oriented, axis=0)

        mesh_out = self._original_mesh.copy(deep=True)
        mesh_out.points = oriented
        self._viewer.display_mesh(mesh_out)
        self._viewer.reset_camera()

        self._original_mesh = mesh_out.copy(deep=True)
        self._original_verts = np.array(mesh_out.points).copy()
        self._current_result = None

        self._three_points.clear()
        self._picking_active = False
        self._viewer.set_click_handler(getattr(self, '_prev_click_handler', None))
        for lbl in self._tp_labels:
            lbl.setText(lbl.text().split(":")[0] + ": —")
            lbl.setStyleSheet("font-size: 11px; color: #8B9DC3; font-family: monospace;")
        self._btn_start_pick.setText("Start picking")
        self._status.setText(f"✓ {result.message} — orientation applied.")

    # ══════════════════════════════════════════════════════
    # Export
    # ══════════════════════════════════════════════════════

    def _on_export(self):
        files = self._state.files
        idx = self._state.current_index
        if not (0 <= idx < len(files)):
            self._status.setText("No mesh to export.")
            return
        viewer_mesh = self._viewer._mesh
        if viewer_mesh is None:
            self._status.setText("No mesh to export.")
            return

        verts = np.array(viewer_mesh.points)
        faces = self._original_faces if self._original_faces is not None else np.zeros((0,3), dtype=int)

        try:
            out_dir_text = self._out_dir_edit.text().strip()
            out = export_oriented_mesh(
                original_path=Path(files[idx]),
                oriented_vertices=verts, faces=faces,
                output_dir=Path(out_dir_text) if out_dir_text else Path(files[idx]).parent,
            )
            self._status.setText(f"Exported: {out.name}")
            self._state.status_message.emit(f"Exported: {out}")
        except Exception as e:
            self._status.setText(f"Export error: {e}")

    # ══════════════════════════════════════════════════════
    # Batch
    # ══════════════════════════════════════════════════════

    def _on_pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory",
                                              self._out_dir_edit.text() or "")
        if d:
            self._out_dir_edit.setText(d)

    def _on_batch(self):
        if not self._require_reference():
            return
        files = self._state.files
        if not files:
            self._status.setText("No files loaded.")
            return

        out_dir_text = self._out_dir_edit.text().strip()
        output_dir = Path(out_dir_text) if out_dir_text else None

        self._flagged_indices.clear()
        self._progress.setVisible(True)
        self._progress.setMaximum(len(files))
        self._progress.setValue(0)
        self._btn_batch.setEnabled(False)
        self._batch_summary.setText("")

        self._batch_thread = QThread()
        self._worker = _BatchWorker(
            files=files,
            ref_verts=self._reference_verts,
            threshold=self._threshold_spin.value(),
            output_dir=output_dir,
        )
        self._worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_batch_progress)
        self._worker.item_done.connect(self._on_batch_item_done)
        self._worker.finished.connect(self._on_batch_finished)
        self._worker.finished.connect(self._batch_thread.quit)
        self._batch_thread.start()

    def _on_batch_progress(self, idx, total, name):
        self._progress.setValue(idx + 1)
        self._status.setText(f"Orienting {idx + 1}/{total}: {name}")

    def _on_batch_item_done(self, idx, status, error):
        """Mark flagged items in the file list panel."""
        if status == "flagged":
            self._flagged_indices.add(idx)
            self._mark_file_list_item(idx, flagged=True, error=error)
        elif status == "done":
            self._mark_file_list_item(idx, flagged=False, error=error)
        elif status == "error":
            self._mark_file_list_item(idx, error_status=True)

    def _on_batch_finished(self, done, flagged):
        self._progress.setVisible(False)
        self._btn_batch.setEnabled(True)

        summary = f"✓ {done} oriented"
        if flagged > 0:
            summary += f"  ·  ⚠ {flagged} flagged (review needed)"
        self._batch_summary.setText(summary)
        self._status.setText(f"Batch complete.")
        self._state.status_message.emit(
            f"MeshOrient batch: {done} done, {flagged} flagged"
        )

    def _mark_file_list_item(self, idx, flagged=False, error=0.0, error_status=False):
        """
        Highlight items in the file list panel.
        Accesses the FileListPanel's QListWidget through AppState's parent.
        """
        try:
            # Navigate up to MainWindow → _file_panel → list widget
            main_window = self._state.parent()
            file_panel = main_window._file_panel
            list_widget = file_panel._list   # QListWidget in FileListPanel

            if idx >= list_widget.count():
                return

            item = list_widget.item(idx)
            name = Path(self._state.files[idx]).name

            if error_status:
                item.setText(f"❌  {name}")
                item.setForeground(QColor("#E24B4A"))
            elif flagged:
                item.setText(f"⚠  {name}  ({error:.3f} mm)")
                item.setForeground(QColor("#F0A030"))
            else:
                item.setText(f"✓  {name}  ({error:.3f} mm)")
                item.setForeground(QColor("#4ECDC4"))
        except Exception:
            pass  # silently fail if UI structure differs