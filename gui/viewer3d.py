"""
DAMOS – 3D Viewer Widget
"""

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSizePolicy, QLabel
)
from PySide6.QtCore import Qt, Signal, QTimer


class Viewer3D(QWidget):
    landmark_picked = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mesh = None
        self._landmark_actors: dict = {}
        self._pick_mode: bool = False
        self._pending_label: str = ""
        self._show_axes = True
        self._show_edges = False
        self._click_handler = None

        self._build_ui()
        self._setup_plotter()

    # ── UI ────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QWidget()
        toolbar.setFixedHeight(34)
        toolbar.setStyleSheet(
            "background: #1A2030; border-bottom: 1px solid #323C52;"
        )
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(6, 2, 6, 2)
        tb_layout.setSpacing(4)

        def _btn(label: str, tooltip: str) -> QPushButton:
            b = QPushButton(label)
            b.setObjectName("NavButton")
            b.setFixedHeight(26)
            b.setFixedWidth(82)
            b.setToolTip(tooltip)
            return b

        self.btn_reset = _btn("Reset",    "Reset camera")
        self.btn_top   = _btn("Occlusal", "Top-down view (+Z)")
        self.btn_front = _btn("Front",    "Front view")
        self.btn_side  = _btn("Side",     "Side view")
        self.btn_axes  = _btn("Axes",     "Toggle axes")
        self.btn_edges = _btn("Edges",    "Toggle edges")

        for b in (self.btn_reset, self.btn_top, self.btn_front,
                  self.btn_side, self.btn_axes, self.btn_edges):
            tb_layout.addWidget(b)

        tb_layout.addStretch(1)

        self._pick_label = QLabel("")
        self._pick_label.setStyleSheet("color: #F0A030; font-size: 11px;")
        tb_layout.addWidget(self._pick_label)

        layout.addWidget(toolbar)

        self._pv_frame = QWidget()
        self._pv_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        pv_layout = QVBoxLayout(self._pv_frame)
        pv_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._pv_frame)

    def _setup_plotter(self):
        self._plotter = QtInteractor(self._pv_frame)
        self._plotter.set_background("#E8EDF4")
        self._pv_frame.layout().addWidget(self._plotter)

        self.btn_reset.clicked.connect(self.reset_camera)
        self.btn_top.clicked.connect(self.view_occlusal)
        self.btn_front.clicked.connect(self._plotter.view_xz)
        self.btn_side.clicked.connect(self._plotter.view_yz)
        self.btn_axes.clicked.connect(self._toggle_axes)
        self.btn_edges.clicked.connect(self._toggle_edges)

        # Single persistent picker — all panels register via set_click_handler()
        try:
            self._plotter.enable_point_picking(
                callback=self._dispatch_left_click,
                use_picker=True,
                show_message=False,
                show_point=False,
                left_clicking=True,
            )
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────

    def set_click_handler(self, callback):
        """Register the active left-click handler (replaces any previous one)."""
        self._click_handler = callback

    def display_mesh(self, mesh):
        if mesh is not None and not isinstance(mesh, pv.PolyData):
            try:
                mesh = mesh.extract_surface()
            except Exception:
                pass
        self._mesh = mesh
        self._refresh()

    def clear(self):
        self._mesh = None
        self._landmark_actors.clear()
        self._plotter.clear()
        self._plotter.render()

    def add_landmarks(self, landmarks: dict, label_visible: bool = True):
        self._clear_landmarks()
        for label, xyz in landmarks.items():
            if xyz is None:
                continue
            pt = np.array(xyz, dtype=float).reshape(1, 3)
            self._plotter.add_mesh(
                pv.PolyData(pt),
                color=self._label_color(label),
                point_size=14 if label.startswith("cusp") else 10,
                render_points_as_spheres=True,
                name=f"lmk_{label}",
            )
            if label_visible:
                self._plotter.add_point_labels(
                    pt, [label],
                    font_size=9,
                    text_color="white",
                    shape=None,
                    name=f"lbl_{label}",
                )
            self._landmark_actors[label] = f"lmk_{label}"
        self._plotter.render()

    def remove_landmark(self, label: str):
        if label in self._landmark_actors:
            self._plotter.remove_actor(self._landmark_actors.pop(label))
            try:
                self._plotter.remove_actor(f"lbl_{label}")
            except Exception:
                pass
        self._plotter.render()

    def enter_pick_mode(self, label: str):
        self._pick_mode = True
        self._pending_label = label
        self._pick_label.setText(f"  Picking: {label}")
        self._plotter.enable_point_picking(
            callback=self._on_point_picked,
            show_message=False,
            use_picker=True,
        )

    def exit_pick_mode(self):
        self._pick_mode = False
        self._pending_label = ""
        self._pick_label.setText("")
        self._plotter.disable_picking()

    def reset_camera(self):
        self._plotter.reset_camera()
        self._plotter.render()

    def view_occlusal(self):
        self._plotter.view_xy()
        self._plotter.reset_camera()
        self._plotter.render()

    def screenshot(self, path: str):
        self._plotter.screenshot(path)

    # ── Private ───────────────────────────────────────────

    def _refresh(self):
        self._plotter.clear()

        if self._mesh is not None:
            try:
                mesh = self._mesh.copy(deep=True)
                if not isinstance(mesh, pv.PolyData):
                    mesh = mesh.extract_surface()
                mesh.point_data["elevation"] = mesh.points[:, 2]

                self._plotter.add_mesh(
                    mesh,
                    scalars="elevation",
                    cmap="gist_earth",
                    smooth_shading=True,
                    show_edges=self._show_edges,
                    opacity=1.0,
                    name="main_mesh",
                    show_scalar_bar=True,
                    scalar_bar_args={
                        "title":           "Z (mm)",
                        "title_font_size": 11,
                        "label_font_size": 10,
                        "n_labels":        4,
                        "fmt":             "%.1f",
                        "width":           0.07,
                        "height":          0.30,
                        "position_x":      0.92,
                        "position_y":      0.06,
                        "vertical":        True,
                        "color":           "white",
                        "shadow":          True,
                    },
                )
                self._plotter.enable_eye_dome_lighting()
            except Exception:
                pass

        self._plotter.render()
        QTimer.singleShot(50, self._post_render)

    def _clear_landmarks(self):
        for actor_name in self._landmark_actors.values():
            try:
                self._plotter.remove_actor(actor_name)
            except Exception:
                pass
        for label in list(self._landmark_actors.keys()):
            try:
                self._plotter.remove_actor(f"lbl_{label}")
            except Exception:
                pass
        self._landmark_actors.clear()

    def _dispatch_left_click(self, point, *args):
        """Single entry point for all left-click picking."""
        if point is not None and self._click_handler is not None:
            self._click_handler(point)

    def _on_point_picked(self, point):
        if not self._pick_mode or point is None:
            return
        xyz = np.array(point, dtype=float)
        label = self._pending_label
        self.exit_pick_mode()
        self.landmark_picked.emit(label, xyz)

    def _toggle_axes(self):
        self._show_axes = not self._show_axes
        self._refresh()   # rebuild scene so axes state is respected after clear()

    def _toggle_edges(self):
        self._show_edges = not self._show_edges
        self._refresh()

    @staticmethod
    def _label_color(label: str) -> str:
        if label.startswith("cusp"):   return "#FF6B35"
        if label.startswith("inter"):  return "#4ECDC4"
        if label == "center":          return "#FFE66D"
        if label.startswith("semi"):   return "#A8DADC"
        return "#C0C0C0"

    def _post_render(self):
        """Second render pass so axes appear reliably."""
        try:
            if self._show_axes:
                self._plotter.show_axes()
            self._plotter.render()
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            self._plotter.close()
        except Exception:
            pass
        super().closeEvent(event)
