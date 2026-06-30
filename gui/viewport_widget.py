"""
DAMOS 3D Viewport Widget
=========================
PyVistaQt-based 3D viewport with:
- Mesh display with occlusal plane overlay
- Normal vector arrow
- Landmark picking (3-click mode)
- Preset camera views (Top, Front, Side)
- Visual feedback for orientation quality

Author: Albert Epitíe Dyowe Roig
"""

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Signal, Qt
from typing import Optional, List


class ViewportWidget(QWidget):
    """
    3D viewport for mesh visualization and interactive orientation.
    
    Signals
    -------
    landmark_picked : Signal(np.ndarray)
        Emitted when user picks a point on the mesh surface.
    orientation_applied : Signal()
        Emitted when orientation is visually updated.
    """

    landmark_picked = Signal(object)  # np.ndarray
    orientation_applied = Signal()

    # Color scheme
    MESH_COLOR = "#AFA9EC"
    MESH_EDGE_COLOR = "#534AB7"
    PLANE_COLOR = "#E24B4A"
    PLANE_OPACITY = 0.25
    NORMAL_COLOR = "#1D9E75"
    LANDMARK_COLOR = "#D85A30"
    AXES_COLORS = {"x": "#E24B4A", "y": "#378ADD", "z": "#1D9E75"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mesh: Optional[pv.PolyData] = None
        self._mesh_actor = None
        self._plane_actor = None
        self._normal_actor = None
        self._landmark_actors: List = []
        self._picking_enabled = False
        self._picked_landmarks: List[np.ndarray] = []

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # PyVista interactor
        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")
        self.plotter.enable_anti_aliasing("ssaa")
        layout.addWidget(self.plotter.interactor, stretch=1)

        # View control buttons
        btn_bar = QHBoxLayout()
        btn_bar.setContentsMargins(8, 4, 8, 4)
        btn_bar.setSpacing(4)

        self.btn_top = QPushButton("Top")
        self.btn_front = QPushButton("Front")
        self.btn_side = QPushButton("Side")
        self.btn_reset = QPushButton("Reset")

        for btn in (self.btn_top, self.btn_front, self.btn_side, self.btn_reset):
            btn.setFixedHeight(28)
            btn.setStyleSheet("""
                QPushButton {
                    background: #f5f5f5; border: 1px solid #ddd;
                    border-radius: 4px; padding: 0 10px;
                    font-size: 11px; color: #555;
                }
                QPushButton:hover { background: #eee; }
            """)
            btn_bar.addWidget(btn)

        btn_bar.addStretch()

        # Info label
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("font-size: 11px; color: #888; padding: 0 4px;")
        btn_bar.addWidget(self.info_label)

        layout.addLayout(btn_bar)

        # Connect buttons
        self.btn_top.clicked.connect(lambda: self._set_view("top"))
        self.btn_front.clicked.connect(lambda: self._set_view("front"))
        self.btn_side.clicked.connect(lambda: self._set_view("side"))
        self.btn_reset.clicked.connect(lambda: self._set_view("iso"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_mesh(self, mesh: pv.PolyData):
        """Load and display a mesh."""
        self._mesh = mesh
        self._clear_overlays()

        if self._mesh_actor:
            self.plotter.remove_actor(self._mesh_actor)

        self._mesh_actor = self.plotter.add_mesh(
            mesh,
            color=self.MESH_COLOR,
            edge_color=self.MESH_EDGE_COLOR,
            show_edges=False,
            smooth_shading=True,
            specular=0.3,
            specular_power=15,
            name="tooth_mesh",
        )

        self.plotter.reset_camera()
        self._add_axes_widget()
        self.info_label.setText(
            f"{mesh.n_points:,} vertices · {mesh.n_cells:,} faces"
        )

    def load_mesh_from_file(self, filepath: str):
        """Load mesh from file path."""
        mesh = pv.read(filepath)
        self.load_mesh(mesh)

    def show_orientation_plane(
        self,
        origin: np.ndarray,
        normal: np.ndarray,
        size: Optional[float] = None,
    ):
        """
        Display the occlusal plane as a semi-transparent disc.
        
        Parameters
        ----------
        origin : np.ndarray
            Center of the plane.
        normal : np.ndarray
            Normal vector of the plane.
        size : float, optional
            Radius of the plane disc. Auto-computed from mesh bounds if None.
        """
        if self._plane_actor:
            self.plotter.remove_actor(self._plane_actor)

        if size is None and self._mesh is not None:
            bounds = self._mesh.bounds
            size = max(
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                bounds[5] - bounds[4],
            ) * 0.6

        plane = pv.Disc(center=origin, normal=normal, inner=0, outer=size, r_res=1, c_res=60)
        self._plane_actor = self.plotter.add_mesh(
            plane,
            color=self.PLANE_COLOR,
            opacity=self.PLANE_OPACITY,
            name="occlusal_plane",
        )

    def show_normal_arrow(self, origin: np.ndarray, direction: np.ndarray, length: Optional[float] = None):
        """Display the +Z normal arrow."""
        if self._normal_actor:
            self.plotter.remove_actor(self._normal_actor)

        if length is None and self._mesh is not None:
            bounds = self._mesh.bounds
            length = max(
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                bounds[5] - bounds[4],
            ) * 0.4

        arrow = pv.Arrow(
            start=origin,
            direction=direction,
            scale=length,
            tip_length=0.2,
            tip_radius=0.06,
            shaft_radius=0.02,
        )
        self._normal_actor = self.plotter.add_mesh(
            arrow,
            color=self.NORMAL_COLOR,
            name="normal_arrow",
        )

    def update_oriented_mesh(self, vertices: np.ndarray):
        """
        Update mesh vertex positions after orientation.
        
        Parameters
        ----------
        vertices : np.ndarray
            (N, 3) new vertex coordinates.
        """
        if self._mesh is not None:
            self._mesh.points = vertices
            self.plotter.render()
            self.orientation_applied.emit()

    def enable_landmark_picking(self, enabled: bool = True):
        """
        Enable/disable landmark picking mode.
        
        In picking mode, clicking on the mesh surface adds a landmark sphere.
        After 3 landmarks, emits the landmark_picked signal.
        """
        self._picking_enabled = enabled
        self._picked_landmarks.clear()
        self._clear_landmark_actors()

        if enabled:
            self.plotter.enable_surface_point_picking(
                callback=self._on_point_picked,
                show_message=False,
                show_point=False,
                use_picker=True,
            )
            self.info_label.setText("Click 3 points on the occlusal surface")
        else:
            try:
                self.plotter.disable_picking()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Camera presets
    # ------------------------------------------------------------------

    def _set_view(self, view: str):
        """Set camera to predefined view."""
        if view == "top":
            self.plotter.view_xy()
        elif view == "front":
            self.plotter.view_xz()
        elif view == "side":
            self.plotter.view_yz()
        else:
            self.plotter.reset_camera()
            self.plotter.camera.azimuth = 30
            self.plotter.camera.elevation = 20

        self.plotter.render()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_axes_widget(self):
        """Add orientation axes in corner."""
        self.plotter.add_axes(
            line_width=2,
            color="gray",
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
        )

    def _on_point_picked(self, point):
        """Handle point picking for 3-landmark mode."""
        if not self._picking_enabled:
            return

        coord = np.array(point)
        self._picked_landmarks.append(coord)

        # Add visual sphere at picked point
        sphere = pv.Sphere(radius=0.3, center=coord)
        actor = self.plotter.add_mesh(
            sphere,
            color=self.LANDMARK_COLOR,
            name=f"landmark_{len(self._picked_landmarks)}",
        )
        self._landmark_actors.append(actor)

        remaining = 3 - len(self._picked_landmarks)
        if remaining > 0:
            self.info_label.setText(f"{remaining} more point(s) needed")
        else:
            self.info_label.setText("3 landmarks defined")
            landmarks = np.array(self._picked_landmarks[:3])
            self.landmark_picked.emit(landmarks)
            self._picking_enabled = False

    def _clear_overlays(self):
        """Remove all overlay actors."""
        if self._plane_actor:
            self.plotter.remove_actor(self._plane_actor)
            self._plane_actor = None
        if self._normal_actor:
            self.plotter.remove_actor(self._normal_actor)
            self._normal_actor = None
        self._clear_landmark_actors()

    def _clear_landmark_actors(self):
        """Remove landmark spheres."""
        for actor in self._landmark_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self._landmark_actors.clear()

    def close(self):
        """Clean shutdown."""
        self.plotter.close()
        super().close()
