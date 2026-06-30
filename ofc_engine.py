import os
import sys
from dataclasses import dataclass
from collections import Counter
from typing import Optional, List, Dict

import numpy as np
import pyvista as pv
# pyvistaqt loaded later if available
# PyQt5 loaded later if available


# =========================================================
# DATA CLASS
# =========================================================
@dataclass
class TopographyResult:
    mean_slope: float
    slope_sd: float
    simple_relief: float
    area_3d: float
    area_2d: float
    relief: float
    rfi_1: float
    rfi_2: float
    min_z: float
    max_z: float
    n_faces: int
    mesh_for_display: pv.PolyData
    dne: float = 0.0
    opcr: float = 0.0
    md: float = 0.0
    bl: float = 0.0


# =========================================================
# ENGINE
# =========================================================
class TopographyEngine:
    def __init__(self):
        self.mesh: Optional[pv.PolyData] = None
        self.points: Optional[np.ndarray] = None

    def load_mesh(self, filepath: str):
        mesh = pv.read(filepath)
        mesh = mesh.extract_surface().triangulate()

        if mesh.n_points == 0 or mesh.n_cells == 0:
            raise ValueError("Mesh is empty or invalid.")

        self.mesh = mesh
        self.points = mesh.points
        return self.mesh

    def set_lowest_point_to_zero(self, mesh: pv.PolyData) -> pv.PolyData:
        m = mesh.copy(deep=True)
        zmin = float(np.min(m.points[:, 2]))
        m.points[:, 2] = m.points[:, 2] - zmin
        return m

    def compute_triangle_slopes(self, mesh: pv.PolyData) -> np.ndarray:
        """
        Flat horizontal face -> 0°
        Vertical face -> 90°
        """
        surf = mesh.extract_surface().triangulate()
        normals_mesh = surf.compute_normals(
            cell_normals=True,
            point_normals=False,
            auto_orient_normals=False,
            consistent_normals=False,
            split_vertices=False,
            inplace=False,
        )

        normals = normals_mesh.cell_data["Normals"]
        if normals is None or len(normals) == 0:
            raise ValueError("Could not compute face normals.")

        nz = np.abs(normals[:, 2])
        nz = np.clip(nz, 0.0, 1.0)

        slopes = np.degrees(np.arccos(nz))
        return slopes

    def compute_projected_area_2d(self, mesh: pv.PolyData) -> float:
        surf = mesh.extract_surface().triangulate()
        faces = surf.faces.reshape(-1, 4)[:, 1:4]
        pts = surf.points

        v0 = pts[faces[:, 0]][:, :2]
        v1 = pts[faces[:, 1]][:, :2]
        v2 = pts[faces[:, 2]][:, :2]

        area2d = 0.5 * np.abs(
            (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) -
            (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
        )
        return float(np.sum(area2d))

    def compute_area_3d(self, mesh: pv.PolyData) -> float:
        surf = mesh.extract_surface().triangulate()
        return float(surf.area)

    def decimate_to_target_faces(self, mesh: pv.PolyData, target_faces: int) -> pv.PolyData:
        surf = mesh.extract_surface().triangulate()

        current_faces = surf.n_cells
        if target_faces <= 0 or current_faces <= target_faces:
            return surf

        reduction = 1.0 - (target_faces / current_faces)
        reduction = min(max(reduction, 0.0), 0.99)

        try:
            dec = surf.decimate(reduction)
            dec = dec.extract_surface().triangulate()
            if dec.n_cells > 0:
                return dec
            return surf
        except Exception:
            return surf

    def compute_dne(self, mesh: pv.PolyData,
                    docondition: bool = True,
                    dooutlier: bool = True,
                    outlierperc: float = 95.0) -> tuple:
        """
        Dirichlet Normal Energy — fórmula exacta Winchester/MorphoTester.
        (Bunn et al. 2011; Winchester in review)

        Per cada cara i amb vèrtexs [v0,v1,v2]:
          b1 = pos[v1]-pos[v0],  b2 = pos[v2]-pos[v0]   (espai de posicions)
          c1 = vn[v1]-vn[v0],   c2 = vn[v2]-vn[v0]      (espai de normals)
          G     = [[b1·b1, b1·b2], [b2·b1, b2·b2]]       (tensor mètric)
          fstar = [[c1·c1, c1·c2], [c2·c1, c2·c2]]       (pullback normal)
          e[i]  = trace(G⁻¹ × fstar)                      (densitat energia)
          area  = 0.5 · sqrt(det G)
          equantity[i] = e[i] · area[i]

        Exclou: cares frontera, cares amb cond(G)>1e5, NaNs.
        Opcionalment elimina outliers (95è percentil per defecte).

        Les normals de vèrtex es calculen com en normcore.py de MorphoTester:
          vnormal[v] = normalize(Σ fnormal_unitari dels triangles adjacents)

        Retorna (dne_total: float, dne_per_face: np.ndarray)
        """
        surf  = mesh.extract_surface().triangulate()
        pts   = np.array(surf.points, dtype=np.float64)
        faces = surf.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
        n_faces = surf.n_cells
        n_verts = surf.n_points

        # ── 1. Normals unitàries per cara ────────────────────────────────
        surf_fn = surf.compute_normals(
            cell_normals=True, point_normals=False,
            auto_orient_normals=True, consistent_normals=True,
            inplace=False,
        )
        fn = np.array(surf_fn.cell_data["Normals"], dtype=np.float64)

        # ── 2. Normals de vèrtex: normalize(Σ fn_unit dels triangles veïns) ─
        vnormal = np.zeros((n_verts, 3), dtype=np.float64)
        np.add.at(vnormal, faces[:, 0], fn)
        np.add.at(vnormal, faces[:, 1], fn)
        np.add.at(vnormal, faces[:, 2], fn)
        vn_norms = np.linalg.norm(vnormal, axis=1, keepdims=True)
        vn_norms = np.where(vn_norms < 1e-12, 1.0, vn_norms)
        vnormal /= vn_norms

        # ── 3. Identificar cares frontera (arestes compartides per 1 sola cara) ─
        edge_to_faces: dict = {}
        for fi, tri in enumerate(faces):
            for k in range(3):
                e = (min(tri[k], tri[(k + 1) % 3]),
                     max(tri[k], tri[(k + 1) % 3]))
                edge_to_faces.setdefault(e, []).append(fi)
        boundary_set = set()
        for flist in edge_to_faces.values():
            if len(flist) == 1:
                boundary_set.add(flist[0])

        # ── 4. Vectoritzar càlcul d'energia ─────────────────────────────
        v0i, v1i, v2i = faces[:, 0], faces[:, 1], faces[:, 2]
        b1 = pts[v1i] - pts[v0i]   # (N,3) edge vectors in position space
        b2 = pts[v2i] - pts[v0i]
        c1 = vnormal[v1i] - vnormal[v0i]  # (N,3) edge vectors in normal space
        c2 = vnormal[v2i] - vnormal[v0i]

        # Components del tensor mètric G (simètric 2×2)
        g00 = np.einsum('ij,ij->i', b1, b1)
        g01 = np.einsum('ij,ij->i', b1, b2)
        g11 = np.einsum('ij,ij->i', b2, b2)
        det = g00 * g11 - g01 * g01

        # Àrea per cara
        face_areas = 0.5 * np.sqrt(np.maximum(det, 0.0))

        # Components de fstarh (simètric 2×2)
        f00 = np.einsum('ij,ij->i', c1, c1)
        f01 = np.einsum('ij,ij->i', c1, c2)
        f11 = np.einsum('ij,ij->i', c2, c2)

        # trace(G⁻¹ × fstarh) = (g11·f00 - 2·g01·f01 + g00·f11) / det
        e = np.full(n_faces, np.nan)
        valid = det > 1e-20
        e[valid] = (g11[valid] * f00[valid]
                    - 2.0 * g01[valid] * f01[valid]
                    + g00[valid] * f11[valid]) / det[valid]

        # ── 5. Comprovació de nombre de condicionament ────────────────────
        if docondition:
            tr  = g00 + g11
            disc = np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0))
            lam_max = (tr + disc) * 0.5
            lam_min = (tr - disc) * 0.5
            safe = lam_min > 1e-30
            cond_num = np.where(safe, lam_max / lam_min, np.inf)
            e[cond_num > 100000.0] = 0.0

        # ── 6. Posar a zero les cares frontera i NaN ─────────────────────
        if boundary_set:
            e[np.fromiter(boundary_set, dtype=np.int64)] = 0.0
        nan_mask = np.isnan(e)
        e[nan_mask] = 0.0

        # ── 7. equantity = e × area ───────────────────────────────────────
        equantity = e * face_areas

        # ── 8. Eliminació d'outliers (95è percentil per defecte) ─────────
        if dooutlier:
            source = e   # outliertype=0 → outliers per densitat d'energia
            pos_mask = source > 0.0
            if pos_mask.any():
                pct = np.percentile(source[pos_mask], outlierperc)
                equantity[source > pct] = 0.0

        dne_per_face = equantity
        return float(np.sum(dne_per_face)), dne_per_face

    def compute_opcr(self, mesh: pv.PolyData, min_patch_faces: int = 3,
                     n_rotations: int = 8) -> tuple:
        """
        Orientation Patch Count Rotated (Boyer et al. 2010 / MorphoTester).

        Replica exactament l'algorisme de MeshOPCR de Winchester:
          - n_rotations (8) rotacions de theta=5.625° al voltant de l'eix Z
          - A cada rotació: agrupa cares adjacents del mateix bin azimutal
            en parxes connexos (mín. min_patch_faces cares)
          - OPCR = mitjana dels 8 comptadors OPC

        Optimitzat: en lloc de girar la malla, desplacem el bin offset per θ.
        Les cares planes (nx=ny=0) s'exclouen (bin=-1, no compten).

        Retorna (opcr_float: float, face_bins_rotation0: np.ndarray int8 0–7)
        """
        surf  = mesh.extract_surface().triangulate()
        surf_fn = surf.compute_normals(
            cell_normals=True, point_normals=False,
            auto_orient_normals=True, consistent_normals=True,
            inplace=False,
        )
        normals   = np.array(surf_fn.cell_data["Normals"], dtype=np.float64)
        n_faces   = surf.n_cells
        raw_faces = surf.faces.reshape(-1, 4)[:, 1:].astype(np.int64)

        # Cares planes (normal XY ≈ zero) → excloses (com MorphoTester)
        xy_mag   = np.sqrt(normals[:, 0] ** 2 + normals[:, 1] ** 2)
        is_flat  = xy_mag < 1e-10

        # Azimut base: atan2(ny, nx) en graus [0, 360)
        base_az  = np.degrees(np.arctan2(normals[:, 1], normals[:, 0]))
        base_az  = np.where(base_az < 0.0, base_az + 360.0, base_az)

        # Precomputar adjacències (llista d'arestes amb 2 cares veïnes)
        adj_pairs: list = []
        edge_to_faces: dict = {}
        for fi, tri in enumerate(raw_faces):
            for k in range(3):
                e = (min(tri[k], tri[(k + 1) % 3]),
                     max(tri[k], tri[(k + 1) % 3]))
                if e in edge_to_faces:
                    adj_pairs.append((edge_to_faces[e], fi))
                else:
                    edge_to_faces[e] = fi

        theta_deg = 360.0 / (n_rotations * 8)   # 5.625°
        opc_list  = []

        for rot in range(n_rotations):
            offset  = 22.5 + rot * theta_deg          # +22.5° molaR, +rot×5.625°
            az_rot  = (base_az - rot * theta_deg + 22.5) % 360.0
            face_bin = (az_rot / 45.0).astype(int) % 8
            face_bin[is_flat] = -1   # planes no pertanyen a cap bin

            # Union-Find per als parxes d'aquesta rotació
            parent = list(range(n_faces))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x: int, y: int) -> None:
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py

            for fi, fj in adj_pairs:
                if face_bin[fi] == face_bin[fj] and face_bin[fi] >= 0:
                    union(fi, fj)

            # Comptar parxes de mida ≥ min_patch_faces
            patch_sizes = Counter(find(i) for i in range(n_faces)
                                  if face_bin[i] >= 0)
            opc = sum(1 for sz in patch_sizes.values() if sz >= min_patch_faces)
            opc_list.append(opc)

        opcr_float = float(np.mean(opc_list))

        # Bins per la rotació 0 (per visualització)
        az0      = (base_az + 22.5) % 360.0
        face_bin0 = (az0 / 45.0).astype(int) % 8
        face_bin0[is_flat] = 0
        return opcr_float, face_bin0.astype(np.int8)

    def compute_bounding_box_dims(self, mesh: pv.PolyData) -> tuple:
        """
        Retorna (mesiodistal_mm, bucolingual_mm) com les extensions Y i X
        del bounding box de la malla.
        Suposa que la malla està orientada amb el pla oclusal ≈ XY i Z apical.
        """
        pts = mesh.points
        md  = float(pts[:, 1].max() - pts[:, 1].min())   # extensió Y
        bl  = float(pts[:, 0].max() - pts[:, 0].min())   # extensió X
        return md, bl

    def compute_metrics(
        self,
        smooth_iterations: int = 0,
        target_faces: int = 0,
        set_origin_to_lowest: bool = True,
    ) -> TopographyResult:
        if self.mesh is None:
            raise ValueError("No mesh loaded.")

        mesh = self.mesh.copy(deep=True)

        if smooth_iterations > 0:
            mesh = mesh.smooth(n_iter=smooth_iterations, relaxation_factor=0.01)

        if target_faces > 0:
            mesh = self.decimate_to_target_faces(mesh, target_faces)

        if set_origin_to_lowest:
            mesh = self.set_lowest_point_to_zero(mesh)

        slopes = self.compute_triangle_slopes(mesh)
        mean_slope = float(np.mean(slopes))
        slope_sd = float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0.0

        z = mesh.points[:, 2]
        min_z = float(np.min(z))
        max_z = float(np.max(z))
        simple_relief = max_z - min_z

        area_3d = self.compute_area_3d(mesh)
        area_2d = self.compute_projected_area_2d(mesh)

        if area_2d <= 1e-12:
            relief = 0.0
            rfi_1 = 0.0
            rfi_2 = 0.0
        else:
            ratio = area_3d / area_2d
            relief = 100.0 * ratio
            rfi_1 = ratio
            rfi_2 = float(np.log(np.sqrt(ratio)))

        dne,  dne_per_face = self.compute_dne(mesh)
        opcr, face_bins    = self.compute_opcr(mesh)
        md, bl             = self.compute_bounding_box_dims(mesh)

        # Adjuntar dades per cara a la malla de visualització
        mesh.cell_data["DNE_face"]  = dne_per_face
        mesh.cell_data["OrientBin"] = face_bins.astype(float)

        return TopographyResult(
            mean_slope=mean_slope,
            slope_sd=slope_sd,
            simple_relief=simple_relief,
            area_3d=area_3d,
            area_2d=area_2d,
            relief=relief,
            rfi_1=rfi_1,
            rfi_2=rfi_2,
            min_z=min_z,
            max_z=max_z,
            n_faces=mesh.n_cells,
            mesh_for_display=mesh,
            dne=dne,
            opcr=opcr,
            md=md,
            bl=bl,
        )

    def export_single_csv(self, filepath: str, result: TopographyResult):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("metric,value\n")
            f.write(f"mean_slope,{result.mean_slope}\n")
            f.write(f"slope_sd,{result.slope_sd}\n")
            f.write(f"simple_relief,{result.simple_relief}\n")
            f.write(f"area_3d,{result.area_3d}\n")
            f.write(f"area_2d,{result.area_2d}\n")
            f.write(f"relief,{result.relief}\n")
            f.write(f"rfi_1,{result.rfi_1}\n")
            f.write(f"rfi_2,{result.rfi_2}\n")
            f.write(f"min_z,{result.min_z}\n")
            f.write(f"max_z,{result.max_z}\n")
            f.write(f"n_faces,{result.n_faces}\n")
            f.write(f"dne,{result.dne}\n")
            f.write(f"opcr,{result.opcr}\n")
            f.write(f"md,{result.md}\n")
            f.write(f"bl,{result.bl}\n")

    def export_batch_csv(self, filepath: str, batch_results: Dict[str, TopographyResult], ordered_files: List[str]):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("file,mean_slope,slope_sd,simple_relief,area_3d,area_2d,relief,"
                    "rfi_1,rfi_2,min_z,max_z,n_faces,dne,opcr,md,bl\n")
            for path in ordered_files:
                if path not in batch_results:
                    continue
                r    = batch_results[path]
                name = os.path.basename(path)
                f.write(
                    f"{name},{r.mean_slope},{r.slope_sd},{r.simple_relief},"
                    f"{r.area_3d},{r.area_2d},{r.relief},{r.rfi_1},{r.rfi_2},"
                    f"{r.min_z},{r.max_z},{r.n_faces},"
                    f"{r.dne},{r.opcr},{r.md},{r.bl}\n"
                )



# GUI classes (only needed when running standalone)
try:
    from PyQt5 import QtWidgets, QtCore
    from pyvistaqt import QtInteractor

    # =========================================================
    # UI HELPERS
    # =========================================================
    class CollapsibleBox(QtWidgets.QWidget):
        def __init__(self, title: str, expanded: bool = False):
            super().__init__()
    
            self.toggle_button = QtWidgets.QToolButton()
            self.toggle_button.setText(title)
            self.toggle_button.setCheckable(True)
            self.toggle_button.setChecked(expanded)
            self.toggle_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            self.toggle_button.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
            self.toggle_button.clicked.connect(self._on_toggled)
    
            self.content = QtWidgets.QFrame()
            self.content.setObjectName("SectionContent")
            self.content_layout = QtWidgets.QVBoxLayout(self.content)
            self.content_layout.setContentsMargins(10, 10, 10, 10)
            self.content_layout.setSpacing(8)
            self.content.setVisible(expanded)
    
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            layout.addWidget(self.toggle_button)
            layout.addWidget(self.content)
    
        def _on_toggled(self):
            checked = self.toggle_button.isChecked()
            self.toggle_button.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
            self.content.setVisible(checked)
    
        def addWidget(self, widget):
            self.content_layout.addWidget(widget)
    
        def addLayout(self, layout):
            self.content_layout.addLayout(layout)
    
    
    # =========================================================
    # GUI
    # =========================================================
    class TopographyWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Dental Topography")
            self.resize(1700, 1000)
    
            self.engine = TopographyEngine()
            self.current_mesh = None
            self.current_result: Optional[TopographyResult] = None
    
            self.batch_files: List[str] = []
            self.batch_results: Dict[str, TopographyResult] = {}
            self.current_index: int = -1
            self.current_folder: str = ""
    
            self._build_ui()
            self._apply_style()
    
        def _apply_style(self):
            self.setStyleSheet("""
                QMainWindow { background: #e8edf4; }
    
                QFrame#SidePanel {
                    background: #243447;
                    border: 1px solid #1b2733;
                    border-radius: 18px;
                }
    
                QFrame#ViewerFrame {
                    background: #ffffff;
                    border: 1px solid #d5dee8;
                    border-radius: 18px;
                }
    
                QScrollArea { border: none; background: transparent; }
                QWidget#ScrollContent { background: transparent; }
    
                QToolButton {
                    background: #2f4359;
                    color: #ffffff;
                    border: 1px solid #3d536b;
                    border-radius: 12px;
                    padding: 10px 12px;
                    text-align: left;
                    font-size: 14px;
                    font-weight: 700;
                }
    
                QToolButton:hover { background: #39506a; }
    
                QFrame#SectionContent {
                    background: #2f4359;
                    border: 1px solid #3d536b;
                    border-radius: 14px;
                }
    
                QLabel#MainTitle {
                    font-size: 22px;
                    font-weight: 800;
                    color: #ffffff;
                    padding-bottom: 4px;
                }
    
                QLabel#FileLabel {
                    color: #d7e3f1;
                    font-size: 12px;
                    padding-bottom: 4px;
                }
    
                QLabel#ValueLabel {
                    color: #ffffff;
                    font-size: 18px;
                    font-weight: 800;
                    padding: 4px 0px;
                }
    
                QLabel#HelpLabel {
                    color: #d7e3f1;
                    font-size: 11px;
                    padding-top: 2px;
                    padding-bottom: 4px;
                }
    
                QLabel { color: #e5edf7; font-size: 12px; }
    
                QPushButton {
                    background: #6fa8dc;
                    color: #102133;
                    border: none;
                    border-radius: 12px;
                    padding: 10px 12px;
                    font-size: 13px;
                    font-weight: 700;
                    min-height: 22px;
                }
    
                QPushButton:hover { background: #87b9e6; }
                QPushButton:pressed { background: #5e99cf; }
    
                QPushButton:disabled {
                    background: #55697d;
                    color: #b9c7d6;
                }
    
                QSpinBox, QDoubleSpinBox, QComboBox, QProgressBar {
                    background: #f8fbff;
                    color: #1f2937;
                    border: 1px solid #bfd0e3;
                    border-radius: 10px;
                    padding: 6px 10px;
                    min-height: 30px;
                    font-size: 13px;
                }
    
                QProgressBar { text-align: center; }
                QProgressBar::chunk {
                    background-color: #6fa8dc;
                    border-radius: 8px;
                }
    
                QPlainTextEdit {
                    background: #eef4fb;
                    color: #1e293b;
                    border: 1px solid #c6d4e3;
                    border-radius: 10px;
                    padding: 8px;
                    font-size: 12px;
                }
    
                QCheckBox {
                    color: #edf4fb;
                    font-size: 12px;
                    spacing: 8px;
                    min-height: 22px;
                }
    
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border-radius: 4px;
                    border: 2px solid #6fa8dc;
                    background: #ffffff;
                }
    
                QCheckBox::indicator:checked {
                    background: #6fa8dc;
                    border: 2px solid #ffffff;
                }
    
                QCheckBox::indicator:unchecked {
                    background: #ffffff;
                    border: 2px solid #6fa8dc;
                }
            """)
    
        def _build_ui(self):
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
    
            root_layout = QtWidgets.QHBoxLayout(central)
            root_layout.setContentsMargins(10, 10, 10, 10)
            root_layout.setSpacing(10)
    
            side_panel = QtWidgets.QFrame()
            side_panel.setObjectName("SidePanel")
            side_panel.setMinimumWidth(430)
            side_panel.setMaximumWidth(500)
    
            side_panel_layout = QtWidgets.QVBoxLayout(side_panel)
            side_panel_layout.setContentsMargins(10, 10, 10, 10)
    
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
    
            scroll_content = QtWidgets.QWidget()
            scroll_content.setObjectName("ScrollContent")
            side_layout = QtWidgets.QVBoxLayout(scroll_content)
            side_layout.setContentsMargins(6, 6, 6, 6)
            side_layout.setSpacing(10)
    
            self.title_label = QtWidgets.QLabel("Dental Topography")
            self.title_label.setObjectName("MainTitle")
    
            self.file_label = QtWidgets.QLabel("No file loaded")
            self.file_label.setObjectName("FileLabel")
            self.file_label.setWordWrap(True)
    
            self.mean_slope_label = QtWidgets.QLabel("Mean slope: -")
            self.mean_slope_label.setObjectName("ValueLabel")
    
            self.relief_label = QtWidgets.QLabel("Relief: -")
            self.relief_label.setObjectName("ValueLabel")
    
            self.rfi1_label = QtWidgets.QLabel("RFI 1: -")
            self.rfi1_label.setObjectName("ValueLabel")
    
            self.rfi2_label = QtWidgets.QLabel("RFI 2: -")
            self.rfi2_label.setObjectName("ValueLabel")
    
            self.extra_label = QtWidgets.QLabel("Slope SD / simple relief: -")
            self.extra_label.setObjectName("ValueLabel")
            self.extra_label.setWordWrap(True)

            self.dne_label = QtWidgets.QLabel("DNE: -")
            self.dne_label.setObjectName("ValueLabel")

            self.opcr_label = QtWidgets.QLabel("OPCR: -")
            self.opcr_label.setObjectName("ValueLabel")

            self.md_label = QtWidgets.QLabel("Mesiodistal: -")
            self.md_label.setObjectName("ValueLabel")

            self.bl_label = QtWidgets.QLabel("Bucolingual: -")
            self.bl_label.setObjectName("ValueLabel")

            self.help_label = QtWidgets.QLabel(
                "Automated dental topography focused on slope, relief and two RFI outputs. "
                "The mesh can be leveled so the lowest point is Z = 0."
            )
            self.help_label.setObjectName("HelpLabel")
            self.help_label.setWordWrap(True)
    
            self.btn_load = QtWidgets.QPushButton("Load tooth mesh")
            self.btn_load_folder = QtWidgets.QPushButton("Select batch folder")
            self.btn_run = QtWidgets.QPushButton("Compute current metrics")
            self.btn_run_batch = QtWidgets.QPushButton("Compute batch metrics")
    
            self.progress_bar = QtWidgets.QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p%")
    
            top_buttons = QtWidgets.QVBoxLayout()
            top_buttons.setSpacing(8)
            top_buttons.addWidget(self.btn_load)
            top_buttons.addWidget(self.btn_load_folder)
            top_buttons.addWidget(self.btn_run)
            top_buttons.addWidget(self.btn_run_batch)
            top_buttons.addWidget(self.progress_bar)
    
            batch_box = CollapsibleBox("Batch navigation", expanded=False)
            self.folder_label = QtWidgets.QLabel("Folder: none")
            self.counter_label = QtWidgets.QLabel("Tooth: 0 / 0")
            self.combo_files = QtWidgets.QComboBox()
            self.combo_files.setEnabled(False)
            self.btn_prev = QtWidgets.QPushButton("Previous")
            self.btn_next = QtWidgets.QPushButton("Next")
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
    
            nav_row = QtWidgets.QHBoxLayout()
            nav_row.addWidget(self.btn_prev)
            nav_row.addWidget(self.btn_next)
    
            batch_box.addWidget(self.folder_label)
            batch_box.addWidget(self.counter_label)
            batch_box.addWidget(self.combo_files)
            batch_box.addLayout(nav_row)
    
            self.combo_mesh_style = QtWidgets.QComboBox()
            self.combo_mesh_style.addItems(["Shaded", "Elevation", "Slope", "DNE", "Orientation"])
    
            self.chk_show_edges = QtWidgets.QCheckBox("Show mesh edges")
            self.chk_show_edges.setChecked(False)
    
            settings_box = CollapsibleBox("Main settings", expanded=True)
            form = QtWidgets.QFormLayout()
            form.setLabelAlignment(QtCore.Qt.AlignLeft)
            form.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            form.setHorizontalSpacing(14)
            form.setVerticalSpacing(10)
            form.addRow("Mesh style", self.combo_mesh_style)
            settings_box.addLayout(form)
            settings_box.addWidget(self.chk_show_edges)
    
            self.spin_smooth = QtWidgets.QSpinBox()
            self.spin_smooth.setRange(0, 500)
            self.spin_smooth.setValue(0)
    
            self.spin_target_faces = QtWidgets.QSpinBox()
            self.spin_target_faces.setRange(0, 500000)
            self.spin_target_faces.setValue(10000)
    
            self.chk_level_z = QtWidgets.QCheckBox("Set lowest point to Z = 0")
            self.chk_level_z.setChecked(True)
    
            advanced_box = CollapsibleBox("Advanced settings", expanded=False)
            adv_form = QtWidgets.QFormLayout()
            adv_form.setLabelAlignment(QtCore.Qt.AlignLeft)
            adv_form.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            adv_form.setHorizontalSpacing(14)
            adv_form.setVerticalSpacing(10)
            adv_form.addRow("Smooth iterations", self.spin_smooth)
            adv_form.addRow("Target number of faces", self.spin_target_faces)
            advanced_box.addLayout(adv_form)
            advanced_box.addWidget(self.chk_level_z)
    
            self.btn_export_csv = QtWidgets.QPushButton("Export current CSV")
            self.btn_export_all = QtWidgets.QPushButton("Export all CSV")
            self.btn_export_csv.setEnabled(False)
            self.btn_export_all.setEnabled(False)
    
            self.chk_export_images = QtWidgets.QCheckBox("Export images")
            self.chk_export_images.setChecked(False)
    
            export_box = CollapsibleBox("Export", expanded=False)
            export_box.addWidget(self.chk_export_images)
            export_box.addWidget(self.btn_export_csv)
            export_box.addWidget(self.btn_export_all)
    
            self.log_box = QtWidgets.QPlainTextEdit()
            self.log_box.setReadOnly(True)
            self.log_box.setMaximumBlockCount(500)
            self.log_box.setMinimumHeight(180)
    
            log_box = CollapsibleBox("Log", expanded=False)
            log_box.addWidget(self.log_box)
    
            side_layout.addWidget(self.title_label)
            side_layout.addWidget(self.file_label)
            side_layout.addWidget(self.mean_slope_label)
            side_layout.addWidget(self.relief_label)
            side_layout.addWidget(self.rfi1_label)
            side_layout.addWidget(self.rfi2_label)
            side_layout.addWidget(self.extra_label)
            side_layout.addWidget(self.dne_label)
            side_layout.addWidget(self.opcr_label)
            side_layout.addWidget(self.md_label)
            side_layout.addWidget(self.bl_label)
            side_layout.addWidget(self.help_label)
            side_layout.addLayout(top_buttons)
            side_layout.addWidget(batch_box)
            side_layout.addWidget(settings_box)
            side_layout.addWidget(export_box)
            side_layout.addWidget(log_box)
            side_layout.addStretch()
            side_layout.addWidget(advanced_box)
    
            scroll.setWidget(scroll_content)
            side_panel_layout.addWidget(scroll)
    
            viewer_frame = QtWidgets.QFrame()
            viewer_frame.setObjectName("ViewerFrame")
            viewer_layout = QtWidgets.QVBoxLayout(viewer_frame)
            viewer_layout.setContentsMargins(6, 6, 6, 6)
    
            self.plotter = QtInteractor(self)
            viewer_layout.addWidget(self.plotter.interactor)
    
            root_layout.addWidget(side_panel)
            root_layout.addWidget(viewer_frame, stretch=1)
    
            self.btn_load.clicked.connect(self.load_mesh)
            self.btn_load_folder.clicked.connect(self.load_folder)
            self.btn_run.clicked.connect(self.run_metrics)
            self.btn_run_batch.clicked.connect(self.run_batch_metrics)
            self.btn_export_csv.clicked.connect(self.export_csv)
            self.btn_export_all.clicked.connect(self.export_all_csv)
            self.btn_prev.clicked.connect(self.prev_tooth)
            self.btn_next.clicked.connect(self.next_tooth)
            self.combo_files.currentIndexChanged.connect(self.jump_to_index)
            self.combo_mesh_style.currentIndexChanged.connect(self.refresh_scene)
            self.chk_show_edges.toggled.connect(self.refresh_scene)
    
            self.plotter.set_background("#f8fafc")
    
        def log(self, text: str):
            self.log_box.appendPlainText(text)
    
        def load_mesh(self):
            filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open mesh", "", "Mesh files (*.ply *.obj *.stl)"
            )
            if not filepath:
                return
    
            self.batch_files = [filepath]
            self.batch_results = {}
            self.current_folder = os.path.dirname(filepath)
    
            self.combo_files.blockSignals(True)
            self.combo_files.clear()
            self.combo_files.addItem(os.path.basename(filepath), filepath)
            self.combo_files.blockSignals(False)
    
            self._set_current_tooth(0)
            self.progress_bar.setValue(0)
            self.log(f"Loaded single file: {filepath}")
    
        def load_folder(self):
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder")
            if not folder:
                return
    
            exts = (".ply", ".obj", ".stl")
            files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]
            if not files:
                QtWidgets.QMessageBox.warning(self, "No meshes", "No .ply, .obj, or .stl files found in this folder.")
                return
    
            self.batch_files = files
            self.batch_results = {}
            self.current_folder = folder
    
            self.combo_files.blockSignals(True)
            self.combo_files.clear()
            for p in files:
                self.combo_files.addItem(os.path.basename(p), p)
            self.combo_files.blockSignals(False)
    
            self._set_current_tooth(0)
            self.progress_bar.setValue(0)
            self.log(f"Loaded folder with {len(files)} files.")
    
        def _set_current_tooth(self, index: int):
            if index < 0 or index >= len(self.batch_files):
                return
    
            self.current_index = index
            path = self.batch_files[index]
            self.current_mesh = self.engine.load_mesh(path)
            self.current_result = self.batch_results.get(path)
    
            self.file_label.setText(os.path.basename(path))
            self.folder_label.setText(f"Folder: {self.current_folder if self.current_folder else 'none'}")
            self.counter_label.setText(f"Tooth: {index + 1} / {len(self.batch_files)}")
    
            self.combo_files.blockSignals(True)
            self.combo_files.setCurrentIndex(index)
            self.combo_files.blockSignals(False)
    
            self.btn_prev.setEnabled(index > 0)
            self.btn_next.setEnabled(index < len(self.batch_files) - 1)
            self.combo_files.setEnabled(len(self.batch_files) > 0)
            self.btn_export_csv.setEnabled(self.current_result is not None)
            self.btn_export_all.setEnabled(len(self.batch_results) > 0)
    
            self._update_labels()
            self.refresh_scene()
    
        def _update_labels(self):
            if self.current_result is None:
                self.mean_slope_label.setText("Mean slope: -")
                self.relief_label.setText("Relief: -")
                self.rfi1_label.setText("RFI 1: -")
                self.rfi2_label.setText("RFI 2: -")
                self.extra_label.setText("Slope SD / simple relief: -")
                self.dne_label.setText("DNE: -")
                self.opcr_label.setText("OPCR: -")
                self.md_label.setText("Mesiodistal: -")
                self.bl_label.setText("Bucolingual: -")
                return

            r = self.current_result
            self.mean_slope_label.setText(f"Mean slope: {r.mean_slope:.3f}°")
            self.relief_label.setText(f"Relief: {r.relief:.3f}")
            self.rfi1_label.setText(f"RFI 1: {r.rfi_1:.5f}")
            self.rfi2_label.setText(f"RFI 2: {r.rfi_2:.5f}")
            self.extra_label.setText(
                f"Slope SD: {r.slope_sd:.3f} | Simple relief: {r.simple_relief:.3f}"
            )
            self.dne_label.setText(f"DNE: {r.dne:.4f}")
            self.opcr_label.setText(f"OPCR: {r.opcr:.3f}")
            self.md_label.setText(f"Mesiodistal: {r.md:.3f} mm")
            self.bl_label.setText(f"Bucolingual: {r.bl:.3f} mm")
    
        def run_metrics(self):
            if self.current_mesh is None:
                QtWidgets.QMessageBox.warning(self, "No mesh", "Load a tooth mesh or a folder first.")
                return
    
            try:
                self.current_result = self.engine.compute_metrics(
                    smooth_iterations=self.spin_smooth.value(),
                    target_faces=self.spin_target_faces.value(),
                    set_origin_to_lowest=self.chk_level_z.isChecked(),
                )
    
                if 0 <= self.current_index < len(self.batch_files):
                    path = self.batch_files[self.current_index]
                    self.batch_results[path] = self.current_result
    
                self.progress_bar.setValue(100)
                self._update_labels()
                self.btn_export_csv.setEnabled(True)
                self.btn_export_all.setEnabled(len(self.batch_results) > 0)
    
                self.log(
                    f"Computed | mean_slope={self.current_result.mean_slope:.3f} | "
                    f"relief={self.current_result.relief:.3f} | "
                    f"rfi_1={self.current_result.rfi_1:.5f} | "
                    f"rfi_2={self.current_result.rfi_2:.5f} | "
                    f"DNE={self.current_result.dne:.4f} | "
                    f"OPCR={self.current_result.opcr} | "
                    f"MD={self.current_result.md:.3f} mm | "
                    f"BL={self.current_result.bl:.3f} mm"
                )
                self.refresh_scene()
    
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Computation error", str(e))
                self.log(f"Computation error: {e}")
    
        def run_batch_metrics(self):
            if not self.batch_files:
                QtWidgets.QMessageBox.warning(self, "No folder", "Load a folder or a file first.")
                return
    
            total = len(self.batch_files)
            failed = []
            self.progress_bar.setValue(0)
            QtWidgets.QApplication.processEvents()
    
            for idx, path in enumerate(self.batch_files, start=1):
                try:
                    self.engine.load_mesh(path)
                    result = self.engine.compute_metrics(
                        smooth_iterations=self.spin_smooth.value(),
                        target_faces=self.spin_target_faces.value(),
                        set_origin_to_lowest=self.chk_level_z.isChecked(),
                    )
                    self.batch_results[path] = result
                    self.log(
                        f"OK: {os.path.basename(path)} | "
                        f"slope={result.mean_slope:.3f} | "
                        f"relief={result.relief:.3f} | "
                        f"rfi_1={result.rfi_1:.5f} | "
                        f"rfi_2={result.rfi_2:.5f} | "
                        f"DNE={result.dne:.4f} | "
                        f"OPCR={result.opcr} | "
                        f"MD={result.md:.3f} | BL={result.bl:.3f}"
                    )
                except Exception as e:
                    failed.append((path, str(e)))
                    self.log(f"FAILED: {os.path.basename(path)} -> {e}")
    
                self.progress_bar.setValue(int(idx * 100 / total))
                QtWidgets.QApplication.processEvents()
    
            if self.batch_files:
                self._set_current_tooth(max(self.current_index, 0))
    
            combined_path = os.path.join(self.current_folder if self.current_folder else os.getcwd(), "ALL_Topography.csv")
            try:
                self.engine.export_batch_csv(combined_path, self.batch_results, self.batch_files)
                self.log(f"Combined batch CSV saved: {combined_path}")
            except Exception as e:
                self.log(f"Could not save combined CSV: {e}")
    
            if failed:
                QtWidgets.QMessageBox.warning(self, "Batch finished", f"Batch completed with {len(failed)} failed files.")
            else:
                QtWidgets.QMessageBox.information(self, "Batch finished", "Batch metrics completed successfully.")
    
        def jump_to_index(self, index: int):
            if index >= 0:
                self._set_current_tooth(index)
    
        def prev_tooth(self):
            if self.current_index > 0:
                self._set_current_tooth(self.current_index - 1)
    
        def next_tooth(self):
            if self.current_index < len(self.batch_files) - 1:
                self._set_current_tooth(self.current_index + 1)
    
        def _prepare_display_mesh(self):
            if self.current_result is None:
                if self.current_mesh is None:
                    return None, {}
                mesh = self.current_mesh.copy(deep=True)
                return mesh, {
                    "color": "white",
                    "smooth_shading": True,
                    "show_edges": self.chk_show_edges.isChecked(),
                    "name": "mesh"
                }
    
            mesh = self.current_result.mesh_for_display.copy(deep=True)
            style = self.combo_mesh_style.currentText()
            show_edges = self.chk_show_edges.isChecked()
    
            kwargs = {
                "opacity": 1.0,
                "show_edges": show_edges,
                "name": "mesh",
            }
    
            if style == "Shaded":
                kwargs.update({
                    "color": "white",
                    "smooth_shading": True,
                    "specular": 0.18,
                    "ambient": 0.28,
                    "diffuse": 0.82,
                })
                return mesh, kwargs
    
            if style == "Elevation":
                mesh["ElevationZ"] = mesh.points[:, 2]
                kwargs.update({
                    "scalars": "ElevationZ",
                    "cmap": "terrain",
                    "smooth_shading": True,
                    "specular": 0.10,
                })
                return mesh, kwargs
    
            if style == "Slope":
                slopes = self.engine.compute_triangle_slopes(mesh)
                mesh.cell_data["SlopeDeg"] = slopes
                kwargs.update({
                    "scalars": "SlopeDeg",
                    "cmap": "turbo",
                    "smooth_shading": False,
                    "specular": 0.05,
                })
                return mesh, kwargs

            if style == "DNE" and "DNE_face" in mesh.cell_data.keys():
                kwargs.update({
                    "scalars": "DNE_face",
                    "cmap": "inferno",
                    "smooth_shading": False,
                    "specular": 0.05,
                })
                return mesh, kwargs

            if style == "Orientation" and "OrientBin" in mesh.cell_data.keys():
                kwargs.update({
                    "scalars": "OrientBin",
                    "cmap": "tab10",
                    "clim": [0, 7],
                    "smooth_shading": False,
                    "specular": 0.05,
                })
                return mesh, kwargs

            return mesh, kwargs
    
        def _prepare_export_mesh(self, result: TopographyResult, style: str):
            mesh = result.mesh_for_display.copy(deep=True)
    
            kwargs = {
                "show_edges": self.chk_show_edges.isChecked(),
                "opacity": 1.0,
            }
    
            scalar_bar_args = {
                "title_font_size": 14,
                "label_font_size": 12,
                "shadow": False,
                "italic": False,
                "bold": False,
                "fmt": "%.2f",
                "font_family": "arial",
            }
    
            if style == "Elevation":
                mesh["ElevationZ"] = mesh.points[:, 2]
                kwargs.update({
                    "scalars": "ElevationZ",
                    "cmap": "terrain",
                    "smooth_shading": True,
                    "scalar_bar_args": scalar_bar_args,
                })
                return mesh, kwargs
    
            if style == "Slope":
                slopes = self.engine.compute_triangle_slopes(mesh)
                mesh.cell_data["SlopeDeg"] = slopes
                kwargs.update({
                    "scalars": "SlopeDeg",
                    "cmap": "turbo",
                    "smooth_shading": False,
                    "scalar_bar_args": scalar_bar_args,
                })
                return mesh, kwargs
    
            kwargs.update({
                "color": "white",
                "smooth_shading": True,
            })
            return mesh, kwargs
    
        def export_metric_image(self, result: TopographyResult, outpath: str, style: str):
            plotter = pv.Plotter(off_screen=True, window_size=(1800, 1800))
            plotter.set_background("white")
    
            mesh, kwargs = self._prepare_export_mesh(result, style)
            plotter.add_mesh(mesh, **kwargs)
    
            plotter.show_axes()
            plotter.add_axes()
            plotter.view_xy()
            plotter.camera.zoom(1.15)
    
            plotter.screenshot(outpath)
            plotter.close()
    
        def refresh_scene(self):
            self.plotter.clear()
    
            display_mesh, mesh_kwargs = self._prepare_display_mesh()
            if display_mesh is not None:
                self.plotter.add_mesh(display_mesh, **mesh_kwargs)
                self.plotter.enable_eye_dome_lighting()
    
            self.plotter.show_axes()
            self.plotter.view_xy()
            self.plotter.reset_camera()
    
        def export_csv(self):
            if self.current_result is None:
                return
    
            default_name = "topography.csv"
            if 0 <= self.current_index < len(self.batch_files):
                default_name = os.path.splitext(os.path.basename(self.batch_files[self.current_index]))[0] + "_topography.csv"
    
            filepath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", default_name, "CSV files (*.csv)")
            if not filepath:
                return
    
            self.engine.export_single_csv(filepath, self.current_result)
            self.log(f"CSV saved: {filepath}")
    
            if self.chk_export_images.isChecked():
                base = os.path.splitext(filepath)[0]
                elevation_path = base + "_elevation.png"
                slope_path = base + "_slope.png"
    
                try:
                    self.export_metric_image(self.current_result, elevation_path, "Elevation")
                    self.export_metric_image(self.current_result, slope_path, "Slope")
                    self.log(f"Images saved: {elevation_path} | {slope_path}")
                except Exception as e:
                    self.log(f"Image export failed: {e}")
    
        def export_all_csv(self):
            if not self.batch_results:
                QtWidgets.QMessageBox.warning(self, "No results", "No computed results available to export.")
                return
    
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder")
            if not folder:
                return
    
            export_images = self.chk_export_images.isChecked()
    
            for path, result in self.batch_results.items():
                base = os.path.splitext(os.path.basename(path))[0]
    
                csv_out = os.path.join(folder, f"{base}_topography.csv")
                self.engine.export_single_csv(csv_out, result)
    
                if export_images:
                    elevation_out = os.path.join(folder, f"{base}_elevation.png")
                    slope_out = os.path.join(folder, f"{base}_slope.png")
    
                    try:
                        self.export_metric_image(result, elevation_out, "Elevation")
                        self.export_metric_image(result, slope_out, "Slope")
                    except Exception as e:
                        self.log(f"Image export failed for {base}: {e}")
    
            combined = os.path.join(folder, "ALL_Topography.csv")
            self.engine.export_batch_csv(combined, self.batch_results, self.batch_files)
    
            if export_images:
                self.log(f"Exported CSV files, combined CSV, and PNG images to: {folder}")
            else:
                self.log(f"Exported CSV files and combined CSV to: {folder}")
    
    
    # =========================================================
    # MAIN
    # =========================================================
    def main():
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle("Fusion")
        win = TopographyWindow()
        win.show()
        sys.exit(app.exec_())
    
    
    if __name__ == "__main__":
        main()
except (ImportError, NameError):
    pass
