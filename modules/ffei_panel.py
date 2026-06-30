"""
DAMOS – FFEI Panel  (Food-Flow Estimation Index)
Uses the ffei engine for watershed-based topographic-functional analysis.

Metrics computed:
  FFEI             – proportion of occlusal area draining to central fovea
  n_basins         – number of functional drainage basins
  n_cusps          – detected cusp count
  n_depressions    – detected fovea/depression count
  slope_mean       – mean face slope (degrees)
  slope_sd         – slope standard deviation
  z_range          – max Z − min Z (mm)
  cusp_fovea_depth – relative depth between mean cusp and deepest fovea
  basin_area_gini  – inequality of basin sizes (0 = equal, 1 = one dominates)
  primary_fovea_depth – depth of deepest fovea (mm)

Display modes:
  Basins     – watershed regions colour-coded
  Height     – elevation colourmap
  Slope      – slope magnitude colourmap
  Shaded     – neutral shading (bone-like)
"""

import os
import csv
from typing import Optional, Dict

import numpy as np
import pyvista as pv

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QPlainTextEdit, QProgressBar, QFileDialog,
    QScrollArea, QFrame, QLabel, QLineEdit, QComboBox, QSlider
)
from PySide6.QtCore import QThread, Signal, QObject, QTimer, Qt

from gui.app_state import AppState
from gui.viewer3d import Viewer3D

# ── Import FFEI engine ────────────────────────────────────
# The ffei package lives next to this file or in the DAMOS root
import sys
_DAMOS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _DAMOS_ROOT)
sys.path.insert(0, os.path.join(_DAMOS_ROOT, "ffei"))

try:
    from ffei.surface_fields import compute_surface_fields, SurfaceFields
    from ffei.detection import (
        detect_minima, detect_maxima, cluster_features, DetectedFeatures
    )
    from ffei.watershed import (
        analyze_basins,
        WatershedResult, FFEIResult, BasinVolumes
    )
    from ffei.metrics import extract_metrics
    from ffei.escape_channels import compute_escape_channels, paths_to_polydata
    _ENGINE_OK = True
except ImportError as _imp_err:
    _ENGINE_OK = False
    _ENGINE_ERR = str(_imp_err)

# Import AutoLMK's MorphologyEngine for cusp detection
try:
    from modules.autolmk_panel import MorphologyEngine
    _AUTOLMK_OK = True
except ImportError:
    _AUTOLMK_OK = False


# ── Basin colour palette ──────────────────────────────────

_BASIN_COLORS = [
    "#4DC9F6", "#F67019", "#5CD68A", "#E8A838",
    "#AB7FD6", "#E6426E", "#72D4D4", "#F2BE54",
    "#8CAEE8", "#C6DA44", "#D98CB0", "#80C09A",
    "#C09E72", "#9AD472", "#D472D4", "#72A0C0",
    "#E69898", "#8CE6B4", "#CCCC80", "#A872B4",
]


# ─────────────────────────────────────────────────────────────────────────────
# Analysis result container
# ─────────────────────────────────────────────────────────────────────────────

class FFEIAnalysisResult:
    """Holds everything needed for display and export."""
    __slots__ = (
        "specimen_id", "vertices", "faces", "pv_mesh",
        "fields", "minima", "maxima",
        "watershed", "ffei_result", "basin_volumes", "metrics",
        # Escape channels (computed in worker thread — pure numpy)
        "escape_paths", "_adjacency",
        # Escape channels PyVista PolyData (built on main thread, two types)
        "escape_channels_pv", "retention_channels_pv", "path_types",
    )
    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)


def _detect_landmarks(
    pv_mesh: pv.PolyData,
    specimen_id: str,
    tooth_type: str = "M1 lower – Sapiens (5 cusps)",
    n_cusps: int = 5,
) -> FFEIAnalysisResult:
    """Phase 1: detect cusps, depressions, and compute surface fields.

    This is expensive but only needs to run once per mesh.
    Results are cached and reused when watershed parameters change.
    """
    vertices = np.array(pv_mesh.points, dtype=np.float64)

    if pv_mesh.is_all_triangles:
        faces_flat = pv_mesh.faces.reshape(-1, 4)[:, 1:]
    else:
        tri = pv_mesh.triangulate()
        faces_flat = tri.faces.reshape(-1, 4)[:, 1:]
        vertices = np.array(tri.points, dtype=np.float64)
    faces = np.array(faces_flat, dtype=np.int64)

    res = FFEIAnalysisResult()
    res.specimen_id = specimen_id
    res.vertices = vertices
    res.faces = faces
    res.pv_mesh = pv_mesh

    # Layer 1: Surface fields
    res.fields = compute_surface_fields(vertices, faces)

    # Layer 2: Feature detection
    if _AUTOLMK_OK:
        try:
            engine = MorphologyEngine()
            engine.points = vertices

            params = engine.get_detection_parameters(tooth_type, n_cusps)

            raw_cusps = engine.detect_cusps(
                vertices,
                n_cusps=params["n_cusps"],
                min_distance=params["min_distance"],
                alpha=params["alpha"],
            )
            refined_cusps = engine.refine_cusps_to_local_maxima(
                vertices, raw_cusps,
                refine_radius=params["refine_radius"],
            )
            ordered_cusps = engine.order_points_consistently(
                refined_cusps, tooth_type,
            )

            cusp_indices = []
            for c in ordered_cusps:
                dists = np.linalg.norm(vertices - c, axis=1)
                cusp_indices.append(np.argmin(dists))
            cusp_indices = np.array(cusp_indices, dtype=int)

            res.maxima = DetectedFeatures(
                indices=cusp_indices,
                positions=vertices[cusp_indices],
                values=res.fields.height[cusp_indices],
                labels=np.arange(len(cusp_indices)),
            )

            intercuspal = engine.detect_intercuspal_points(
                vertices, ordered_cusps, tooth_type,
            )
            central = engine.detect_central_point(vertices, ordered_cusps)

            all_depressions = np.vstack([intercuspal, central.reshape(1, 3)])
            dep_indices = []
            for d in all_depressions:
                dists = np.linalg.norm(vertices - d, axis=1)
                dep_indices.append(np.argmin(dists))
            dep_indices = np.array(dep_indices, dtype=int)

            res.minima = DetectedFeatures(
                indices=dep_indices,
                positions=vertices[dep_indices],
                values=res.fields.height[dep_indices],
                labels=np.arange(len(dep_indices)),
            )

        except Exception:
            _detect_features_fallback(res, vertices, faces, tooth_type)
    else:
        _detect_features_fallback(res, vertices, faces, tooth_type)

    return res


def _compute_watershed(
    res: FFEIAnalysisResult,
    merge_depth: float,
    merge_area: float,
) -> FFEIAnalysisResult:
    """Phase 2: watershed + merge + FFEI.

    Reuses the landmarks and surface fields already stored in `res`.
    Can be called repeatedly with different parameters without
    re-detecting landmarks.
    """
    vertices = res.vertices
    faces = res.faces

    # Priority-flood watershed (Zuccotti 1998 in 3D).
    # merge_depth is repurposed as min_basin_depth_mm.
    res.watershed, res.ffei_result, res.basin_volumes = analyze_basins(
        vertices, faces,
        min_basin_depth_mm=max(0.001, float(merge_depth)),
    )

    res.metrics = extract_metrics(
        vertices, faces, res.fields,
        res.minima, res.maxima,
        res.watershed, res.ffei_result,
        basin_volumes=res.basin_volumes,
        specimen_id=res.specimen_id,
    )

    # ── Escape channels (pure numpy — safe on worker thread) ──────────────
    try:
        if res.maxima is not None and len(res.maxima.indices) > 0:
            ch = compute_escape_channels(
                vertices, faces, res.fields.height, res.maxima.indices
            )
            res.escape_paths = ch["paths"]
            res._adjacency   = ch["adjacency"]
            res.path_types   = ch["path_types"]
        else:
            res.escape_paths = []
            res._adjacency   = []
            res.path_types   = []
    except Exception:
        res.escape_paths = []
        res._adjacency   = []
        res.path_types   = []

    return res


def _run_ffei_analysis(
    pv_mesh: pv.PolyData,
    specimen_id: str,
    merge_depth: float,
    merge_area: float,
    tooth_type: str = "M1 lower – Sapiens (5 cusps)",
    n_cusps: int = 5,
) -> FFEIAnalysisResult:
    """Full pipeline (for batch mode). Calls both phases sequentially."""
    res = _detect_landmarks(pv_mesh, specimen_id, tooth_type, n_cusps)
    return _compute_watershed(res, merge_depth, merge_area)


def _detect_features_fallback(res, vertices, faces,
                              tooth_type="Auto (generic)"):
    """Original FFEI feature detection (when AutoLMK is not available).

    Uses tooth-type-specific parameters from detection presets to avoid
    filtering out legitimate features on bilophodont or non-standard teeth.
    """
    from ffei.detection import get_detection_preset

    preset = get_detection_preset(tooth_type)

    raw_min = detect_minima(
        vertices, faces, res.fields.height,
        slope=res.fields.slope,
        centroid_filter=preset["centroid_filter_minima"],
        depth_threshold=preset["depth_threshold"],
        slope_max=preset["slope_max"],
    )
    res.minima = cluster_features(raw_min)

    raw_max = detect_maxima(
        vertices, faces, res.fields.height,
        centroid_filter=preset["centroid_filter_maxima"],
        prominence_threshold=preset["prominence_threshold"],
    )
    res.maxima = cluster_features(raw_max)


# ─────────────────────────────────────────────────────────────────────────────
# Worker (batch processing in background thread)
# ─────────────────────────────────────────────────────────────────────────────

# ── Max faces for FFEI computation (auto-decimate if larger) ──────────────
_FFEI_MAX_FACES = 8_000


def _prepare_mesh_for_ffei(pv_mesh: pv.PolyData) -> pv.PolyData:
    """Ensure the mesh is triangulated and not too dense for FFEI.

    FFEI's watershed algorithm is O(N_faces); meshes above _FFEI_MAX_FACES
    are decimated to keep computation under a few seconds.
    """
    mesh = pv_mesh.extract_surface().triangulate()
    if mesh.n_cells > _FFEI_MAX_FACES:
        target = _FFEI_MAX_FACES
        reduction = 1.0 - target / mesh.n_cells
        reduction = min(max(reduction, 0.0), 0.95)
        try:
            dec = mesh.decimate(reduction)
            if dec.n_cells > 0:
                mesh = dec.extract_surface().triangulate()
        except Exception:
            pass   # fall through with original
    return mesh


# ── Single-mesh worker (background QThread) ────────────────────────────────

class _SingleMeshWorker(QObject):
    """Run the full FFEI pipeline for one mesh in a background thread."""
    status  = Signal(str)          # progress messages
    done    = Signal(object)       # FFEIAnalysisResult or None on error
    error   = Signal(str)

    def __init__(self, path: str, params: dict,
                 landmark_cache: dict, specimen_id: str):
        super().__init__()
        self._path   = path
        self._params = params
        self._cache  = landmark_cache
        self._sid    = specimen_id

    def run(self):
        try:
            tooth_type = self._params.get("tooth_type", "Auto (generic)")
            n_cusps    = self._params.get("n_cusps", 5)
            cache_key  = (self._path, tooth_type, n_cusps)

            if cache_key in self._cache:
                result = self._cache[cache_key]
                self.status.emit(
                    f"Using cached landmarks "
                    f"({len(result.maxima.indices)} cusps)"
                )
            else:
                self.status.emit("Loading and preparing mesh…")
                raw_mesh = pv.read(self._path)
                mesh = _prepare_mesh_for_ffei(raw_mesh)
                n_orig = raw_mesh.n_cells
                n_used = mesh.n_cells
                if n_used < n_orig:
                    self.status.emit(
                        f"Mesh decimated: {n_orig} → {n_used} faces "
                        f"(FFEI limit {_FFEI_MAX_FACES})"
                    )

                self.status.emit("Computing surface fields…")
                result = _detect_landmarks(
                    mesh, self._sid,
                    tooth_type=tooth_type,
                    n_cusps=n_cusps,
                )
                self._cache[cache_key] = result
                src = "AutoLMK" if _AUTOLMK_OK else "fallback"
                self.status.emit(
                    f"Features detected ({src}): "
                    f"{len(result.maxima.indices)} cusps, "
                    f"{len(result.minima.indices)} depressions"
                )

            self.status.emit("Running watershed segmentation…")
            result = _compute_watershed(
                result,
                merge_depth=self._params.get("merge_depth", 0.02),
                merge_area=self._params.get("merge_area",  0.01),
            )
            self.done.emit(result)

        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class _FFEIWorker(QObject):
    progress = Signal(int, int, str, object)   # current, total, message, result
    finished = Signal()

    def __init__(self, files: list, params: dict, output_dir: str):
        super().__init__()
        self._files = files
        self._params = params
        self._output_dir = output_dir
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        if not _ENGINE_OK:
            self.progress.emit(0, 1, f"ERROR: FFEI engine not found: {_ENGINE_ERR}", None)
            self.finished.emit()
            return

        n = len(self._files)
        all_metrics = []
        os.makedirs(self._output_dir, exist_ok=True)

        for i, path in enumerate(self._files, 1):
            if self._abort:
                self.progress.emit(i, n, "Aborted.", None)
                break

            fname = os.path.basename(path)
            specimen_id = os.path.splitext(fname)[0]

            try:
                mesh = pv.read(path)
                result = _run_ffei_analysis(
                    mesh, specimen_id,
                    merge_depth=self._params.get("merge_depth", 0.02),
                    merge_area=self._params.get("merge_area", 0.01),
                )
                all_metrics.append(result.metrics)

                # Per-specimen CSV
                base = os.path.splitext(fname)[0]
                csv_path = os.path.join(self._output_dir, f"{base}_ffei.csv")
                _write_single_csv(csv_path, result.metrics)

                self.progress.emit(
                    i, n,
                    f"OK  {fname}   FFEI={result.ffei_result.ffei:.4f}  "
                    f"basins={result.watershed.n_basins}  "
                    f"slope={result.metrics['slope_mean_deg']:.1f}°",
                    result,
                )
            except Exception as e:
                self.progress.emit(i, n, f"FAIL  {fname}: {e}", None)

        # Combined CSV
        if all_metrics:
            combined = os.path.join(self._output_dir, "ALL_FFEI.csv")
            _write_batch_csv(combined, all_metrics)

        self.finished.emit()


def _write_single_csv(path: str, metrics: dict):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        w.writeheader()
        w.writerow(metrics)


def _write_batch_csv(path: str, metrics_list: list):
    if not metrics_list:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics_list[0].keys()))
        w.writeheader()
        w.writerows(metrics_list)


# ─────────────────────────────────────────────────────────────────────────────
# Shear index helper (module-level, uses numpy only)
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# Panel
# ─────────────────────────────────────────────────────────────────────────────

class FFEIPanel(QWidget):
    """FFEI — Food-Flow Estimation Index analysis panel."""

    def __init__(self, state: AppState, viewer: Viewer3D, parent=None):
        super().__init__(parent)
        self._state = state
        self._viewer = viewer
        self._current_result: Optional[FFEIAnalysisResult] = None

        # Batch worker
        self._worker: Optional[_FFEIWorker] = None
        self._worker_thread: Optional[QThread] = None

        # Single-mesh worker (background thread)
        self._single_worker = None
        self._single_worker_thread: Optional[QThread] = None

        # Landmark / surface-field cache
        # Key: (filepath, tooth_type, n_cusps) → FFEIAnalysisResult (phase 1 only)
        self._landmark_cache: dict = {}

        self._build_ui()
        self._connect_signals()

    # ── UI ─────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        if not _ENGINE_OK:
            warn = QLabel(
                f"FFEI engine not found.\n{_ENGINE_ERR}\n\n"
                "Place the ffei/ package in the DAMOS root folder and restart."
            )
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #F0A030; padding: 12px;")
            root.addWidget(warn)
            root.addStretch(1)
            return

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── Tooth Type ─────────────────────────────────────
        grp_tooth = QGroupBox("Tooth Configuration")
        f_tooth = QFormLayout(grp_tooth)

        self.cmb_tooth_type = QComboBox()
        self.cmb_tooth_type.addItems([
            "Auto (generic)",
            "M1 lower – Sapiens (5 cusps)",
            "M1 upper – Sapiens (4 cusps)",
            "M2 lower – Sapiens (4 cusps)",
            "M2 upper – Sapiens (4 cusps)",
            "Pm – Sapiens (2 cusps)",
            "Bilophodont (4 cusps)",
        ])
        self.cmb_tooth_type.setToolTip(
            "Determines cusp count, detection parameters,\n"
            "and watershed sensitivity.\n\n"
            "M1 lower – Sapiens: 5 cusps (Protoconid–Hypoconulid)\n"
            "M1 upper – Sapiens: 4 cusps (Paracone–Protocone)\n"
            "M2 lower/upper – Sapiens: 4 cusps\n"
            "Pm – Sapiens: 2 cusps (Buccal–Lingual)\n"
            "Bilophodont: 4 cusps (cercopithecid molars)"
        )
        f_tooth.addRow("Tooth type:", self.cmb_tooth_type)

        self.spn_n_cusps = QSpinBox()
        self.spn_n_cusps.setRange(2, 7)
        self.spn_n_cusps.setValue(5)
        self.spn_n_cusps.setToolTip("Number of cusps to detect (used by Auto mode)")
        f_tooth.addRow("Cusps to detect:", self.spn_n_cusps)

        # Sync n_cusps with tooth type
        self.cmb_tooth_type.currentTextChanged.connect(self._on_tooth_type_changed)

        layout.addWidget(grp_tooth)

        # ── Parameters ────────────────────────────────────
        grp_params = QGroupBox("Watershed Parameters")
        f_params = QFormLayout(grp_params)

        self.spn_merge_depth = QDoubleSpinBox()
        self.spn_merge_depth.setRange(0.001, 0.50)
        self.spn_merge_depth.setSingleStep(0.005)
        self.spn_merge_depth.setValue(0.05)
        self.spn_merge_depth.setDecimals(3)
        self.spn_merge_depth.setToolTip(
            "Minimum water depth (mm) for a basin to be retained.\n"
            "Smaller values → more, shallower basins detected.\n"
            "Larger values → only clearly defined basins counted."
        )
        f_params.addRow("Min basin depth (mm):", self.spn_merge_depth)

        self.spn_merge_area = QDoubleSpinBox()
        self.spn_merge_area.setRange(0.005, 0.20)
        self.spn_merge_area.setSingleStep(0.005)
        self.spn_merge_area.setValue(0.020)
        self.spn_merge_area.setDecimals(3)
        self.spn_merge_area.setEnabled(False)
        self.spn_merge_area.setToolTip("Not used in the priority-flood pipeline.")
        f_params.addRow("(unused):", self.spn_merge_area)

        layout.addWidget(grp_params)

        # ── Display ───────────────────────────────────────
        grp_view = QGroupBox("Display Style")
        f_view = QFormLayout(grp_view)
        self.cmb_style = QComboBox()
        self.cmb_style.addItems([
            "Basins", "Height", "Slope",
            "Shaded", "Water fill", "Flow accumulation",
        ])
        f_view.addRow("Colour map:", self.cmb_style)

        self.cmb_features = QComboBox()
        self.cmb_features.addItems([
            "None", "Cusps", "Foveae", "Both",
            "Escape channels", "All",
        ])
        self.cmb_features.setCurrentIndex(3)
        f_view.addRow("Show features:", self.cmb_features)

        # Water level control (only visible when "Water fill" style is active)
        self._water_level_label = QLabel("Water level:")
        self._water_ctrl = QWidget()
        wc_lay = QHBoxLayout(self._water_ctrl)
        wc_lay.setContentsMargins(0, 0, 0, 0)
        wc_lay.setSpacing(6)
        self.sld_water_level = QSlider(Qt.Orientation.Horizontal)
        self.sld_water_level.setRange(0, 500)
        self.sld_water_level.setValue(100)
        self.sld_water_level.setTickInterval(100)
        self.sld_water_level.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sld_water_level.setToolTip(
            "Manual water level as % of the computed spillover depth.\n"
            "100% = fills to the saddle point.\n"
            "> 100% = hypothetical overfill for exploration (up to 500%)."
        )
        self.lbl_water_pct = QLabel("100%")
        self.lbl_water_pct.setMinimumWidth(42)
        self.chk_fix_water = QPushButton("Fixa")
        self.chk_fix_water.setCheckable(True)
        self.chk_fix_water.setChecked(False)
        self.chk_fix_water.setFixedWidth(44)
        self.chk_fix_water.setToolTip(
            "When pressed: water level is preserved when recomputing.\n"
            "When released: resets to 100% on each new computation."
        )
        wc_lay.addWidget(self.sld_water_level)
        wc_lay.addWidget(self.lbl_water_pct)
        wc_lay.addWidget(self.chk_fix_water)
        f_view.addRow(self._water_level_label, self._water_ctrl)
        self._water_level_label.setVisible(False)
        self._water_ctrl.setVisible(False)

        layout.addWidget(grp_view)

        # ── Compute ───────────────────────────────────────
        grp_cur = QGroupBox("Current Mesh")
        f_cur = QVBoxLayout(grp_cur)

        self.btn_compute = QPushButton("Compute FFEI")
        self.btn_compute.setObjectName("PrimaryButton")
        self.btn_compute.setMinimumHeight(36)
        f_cur.addWidget(self.btn_compute)

        # FFEI headline result
        self.lbl_ffei_big = QLabel("—")
        from PySide6.QtCore import Qt as _Qt
        self.lbl_ffei_big.setAlignment(_Qt.AlignmentFlag.AlignCenter)
        self.lbl_ffei_big.setStyleSheet(
            "font-size: 28px; font-weight: 700; color: #4FC3F7; "
            "padding: 8px 0; letter-spacing: -1px;"
        )
        f_cur.addWidget(self.lbl_ffei_big)

        # FFEI interpretation guide
        interp = QLabel(
            "<b>FFEI</b> = proportion of occlusal area that <i>retains</i> food "
            "(drains to an internal fovea rather than off the tooth edge).<br>"
            "<span style='color:#4FC3F7'>■</span> Coloured basins = <b>retention</b> &nbsp;"
            "<span style='color:#888'>■</span> Grey basins = <b>escape</b>"
        )
        interp.setWordWrap(True)
        interp.setStyleSheet(
            "color: #7A9AB8; background: #1A2535; border: 1px solid #2A3F5E; "
            "border-radius: 5px; padding: 5px 7px; font-size: 10px;"
        )
        f_cur.addWidget(interp)

        # Metrics table
        _metrics_def = [
            ("FFEI:",                "lbl_ffei"),
            ("Basins (total):",      "lbl_basins"),
            ("  Retention:",         "lbl_retention_n"),
            ("  Escape:",            "lbl_escape_n"),
            ("Cusps:",               "lbl_cusps"),
            ("Depressions:",         "lbl_deps"),
            ("Retention area (mm²):","lbl_ret_area"),
            ("Escape area (mm²):",   "lbl_esc_area"),
            ("Mean slope (°):",      "lbl_slope"),
            ("Max slope (°):",       "lbl_slope_max"),
            ("Slope SD (°):",        "lbl_slope_sd"),
            ("Z range (mm):",        "lbl_zrange"),
            ("Height SD:",           "lbl_hsd"),
            ("Cusp–fovea depth:",    "lbl_cf_depth"),
            ("Fovea depth (mm):",    "lbl_fovea_d"),
            ("Basin Gini:",          "lbl_gini"),
            ("Total area (mm²):",    "lbl_tarea"),
            ("Basin vol. (mm³):",    "lbl_basin_vol"),
            ("Ret. volume (mm³):",   "lbl_ret_vol"),
            ("Central vol. (mm³):",  "lbl_central_vol"),
        ]
        f_res = QFormLayout()
        for label, attr in _metrics_def:
            lbl = QLabel("—")
            lbl.setObjectName("ValueLabel")
            setattr(self, attr, lbl)
            f_res.addRow(label, lbl)
        f_cur.addLayout(f_res)
        layout.addWidget(grp_cur)

        # ── Export ────────────────────────────────────────
        grp_exp = QGroupBox("Export")
        f_exp = QVBoxLayout(grp_exp)
        self.btn_export_csv = QPushButton("Save current CSV")
        self.btn_export_csv.setObjectName("NavButton")
        f_exp.addWidget(self.btn_export_csv)
        layout.addWidget(grp_exp)

        # ── Batch ─────────────────────────────────────────
        grp_batch = QGroupBox("Batch Processing")
        f_batch = QVBoxLayout(grp_batch)

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

    # ── Connections ────────────────────────────────────────

    def _connect_signals(self):
        if not _ENGINE_OK:
            return
        self.btn_compute.clicked.connect(self._on_compute)
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        self.btn_browse_in.clicked.connect(self._browse_input)
        self.btn_browse_out.clicked.connect(self._browse_output)
        self.btn_batch_run.clicked.connect(self._on_batch_run)
        self.btn_batch_abort.clicked.connect(self._on_batch_abort)
        self.cmb_style.currentIndexChanged.connect(self._on_style_changed)
        self.cmb_features.currentIndexChanged.connect(self._refresh_scene)
        self.sld_water_level.valueChanged.connect(self._on_water_level_changed)
        self._state.current_index_changed.connect(self._on_index_changed)

    # ── Style + water level ────────────────────────────────

    def _on_style_changed(self):
        is_water = self.cmb_style.currentText() == "Water fill"
        self._water_level_label.setVisible(is_water)
        self._water_ctrl.setVisible(is_water)
        self._refresh_scene()

    def _on_water_level_changed(self, val: int):
        self.lbl_water_pct.setText(f"{val}%")
        self._refresh_scene()

    # ── Tooth type sync ────────────────────────────────────

    # Watershed merge presets per tooth type.
    # Bilophodont teeth need gentler merging to preserve the narrow
    # transverse basins between lophs and small lateral foveae.
    _WATERSHED_PRESETS = {
        "M1 lower – Sapiens (5 cusps)":  {"n_cusps": 5, "merge_depth": 0.05, "merge_area": 0.020},
        "M1 upper – Sapiens (4 cusps)":  {"n_cusps": 4, "merge_depth": 0.05, "merge_area": 0.020},
        "M2 lower – Sapiens (4 cusps)":  {"n_cusps": 4, "merge_depth": 0.05, "merge_area": 0.020},
        "M2 upper – Sapiens (4 cusps)":  {"n_cusps": 4, "merge_depth": 0.05, "merge_area": 0.020},
        "Pm – Sapiens (2 cusps)":        {"n_cusps": 2, "merge_depth": 0.05, "merge_area": 0.020},
        "Bilophodont (4 cusps)":         {"n_cusps": 4, "merge_depth": 0.03, "merge_area": 0.015},
    }

    def _on_tooth_type_changed(self, text: str):
        """Update n_cusps and watershed parameters when tooth type changes.
        Also invalidate landmark cache for current path since
        different tooth type = different cusp detection."""
        preset = self._WATERSHED_PRESETS.get(text)
        if preset:
            self.spn_n_cusps.setValue(preset["n_cusps"])
            self.spn_merge_depth.setValue(preset["merge_depth"])
            self.spn_merge_area.setValue(preset["merge_area"])

        # Invalidate cache entries for the current file
        # (any tooth_type variant)
        path = self._state.current_path
        if path:
            keys_to_remove = [k for k in self._landmark_cache if k[0] == path]
            for k in keys_to_remove:
                del self._landmark_cache[k]

    # ── Index change ──────────────────────────────────────

    def _on_index_changed(self, idx: int):
        # Only load when this panel is the active (visible) one, to avoid
        # multiple panels fighting over the viewer when the index changes.
        if not self.isVisible():
            return
        self._load_current_mesh()

    def showEvent(self, event):
        """Load current mesh when this panel becomes visible."""
        super().showEvent(event)
        if _ENGINE_OK and self._state.current_path is not None:
            self._load_current_mesh()

    def _load_current_mesh(self):
        path = self._state.current_path
        if path is None:
            return
        try:
            mesh = pv.read(path)
            if not isinstance(mesh, pv.PolyData):
                mesh = mesh.extract_surface()
            mesh = mesh.triangulate()
            self._state.set_mesh(mesh)
            self._current_result = None
            self._reset_labels()
            self._viewer.display_mesh(mesh)
            self._state.post_status(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            self._state.post_status(f"Load error: {e}")

    # ── Single-mesh compute (background thread) ───────────────

    def _on_compute(self):
        path = self._state.current_path
        if path is None:
            self._log("No mesh selected.")
            return
        # Guard: block re-entry while a computation is in flight
        if self._single_worker_thread is not None:
            self._log("Already computing — please wait.")
            return

        specimen_id = os.path.splitext(os.path.basename(path))[0]
        params = {
            "tooth_type":  self.cmb_tooth_type.currentText(),
            "n_cusps":     self.spn_n_cusps.value(),
            "merge_depth": self.spn_merge_depth.value(),
            "merge_area":  self.spn_merge_area.value(),
        }

        self.btn_compute.setEnabled(False)
        self.btn_compute.setText("Computing…")
        self._state.set_busy(True)

        worker = _SingleMeshWorker(path, params, self._landmark_cache, specimen_id)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.status.connect(self._log)
        worker.done.connect(self._on_single_done)
        worker.error.connect(self._on_single_error)
        # Stop the thread when the worker finishes (either path)
        worker.done.connect(thread.quit)
        worker.error.connect(thread.quit)
        # Clean up Python references after the thread has fully stopped
        thread.finished.connect(self._on_single_thread_finished)

        self._single_worker_thread = thread
        self._single_worker        = worker
        thread.start()

    def _on_single_thread_finished(self):
        """Called by QThread.finished — safe to release references here."""
        self._single_worker_thread = None
        self._single_worker        = None

    def _on_single_done(self, result):
        # Reset water level to 100% unless the user has locked it
        if not self.chk_fix_water.isChecked():
            self.sld_water_level.setValue(100)

        # Build PyVista PolyData for escape + retention channels (main thread only)
        try:
            if result.escape_paths:
                esc_pv, ret_pv = paths_to_polydata(
                    result.vertices, result.escape_paths,
                    path_types=result.path_types,
                )
                result.escape_channels_pv    = esc_pv
                result.retention_channels_pv = ret_pv
            else:
                result.escape_channels_pv    = None
                result.retention_channels_pv = None
        except Exception:
            result.escape_channels_pv    = None
            result.retention_channels_pv = None

        self._current_result = result
        self._update_labels(result)
        self._refresh_scene()

        self._log(
            f"FFEI={result.ffei_result.ffei:.4f}  "
            f"basins={result.watershed.n_basins}  "
            f"vol={result.ffei_result.basin_volume_mm3:.4f}mm³  "
            f"slope={result.metrics['slope_mean_deg']:.1f}°"
        )
        self.btn_compute.setEnabled(True)
        self.btn_compute.setText("Compute FFEI")
        self._state.set_busy(False)

    def _on_single_error(self, msg: str):
        self._log(f"ERROR: {msg}")
        self.btn_compute.setEnabled(True)
        self.btn_compute.setText("Compute FFEI")
        self._state.set_busy(False)

    # ── Scene rendering ───────────────────────────────────

    def _refresh_scene(self):
        plotter = self._viewer._plotter
        plotter.clear()

        style = self.cmb_style.currentText() if _ENGINE_OK else "Height"
        features = self.cmb_features.currentText() if _ENGINE_OK else "None"

        # Determine which mesh + data to show
        result = self._current_result
        if result is not None and result.pv_mesh is not None:
            mesh = result.pv_mesh.copy(deep=True)
            # Triangulate if needed
            if not mesh.is_all_triangles:
                mesh = mesh.triangulate()
        elif self._state.current_mesh is not None:
            mesh = self._state.current_mesh.copy(deep=True)
            if not mesh.is_all_triangles:
                mesh = mesh.triangulate()
        else:
            plotter.render()
            return

        # ── Colour by mode ────────────────────────────────
        if style == "Basins" and result is not None:
            basin_labels = result.watershed.basin_labels
            n_verts = mesh.n_points
            if len(basin_labels) == n_verts:
                # All labeled basins (0..n-1) get vivid colors.
                # Dry vertices (label == -1) get grey via dict fallback.
                central_id = result.ffei_result.central_basin_id
                colors = np.zeros((n_verts, 3), dtype=np.uint8)
                basin_color_map: dict = {}
                color_idx = 0
                for bid in range(result.watershed.n_basins):
                    if bid == central_id:
                        basin_color_map[bid] = [255, 255, 255]  # white = central
                    else:
                        c = _BASIN_COLORS[color_idx % len(_BASIN_COLORS)]
                        basin_color_map[bid] = _hex_to_rgb(c)
                        color_idx += 1
                for vi in range(n_verts):
                    colors[vi] = basin_color_map.get(int(basin_labels[vi]),
                                                     [100, 100, 100])
                mesh.point_data["basin_rgb"] = colors
                plotter.add_mesh(
                    mesh, scalars="basin_rgb", rgb=True,
                    smooth_shading=True, opacity=1.0,
                    show_edges=False, name="mesh",
                )
            else:
                mesh.point_data["elevation"] = mesh.points[:, 2]
                plotter.add_mesh(
                    mesh, scalars="elevation", cmap="gist_earth",
                    smooth_shading=True, name="mesh",
                )

        elif style == "Slope" and result is not None:
            n_verts = mesh.n_points
            slope_data = result.fields.slope
            if len(slope_data) == n_verts:
                mesh.point_data["slope"] = slope_data
                plotter.add_mesh(
                    mesh, scalars="slope", cmap="turbo",
                    smooth_shading=True, opacity=1.0,
                    show_edges=False, name="mesh",
                    show_scalar_bar=True,
                    scalar_bar_args={
                        "title": "Slope (°)", "title_font_size": 11,
                        "label_font_size": 10, "n_labels": 5,
                        "fmt": "%.1f", "width": 0.07, "height": 0.35,
                        "position_x": 0.92, "position_y": 0.05,
                        "vertical": True, "color": "white", "shadow": True,
                    },
                )
            else:
                self._add_height_mesh(plotter, mesh)

        elif style == "Water fill":
            # water_depth is computed directly by the priority-flood algorithm.
            # The slider scales it: 100% = actual fill depth, >100% = overfill.
            fill_frac = (self.sld_water_level.value() / 100.0
                         if hasattr(self, "sld_water_level") else 1.0)
            water_depth = np.zeros(mesh.n_points)
            if result is not None and hasattr(result.watershed, 'water_depth'):
                wd = result.watershed.water_depth
                if len(wd) == mesh.n_points:
                    water_depth = wd * fill_frac

            max_depth = float(water_depth.max())
            if max_depth > 1e-8:
                min_depth = max_depth * 0.05   # always < max_depth
                mesh.point_data["water_depth"] = water_depth.astype(float)
                plotter.add_mesh(
                    mesh, scalars="water_depth", cmap="Blues",
                    clim=[min_depth, max_depth],
                    smooth_shading=True, opacity=1.0,
                    show_edges=False, name="mesh",
                    below_color="#B0B0B0",
                    show_scalar_bar=True,
                    scalar_bar_args={
                        "title": "Water depth (mm)", "title_font_size": 11,
                        "label_font_size": 10, "n_labels": 5,
                        "fmt": "%.2f", "width": 0.07, "height": 0.35,
                        "position_x": 0.92, "position_y": 0.05,
                        "vertical": True, "color": "white", "shadow": True,
                    },
                )
            else:
                self._add_height_mesh(plotter, mesh)

        elif style == "Flow accumulation":
            if result is not None and hasattr(result.watershed, 'flow_acc'):
                fa = result.watershed.flow_acc
                if len(fa) == mesh.n_points:
                    log_acc = np.log1p(fa.astype(float))
                    mesh.point_data["flow_acc"] = log_acc
                    plotter.add_mesh(
                        mesh, scalars="flow_acc", cmap="YlOrRd",
                        smooth_shading=True, opacity=1.0,
                        show_edges=False, name="mesh",
                        show_scalar_bar=True,
                        scalar_bar_args={
                            "title": "Flow acc. (log1p)", "title_font_size": 11,
                            "label_font_size": 10, "n_labels": 4,
                            "fmt": "%.1f", "width": 0.07, "height": 0.30,
                            "position_x": 0.92, "position_y": 0.06,
                            "vertical": True, "color": "white", "shadow": True,
                        },
                    )
                else:
                    self._add_height_mesh(plotter, mesh)
            else:
                self._add_height_mesh(plotter, mesh)

        elif style == "Shaded":
            plotter.add_mesh(
                mesh, color="#D8C8B0",
                smooth_shading=True, specular=0.3,
                ambient=0.25, diffuse=0.8,
                opacity=1.0, show_edges=False, name="mesh",
            )

        else:  # "Height" or default
            self._add_height_mesh(plotter, mesh)

        # ── Feature overlays ──────────────────────────────
        if result is not None and features != "None":
            show_cusps    = features in ("Cusps", "Both", "All")
            show_foveae   = features in ("Foveae", "Both", "All")
            show_channels = features in ("Escape channels", "All")

            if show_cusps and len(result.maxima.indices) > 0:
                pts = pv.PolyData(result.maxima.positions)
                plotter.add_mesh(
                    pts, color="#FF6B35",
                    point_size=14, render_points_as_spheres=True,
                    name="cusps",
                )
                plotter.add_point_labels(
                    result.maxima.positions,
                    [f"C{i}" for i in range(len(result.maxima.indices))],
                    font_size=9, text_color="#FF6B35",
                    shape=None, name="cusp_labels",
                )

            if show_foveae and len(result.minima.indices) > 0:
                pts = pv.PolyData(result.minima.positions)
                plotter.add_mesh(
                    pts, color="#4ECDC4",
                    point_size=12, render_points_as_spheres=True,
                    name="foveae",
                )
                plotter.add_point_labels(
                    result.minima.positions,
                    [f"F{i}" for i in range(len(result.minima.indices))],
                    font_size=9, text_color="#4ECDC4",
                    shape=None, name="fovea_labels",
                )

            # Central fovea marker
            if result.ffei_result is not None:
                cid = result.ffei_result.central_basin_id
                if cid < len(result.watershed.sink_indices):
                    sink_idx = result.watershed.sink_indices[cid]
                    sink_pos = result.vertices[sink_idx].reshape(1, 3)
                    plotter.add_mesh(
                        pv.PolyData(sink_pos), color="#FFE66D",
                        point_size=18, render_points_as_spheres=True,
                        name="central_fovea",
                    )
                    plotter.add_point_labels(
                        sink_pos, ["Central"],
                        font_size=10, text_color="#FFE66D",
                        shape=None, name="central_label",
                    )

            # Escape channels (boundary-reaching) — taronja
            # Retention channels (interior sink) — blau clar
            if show_channels:
                try:
                    esc = result.escape_channels_pv
                    if esc is not None and esc.n_points > 0:
                        plotter.add_mesh(
                            esc, color="#FF4500",
                            line_width=4, render_lines_as_tubes=True,
                            opacity=0.9, name="escape_channels",
                        )
                except Exception:
                    pass
                try:
                    ret = result.retention_channels_pv
                    if ret is not None and ret.n_points > 0:
                        plotter.add_mesh(
                            ret, color="#4FC3F7",
                            line_width=3, render_lines_as_tubes=True,
                            opacity=0.7, name="retention_channels",
                        )
                except Exception:
                    pass

        plotter.enable_eye_dome_lighting()
        plotter.view_xy()
        plotter.reset_camera()
        plotter.render()
        QTimer.singleShot(50, self._post_render)

    @staticmethod
    def _add_height_mesh(plotter, mesh):
        mesh.point_data["elevation"] = mesh.points[:, 2]
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
                "vertical": True, "color": "white", "shadow": True,
            },
        )

    def _post_render(self):
        try:
            self._viewer._plotter.show_axes()
            self._viewer._plotter.render()
        except Exception:
            pass

    # ── Labels ────────────────────────────────────────────

    def _update_labels(self, r: FFEIAnalysisResult):
        m = r.metrics
        self.lbl_ffei_big.setText(f"{m['ffei']:.3f}")
        self.lbl_ffei.setText(f"{m['ffei']:.4f}")
        self.lbl_basins.setText(str(m['n_basins']))
        self.lbl_retention_n.setText(str(m.get('n_retention_basins', '—')))
        self.lbl_escape_n.setText(str(m.get('n_escape_basins', '—')))
        self.lbl_cusps.setText(str(m['n_cusps']))
        self.lbl_deps.setText(str(m['n_functional_depressions']))
        self.lbl_ret_area.setText(f"{m.get('retention_area', 0):.4f}")
        self.lbl_esc_area.setText(f"{m.get('escape_area', 0):.4f}")
        self.lbl_slope.setText(f"{m['slope_mean_deg']:.2f}")
        self.lbl_slope_max.setText(f"{m['slope_max_deg']:.2f}")
        self.lbl_slope_sd.setText(f"{m['slope_sd_deg']:.2f}")
        self.lbl_zrange.setText(f"{m['z_range_mm']:.3f}")
        self.lbl_hsd.setText(f"{m['height_sd']:.4f}")
        val = m.get('cusp_fovea_depth_rel')
        self.lbl_cf_depth.setText(f"{val:.4f}" if val is not None and not _is_nan(val) else "—")
        val = m.get('primary_fovea_depth_mm')
        self.lbl_fovea_d.setText(f"{val:.3f}" if val is not None and not _is_nan(val) else "—")
        self.lbl_gini.setText(f"{m['basin_area_gini']:.4f}")
        self.lbl_tarea.setText(f"{m['total_occlusal_area']:.4f}")
        val = m.get('basin_volume_mm3')
        self.lbl_basin_vol.setText(f"{val:.4f}" if val is not None and not _is_nan(val) else "—")
        val = m.get('retention_volume_mm3')
        self.lbl_ret_vol.setText(f"{val:.4f}" if val is not None and not _is_nan(val) else "—")
        val = m.get('central_basin_volume_mm3')
        self.lbl_central_vol.setText(f"{val:.4f}" if val is not None and not _is_nan(val) else "—")

    def _reset_labels(self):
        self.lbl_ffei_big.setText("—")
        for attr in (
            "lbl_ffei", "lbl_basins", "lbl_retention_n", "lbl_escape_n",
            "lbl_cusps", "lbl_deps", "lbl_ret_area", "lbl_esc_area",
            "lbl_slope", "lbl_slope_max", "lbl_slope_sd",
            "lbl_zrange", "lbl_hsd", "lbl_cf_depth", "lbl_fovea_d",
            "lbl_gini", "lbl_tarea", "lbl_basin_vol", "lbl_ret_vol",
            "lbl_central_vol",
        ):
            getattr(self, attr).setText("—")

    # ── Export ────────────────────────────────────────────

    def _on_export_csv(self):
        if self._current_result is None:
            self._log("No result to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "ffei_result.csv", "CSV (*.csv)"
        )
        if not path:
            return
        _write_single_csv(path, self._current_result.metrics)
        self._log(f"Saved: {path}")

    # ── Batch ─────────────────────────────────────────────

    def _browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select input folder")
        if folder:
            self.le_input.setText(folder)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self.le_output.setText(folder)

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
                if f.lower().endswith((".ply", ".obj", ".stl"))
            ])

        if not files:
            self._log("No mesh files found.")
            return

        output_dir = self.le_output.text().strip() or \
                     os.path.join(input_dir, "ffei_results")

        params = {
            "merge_depth": self.spn_merge_depth.value(),
            "merge_area":  self.spn_merge_area.value(),
        }

        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.btn_batch_run.setEnabled(False)
        self.btn_batch_abort.setEnabled(True)

        self._worker = _FFEIWorker(files, params, output_dir)
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_done)
        self._worker_thread.start()

    def _on_batch_abort(self):
        if self._worker:
            self._worker.abort()

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
        self._state.post_status("FFEI batch complete.")

    # ── Log ───────────────────────────────────────────────

    def _log(self, msg: str):
        self.log_box.appendPlainText(msg)
        self.log_box.ensureCursorVisible()


# ── Utility ───────────────────────────────────────────────

def _hex_to_rgb(hex_str: str) -> list:
    """Convert '#RRGGBB' to [R, G, B] as uint8."""
    h = hex_str.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]


def _is_nan(val) -> bool:
    try:
        return np.isnan(val)
    except (TypeError, ValueError):
        return False