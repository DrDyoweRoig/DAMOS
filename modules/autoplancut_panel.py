"""
DAMOS – AutoPlaneCut module panel  (v6)

Changes from v1:
  • Removed "Lowest central fovea" (redundant with Robust at percentile=0).
  • Fixed dangerous fallback low_points=top_points → progressive relaxation.
  • Added morphology presets (Human bunodont, Cercopithecid, Hominoid, Custom).
  • Added "Cusp-bounded minimum" method: detects the N cusps (via
    MorphologyEngine), builds their convex hull in XY, selects all
    mesh vertices inside, and uses min(Z) as z_base.
    Works for any morphology because the cusp polygon adapts to
    the actual occlusal surface geometry.
  • Fixed import to use relative package import (modules.autolmk_panel).
  • Added manual Z-plane adjustment: after Preview, the user can nudge
    the cut plane up/down with ± buttons (or edit the value directly)
    before committing the cut.
  • Added "Apply cut" button: saves the currently previewed mesh
    (with auto or manually adjusted z_cut) to the batch output folder.

Logic uses pure Python + NumPy for computation, PyVista for I/O and clipping.
"""

import os
import math

import numpy as np
import pyvista as pv

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QPushButton, QComboBox, QDoubleSpinBox,
    QCheckBox, QLineEdit, QPlainTextEdit, QProgressBar,
    QFileDialog, QLabel, QScrollArea, QFrame, QButtonGroup,
    QRadioButton, QSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal, QObject

from gui.app_state import AppState
from gui.viewer3d import Viewer3D

# Reuse the cusp detection engine from AutoLMK.
# Relative import because both panels live in the same package
# (modules.autolmk_panel / modules.autoplancut_panel).
from .autolmk_panel import MorphologyEngine


# ─────────────────────────────────────────────────────────────────────────────
# Morphology presets
# ─────────────────────────────────────────────────────────────────────────────

MORPHOLOGY_PRESETS = {
    "Human bunodont": {
        "occlusal_band_pct": 35.0,
        "low_band_pct":      25.0,
        "central_area_pct":  20.0,
        "percentile":         5.0,
        "offset_z":           0.1,
        "default_method":    "Robust low depression",
        "tooth_type":        "M1 lower – Sapiens (5 cusps)",
        "n_cusps":            4,
    },
    "Cercopithecid bilophodont": {
        "occlusal_band_pct": 50.0,
        "low_band_pct":      35.0,
        "central_area_pct":  40.0,
        "percentile":        10.0,
        "offset_z":           0.1,
        "default_method":    "Cusp-bounded minimum",
        "tooth_type":        "Bilophodont (4 cusps)",
        "n_cusps":            4,
    },
    "Hominoid": {
        "occlusal_band_pct": 45.0,
        "low_band_pct":      30.0,
        "central_area_pct":  30.0,
        "percentile":         5.0,
        "offset_z":           0.15,
        "default_method":    "Cusp-bounded minimum",
        "tooth_type":        "Auto (generic)",
        "n_cusps":            4,
    },
    "Custom": None,
}

# Tooth types available for cusp detection (from MorphologyEngine)
TOOTH_TYPES = [
    "Auto (generic)",
    "M1 lower – Sapiens (5 cusps)",
    "M1 upper – Sapiens (4 cusps)",
    "M2 lower – Sapiens (4 cusps)",
    "M2 upper – Sapiens (4 cusps)",
    "Bilophodont (4 cusps)",
    "Pm – Sapiens (2 cusps)",
]


# ─────────────────────────────────────────────────────────────────────────────
# Logic
# ─────────────────────────────────────────────────────────────────────────────

class AutoPlaneCutLogic:

    # ── Helpers ────────────────────────────────────────────

    @staticmethod
    def _median(values):
        if not values:
            raise ValueError("Cannot compute median of empty list.")
        s = sorted(values)
        m = len(s) // 2
        if len(s) % 2 == 0:
            return 0.5 * (s[m - 1] + s[m])
        return s[m]

    @staticmethod
    def _percentile(values, percentile):
        if not values:
            raise ValueError("Cannot compute percentile of empty list.")
        if percentile <= 0:
            return min(values)
        if percentile >= 100:
            return max(values)
        s = sorted(values)
        k = (len(s) - 1) * (percentile / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return s[int(k)]
        return s[int(f)] * (c - k) + s[int(c)] * (k - f)

    # ── Progressive fallback for low-point selection ──────

    def _select_low_points(self, top_points, low_band_pct, min_count=5):
        """Return low-Z candidates within the occlusal band.

        If the initial *low_band_pct* yields fewer than *min_count*
        points, the band is relaxed by +10 pp at a time up to 100 %.
        A warning string is returned as second value (empty if OK).
        """
        top_z = [p[2] for p in top_points]
        band = low_band_pct
        warning = ""

        while band <= 100.0:
            z_thresh = self._percentile(top_z, band)
            low = [p for p in top_points if p[2] <= z_thresh]
            if len(low) >= min_count:
                if band != low_band_pct:
                    warning = (
                        f"low_band relaxed {low_band_pct:.0f}→{band:.0f}% "
                        f"to reach {len(low)} pts"
                    )
                return low, z_thresh, warning
            band += 10.0

        warning = (
            f"WARNING: could not find ≥{min_count} low points even at "
            f"100 % band; using all {len(top_points)} occlusal-band points"
        )
        z_thresh = self._percentile(top_z, 100.0)
        return list(top_points), z_thresh, warning

    # ── Point-in-polygon (XY) via cross product ──────────

    @staticmethod
    def _points_in_convex_hull_xy(hull_xy: np.ndarray,
                                   query_xy: np.ndarray) -> np.ndarray:
        """Test which query points lie inside a convex hull defined in XY.

        hull_xy : (H, 2) — ordered vertices of the convex polygon
        query_xy: (N, 2) — points to test

        Returns boolean mask of shape (N,).

        Uses cross-product winding: a point is inside a convex polygon
        if it is on the same side of every edge.
        """
        n_hull = len(hull_xy)
        if n_hull < 3:
            return np.zeros(len(query_xy), dtype=bool)

        inside = np.ones(len(query_xy), dtype=bool)
        for i in range(n_hull):
            a = hull_xy[i]
            b = hull_xy[(i + 1) % n_hull]
            # Cross product of edge AB × AP for all query points P
            cross = (b[0] - a[0]) * (query_xy[:, 1] - a[1]) \
                  - (b[1] - a[1]) * (query_xy[:, 0] - a[0])
            inside &= (cross >= 0)

        return inside

    # ── Cusp-bounded minimum ──────────────────────────────

    def compute_cusp_bounded_cut_z(
        self,
        mesh: pv.PolyData,
        tooth_type: str = "Auto (generic)",
        n_cusps: int = 4,
        offset_z: float = 0.1,
        **_,
    ):
        """Detect cusps, build XY convex hull, find min(Z) inside.

        Pipeline:
          1. Detect N cusps using MorphologyEngine.
          2. Compute convex hull of cusp positions in XY.
          3. Select all mesh vertices whose XY falls inside the hull.
          4. z_base = min(Z) of those vertices.
          5. z_cut = z_base − offset.
        """
        from scipy.spatial import ConvexHull

        points = np.asarray(mesh.points, dtype=np.float64)

        # 1. Detect cusps
        engine = MorphologyEngine()
        engine.mesh = mesh
        engine.points = points

        params = engine.get_detection_parameters(tooth_type, n_cusps)
        cusps = engine.detect_cusps(
            points,
            n_cusps=params["n_cusps"],
            min_distance=params["min_distance"],
            alpha=params["alpha"],
        )
        cusps = engine.refine_cusps_to_local_maxima(
            points, cusps, refine_radius=params["refine_radius"]
        )
        cusps = engine.order_points_consistently(cusps, tooth_type=tooth_type)
        n_detected = len(cusps)

        # 2. Convex hull in XY of cusp tips
        cusp_xy = cusps[:, :2]
        if len(cusp_xy) < 3:
            # With 2 cusps (premolars): create a thin buffer around the line
            # by expanding perpendicular to the cusp–cusp axis
            c0, c1 = cusp_xy[0], cusp_xy[1]
            direction = c1 - c0
            perp = np.array([-direction[1], direction[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-12)
            buffer = 1.5  # mm lateral expansion
            hull_verts = np.array([
                c0 + perp * buffer,
                c0 - perp * buffer,
                c1 - perp * buffer,
                c1 + perp * buffer,
            ])
        else:
            hull = ConvexHull(cusp_xy)
            hull_verts = cusp_xy[hull.vertices]

        # Ensure CCW ordering for the cross-product test
        area = 0.0
        nv = len(hull_verts)
        for i in range(nv):
            j = (i + 1) % nv
            area += hull_verts[i, 0] * hull_verts[j, 1]
            area -= hull_verts[j, 0] * hull_verts[i, 1]
        if area < 0:
            hull_verts = hull_verts[::-1]

        # 3. Select vertices inside hull
        query_xy = points[:, :2]
        inside = self._points_in_convex_hull_xy(hull_verts, query_xy)
        n_inside = int(inside.sum())

        if n_inside < 3:
            raise ValueError(
                f"Only {n_inside} mesh vertices inside the cusp polygon. "
                f"Check cusp detection or mesh density."
            )

        # 4. Min Z inside polygon
        z_inside = points[inside, 2]
        z_base = float(z_inside.min())
        z_cut  = z_base - offset_z

        z_all = points[:, 2]
        debug = {
            "zMin":          float(z_all.min()),
            "zMax":          float(z_all.max()),
            "height":        float(z_all.max() - z_all.min()),
            "zThreshTop":    float("nan"),
            "zThreshLow":    float("nan"),
            "topCount":      n_inside,
            "lowCount":      0,
            "focusCount":    n_detected,
            "zBase":         z_base,
            "zCut":          z_cut,
            "method_note":   f"{n_detected} cusps detected, "
                             f"{n_inside} pts inside polygon",
            "cusps":         cusps,
            "hull_verts":    hull_verts,
        }
        return z_cut, debug

    # ── Z-band methods (Absolute lowest + Robust) ─────────

    def compute_occlusal_cut_z(
        self,
        points: np.ndarray,
        occlusal_band_pct: float = 35.0,
        low_band_pct: float = 25.0,
        central_area_pct: float = 20.0,
        method: str = "Robust low depression",
        percentile: float = 5.0,
        offset_z: float = 0.1,
        **_,
    ):
        coords = [tuple(p) for p in points]
        z_vals = [p[2] for p in coords]

        z_min = min(z_vals)
        z_max = max(z_vals)
        height = z_max - z_min

        if height <= 0:
            raise ValueError("Invalid Z range (flat mesh?).")

        # 1) Top occlusal band
        band_frac    = occlusal_band_pct / 100.0
        z_thresh_top = z_max - height * band_frac
        top_points   = [p for p in coords if p[2] >= z_thresh_top]

        if len(top_points) < 10:
            raise ValueError("Too few points in occlusal band.")

        # 2) Low-Z candidates — progressive relaxation
        low_points, z_thresh_low, relax_warning = self._select_low_points(
            top_points, low_band_pct
        )

        z_base      = None
        focus_count = 0
        method_note = relax_warning

        if method == "Absolute lowest fovea":
            z_base      = min(p[2] for p in low_points)
            focus_count = len(low_points)

        elif method == "Robust low depression":
            cx = self._median([p[0] for p in low_points])
            cy = self._median([p[1] for p in low_points])
            dist_data = [
                (p, math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2))
                for p in low_points
            ]
            distances = [d for _, d in dist_data]
            d_thresh  = self._percentile(distances, central_area_pct)
            focus     = [p for p, d in dist_data if d <= d_thresh]
            if len(focus) < 5:
                focus = low_points
            focus_z     = [p[2] for p in focus]
            z_base      = self._percentile(focus_z, percentile)
            focus_count = len(focus)

        else:
            raise ValueError(f"Unknown method: {method}")

        z_cut = z_base - offset_z

        debug = {
            "zMin":        z_min,
            "zMax":        z_max,
            "height":      height,
            "zThreshTop":  z_thresh_top,
            "zThreshLow":  z_thresh_low,
            "topCount":    len(top_points),
            "lowCount":    len(low_points),
            "focusCount":  focus_count,
            "zBase":       z_base,
            "zCut":        z_cut,
            "method_note": method_note,
        }
        return z_cut, debug

    # ── Clipping ──────────────────────────────────────────

    def clip_keep_above(self, mesh: pv.PolyData, z_cut: float) -> pv.PolyData:
        bounds = mesh.bounds

        if z_cut <= bounds[4]:
            raise ValueError(
                f"z_cut={z_cut:.4f} is at or below mesh Z minimum={bounds[4]:.4f}. "
                "Nothing would be cut – try increasing occlusal band % or Z offset."
            )
        if z_cut >= bounds[5]:
            raise ValueError(
                f"z_cut={z_cut:.4f} is at or above mesh Z maximum={bounds[5]:.4f}. "
                "Everything would be removed – try reducing occlusal band %."
            )

        clipped = mesh.clip(
            normal=(0.0, 0.0, 1.0),
            origin=(0.0, 0.0, z_cut),
            invert=False,
        )

        if clipped.n_points == 0:
            raise ValueError(
                f"Clipping produced empty mesh at z_cut={z_cut:.4f}. "
                "Check mesh orientation (+Z must face occlusal)."
            )

        return clipped.clean()

    def keep_largest(self, mesh: pv.PolyData) -> pv.PolyData:
        result = mesh.extract_largest()
        return result.clean() if result.n_points > 0 else mesh


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

class _CutWorker(QObject):
    progress = Signal(int, int, str)
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
        logic = AutoPlaneCutLogic()
        n     = len(self._files)
        os.makedirs(self._output_dir, exist_ok=True)

        method = self._params.get("method", "Robust low depression")
        is_cusp = (method == "Cusp-bounded minimum")

        for i, path in enumerate(self._files, 1):
            if self._abort:
                self.progress.emit(i, n, "Aborted.")
                break

            fname = os.path.basename(path)
            try:
                mesh = pv.read(path).triangulate().clean()

                if is_cusp:
                    z_cut, debug = logic.compute_cusp_bounded_cut_z(
                        mesh, **self._params
                    )
                else:
                    z_cut, debug = logic.compute_occlusal_cut_z(
                        mesh.points, **self._params
                    )

                cut = logic.clip_keep_above(mesh, z_cut)

                if self._params.get("keep_largest", True):
                    cut = logic.keep_largest(cut)

                suffix   = self._params.get("suffix", "_cut")
                out_name = os.path.splitext(fname)[0] + suffix + ".ply"
                out_path = os.path.join(self._output_dir, out_name)
                cut.save(out_path)

                note = debug.get("method_note", "")
                self.progress.emit(
                    i, n,
                    f"OK  {fname}  "
                    f"zCut={z_cut:.4f}  zBase={debug['zBase']:.4f}  "
                    f"focus={debug['focusCount']}cusps"
                    + (f"  [{note}]" if note else "")
                )

            except Exception as e:
                self.progress.emit(i, n, f"FAIL  {fname}: {e}")

        self.finished.emit()


# ─────────────────────────────────────────────────────────────────────────────
# Panel
# ─────────────────────────────────────────────────────────────────────────────

class AutoPlaneCutPanel(QWidget):

    def __init__(self, state: AppState, viewer: Viewer3D, parent=None):
        super().__init__(parent)
        self._state  = state
        self._viewer = viewer
        self._logic  = AutoPlaneCutLogic()
        self._worker_thread: QThread | None = None
        self._worker: _CutWorker | None = None
        self._preset_applying = False

        # State for manual Z-plane adjustment
        self._preview_mesh: pv.PolyData | None = None   # uncut mesh used in preview
        self._detected_z_cut: float | None   = None     # z_cut from last detection
        self._manual_z_cut: float | None     = None     # user-adjusted z_cut

        self._build_ui()
        self._connect_signals()

    # ── UI ─────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner  = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── Input ─────────────────────────────────────────
        grp_in = QGroupBox("Input")
        f_in   = QFormLayout(grp_in)

        self.cmb_format = QComboBox()
        self.cmb_format.addItems(["PLY", "OBJ"])
        f_in.addRow("Input format:", self.cmb_format)

        row_in = QHBoxLayout()
        self.le_input = QLineEdit()
        self.le_input.setPlaceholderText("Input folder path…")
        self.le_input.setReadOnly(True)
        btn_browse = QPushButton("Browse…")
        btn_browse.setObjectName("NavButton")
        btn_browse.clicked.connect(self._browse_input)
        row_in.addWidget(self.le_input)
        row_in.addWidget(btn_browse)
        f_in.addRow("Input folder:", row_in)

        layout.addWidget(grp_in)

        # ── Morphology Preset ─────────────────────────────
        grp_morph = QGroupBox("Morphology Preset")
        f_morph   = QFormLayout(grp_morph)

        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(list(MORPHOLOGY_PRESETS.keys()))
        self.cmb_preset.setCurrentText("Human bunodont")
        f_morph.addRow("Tooth type:", self.cmb_preset)

        self.lbl_preset_info = QLabel("")
        self.lbl_preset_info.setWordWrap(True)
        self.lbl_preset_info.setStyleSheet("color: #7A9AB8; font-size: 11px;")
        f_morph.addRow(self.lbl_preset_info)

        layout.addWidget(grp_morph)

        # ── Cut Detection ─────────────────────────────────
        grp_cut = QGroupBox("Cut Detection")
        f_cut   = QFormLayout(grp_cut)

        self.spn_occ = self._dspin(1, 100, 1, 35, " %")
        f_cut.addRow("Occlusal band:", self.spn_occ)

        self.spn_low = self._dspin(1, 100, 1, 25, " %")
        f_cut.addRow("Low-Z candidate band:", self.spn_low)

        self.spn_cent = self._dspin(1, 100, 1, 20, " %")
        f_cut.addRow("Central area %:", self.spn_cent)

        # Methods: 3 radio buttons
        self.rb_abs = QRadioButton("Absolute lowest fovea")
        self.rb_rob = QRadioButton("Robust low depression")
        self.rb_csp = QRadioButton("Cusp-bounded minimum")
        self.rb_rob.setChecked(True)
        self._method_grp = QButtonGroup(self)
        for rb in (self.rb_abs, self.rb_rob, self.rb_csp):
            self._method_grp.addButton(rb)
        mw = QWidget()
        ml = QVBoxLayout(mw)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(2)
        for rb in (self.rb_abs, self.rb_rob, self.rb_csp):
            ml.addWidget(rb)
        f_cut.addRow("Method:", mw)

        self.spn_pct = self._dspin(0, 50, 1, 5, " %")
        f_cut.addRow("Percentile (robust):", self.spn_pct)

        self.spn_off = self._dspin(0, 1000, 0.01, 0.1)
        self.spn_off.setDecimals(4)
        f_cut.addRow("Z offset (mm):", self.spn_off)

        # ── Cusp-bounded controls ─────────────────────────
        self.cmb_tooth = QComboBox()
        self.cmb_tooth.addItems(TOOTH_TYPES)
        self.cmb_tooth.setCurrentText("Auto (generic)")
        f_cut.addRow("Tooth type (cusps):", self.cmb_tooth)

        self.spn_ncusps = QSpinBox()
        self.spn_ncusps.setMinimum(2)
        self.spn_ncusps.setMaximum(8)
        self.spn_ncusps.setValue(4)
        f_cut.addRow("Number of cusps:", self.spn_ncusps)

        # Link radio → enable/disable
        for rb in (self.rb_abs, self.rb_rob, self.rb_csp):
            rb.toggled.connect(self._update_method_widgets)
        self._update_method_widgets()

        layout.addWidget(grp_cut)

        # ── Preview ───────────────────────────────────────
        grp_prev = QGroupBox("Preview (current mesh)")
        f_prev   = QFormLayout(grp_prev)
        self.btn_preview = QPushButton("Preview cut on current mesh")
        self.btn_preview.setObjectName("PrimaryButton")
        f_prev.addRow(self.btn_preview)
        self.lbl_zcut = QLabel("—")
        self.lbl_zcut.setObjectName("ValueLabel")
        f_prev.addRow("Detected z_cut:", self.lbl_zcut)
        self.lbl_debug = QLabel("")
        self.lbl_debug.setWordWrap(True)
        self.lbl_debug.setStyleSheet("color: #7A9AB8; font-size: 11px;")
        f_prev.addRow(self.lbl_debug)

        # ── Manual Z-plane adjustment ─────────────────────
        self.spn_manual_z = QDoubleSpinBox()
        self.spn_manual_z.setDecimals(4)
        self.spn_manual_z.setRange(-1e6, 1e6)
        self.spn_manual_z.setSingleStep(0.05)
        self.spn_manual_z.setEnabled(False)
        self.spn_manual_z.setToolTip(
            "Manually adjust the cut plane Z position. "
            "Use buttons or type a value directly."
        )

        self.spn_manual_step = QDoubleSpinBox()
        self.spn_manual_step.setDecimals(3)
        self.spn_manual_step.setRange(0.001, 10.0)
        self.spn_manual_step.setValue(0.1)
        self.spn_manual_step.setSuffix(" mm")
        self.spn_manual_step.setToolTip("Step size for ± nudge buttons.")

        self.btn_z_down = QPushButton("▼ lower")
        self.btn_z_down.setObjectName("NavButton")
        self.btn_z_down.setEnabled(False)
        self.btn_z_down.setToolTip("Move cut plane DOWN by step (removes less).")

        self.btn_z_up = QPushButton("▲ raise")
        self.btn_z_up.setObjectName("NavButton")
        self.btn_z_up.setEnabled(False)
        self.btn_z_up.setToolTip("Move cut plane UP by step (removes more).")

        self.btn_z_reset = QPushButton("Reset")
        self.btn_z_reset.setObjectName("NavButton")
        self.btn_z_reset.setEnabled(False)
        self.btn_z_reset.setToolTip("Reset to auto-detected z_cut.")

        row_nudge = QHBoxLayout()
        row_nudge.setSpacing(4)
        row_nudge.addWidget(self.btn_z_down)
        row_nudge.addWidget(self.btn_z_up)
        row_nudge.addWidget(self.btn_z_reset)
        f_prev.addRow("Manual z_cut:", self.spn_manual_z)
        f_prev.addRow("Step:",         self.spn_manual_step)
        f_prev.addRow("Adjust:",       row_nudge)

        # ── Apply cut (save current previewed mesh to output folder) ──
        self.btn_apply = QPushButton("Apply cut (save to output folder)")
        self.btn_apply.setObjectName("PrimaryButton")
        self.btn_apply.setEnabled(False)
        self.btn_apply.setToolTip(
            "Save the currently previewed mesh (with auto or manually "
            "adjusted z_cut) to the batch output folder using the "
            "current suffix. Uses the output folder defined below; if "
            "empty, saves next to the input file in an 'output' subfolder."
        )
        f_prev.addRow(self.btn_apply)

        layout.addWidget(grp_prev)

        # ── Output ────────────────────────────────────────
        grp_out = QGroupBox("Batch Output")
        f_out   = QFormLayout(grp_out)

        row_out = QHBoxLayout()
        self.le_output = QLineEdit()
        self.le_output.setPlaceholderText("Output folder (default: input/output)")
        self.le_output.setReadOnly(True)
        btn_out = QPushButton("Browse…")
        btn_out.setObjectName("NavButton")
        btn_out.clicked.connect(self._browse_output)
        row_out.addWidget(self.le_output)
        row_out.addWidget(btn_out)
        f_out.addRow("Output folder:", row_out)

        self.le_suffix = QLineEdit("_cut")
        f_out.addRow("Output suffix:", self.le_suffix)

        self.chk_keep = QCheckBox()
        self.chk_keep.setChecked(True)
        f_out.addRow("Keep largest component:", self.chk_keep)

        layout.addWidget(grp_out)

        # ── Log ───────────────────────────────────────────
        grp_log = QGroupBox("Log")
        f_log   = QVBoxLayout(grp_log)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        f_log.addWidget(self.progress_bar)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(120)
        f_log.addWidget(self.log_box)
        layout.addWidget(grp_log)
        layout.addStretch(1)

        # ── Run / Abort ───────────────────────────────────
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

    def _dspin(self, mn, mx, step, val, suffix="") -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setMinimum(mn); s.setMaximum(mx)
        s.setSingleStep(step); s.setValue(val)
        if suffix:
            s.setSuffix(suffix)
        return s

    # ── Signals ───────────────────────────────────────────

    def _connect_signals(self):
        self.btn_preview.clicked.connect(self._on_preview)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_abort.clicked.connect(self._on_abort)
        self._state.current_index_changed.connect(self._on_index_changed)
        self.cmb_preset.currentTextChanged.connect(self._on_preset_changed)

        # Manual Z-plane adjustment
        self.btn_z_down.clicked.connect(self._on_z_down)
        self.btn_z_up.clicked.connect(self._on_z_up)
        self.btn_z_reset.clicked.connect(self._on_z_reset)
        self.spn_manual_z.valueChanged.connect(self._on_manual_z_changed)
        self.btn_apply.clicked.connect(self._on_apply_cut)

        # Any manual parameter change → switch preset to "Custom"
        for widget in (
            self.spn_occ, self.spn_low, self.spn_cent, self.spn_pct,
            self.spn_off, self.spn_ncusps,
        ):
            widget.valueChanged.connect(self._on_param_manual_change)
        self.cmb_tooth.currentTextChanged.connect(self._on_param_manual_change)

    def _update_method_widgets(self):
        is_abs = self.rb_abs.isChecked()
        is_rob = self.rb_rob.isChecked()
        is_csp = self.rb_csp.isChecked()

        # Z-band controls: only for Absolute / Robust
        self.spn_occ.setEnabled(is_abs or is_rob)
        self.spn_low.setEnabled(is_abs or is_rob)
        self.spn_cent.setEnabled(is_rob)
        self.spn_pct.setEnabled(is_rob)

        # Cusp-bounded controls
        self.cmb_tooth.setEnabled(is_csp)
        self.spn_ncusps.setEnabled(is_csp)

        # Offset always enabled

    def _on_preset_changed(self, name: str):
        preset = MORPHOLOGY_PRESETS.get(name)
        if preset is None:
            self.lbl_preset_info.setText("Custom parameters – adjust manually.")
            return

        self._preset_applying = True

        self.spn_occ.setValue(preset["occlusal_band_pct"])
        self.spn_low.setValue(preset["low_band_pct"])
        self.spn_cent.setValue(preset["central_area_pct"])
        self.spn_pct.setValue(preset["percentile"])
        self.spn_off.setValue(preset["offset_z"])
        self.cmb_tooth.setCurrentText(preset.get("tooth_type", "Auto (generic)"))
        self.spn_ncusps.setValue(preset.get("n_cusps", 4))

        # Set default method for this morphology
        dm = preset.get("default_method", "Robust low depression")
        if dm == "Absolute lowest fovea":
            self.rb_abs.setChecked(True)
        elif dm == "Cusp-bounded minimum":
            self.rb_csp.setChecked(True)
        else:
            self.rb_rob.setChecked(True)

        self._preset_applying = False

        descriptions = {
            "Human bunodont": (
                "Low-relief bunodont molars (Homo). Uses Z-band detection "
                "(Robust low depression)."
            ),
            "Cercopithecid bilophodont": (
                "Bilophodont molars with transverse crests (e.g. Macaca, "
                "Cercopithecus). Detects 4 cusps and finds the lowest "
                "point inside the cusp polygon."
            ),
            "Hominoid": (
                "High-cusped hominoid molars (Pan, Gorilla, Pongo). "
                "Detects 4 cusps and finds the lowest point inside "
                "the cusp polygon."
            ),
        }
        self.lbl_preset_info.setText(descriptions.get(name, ""))

    def _on_param_manual_change(self):
        if not self._preset_applying:
            self.cmb_preset.blockSignals(True)
            self.cmb_preset.setCurrentText("Custom")
            self.cmb_preset.blockSignals(False)
            self.lbl_preset_info.setText("Custom parameters – adjust manually.")

    # ── Helpers ───────────────────────────────────────────

    def _browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select input folder")
        if folder:
            self.le_input.setText(folder)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self.le_output.setText(folder)

    def _get_params(self) -> dict:
        method = (
            "Absolute lowest fovea"    if self.rb_abs.isChecked() else
            "Cusp-bounded minimum"     if self.rb_csp.isChecked() else
            "Robust low depression"
        )
        return {
            "occlusal_band_pct":  self.spn_occ.value(),
            "low_band_pct":       self.spn_low.value(),
            "central_area_pct":   self.spn_cent.value(),
            "method":             method,
            "percentile":         self.spn_pct.value(),
            "offset_z":           self.spn_off.value(),
            "tooth_type":         self.cmb_tooth.currentText(),
            "n_cusps":            self.spn_ncusps.value(),
            "keep_largest":       self.chk_keep.isChecked(),
            "suffix":             self.le_suffix.text() or "_cut",
        }

    # ── Handlers ──────────────────────────────────────────

    def _on_index_changed(self, idx: int):
        path = self._state.current_path
        if path is None:
            return
        try:
            mesh = pv.read(path).triangulate().clean()
            self._state.set_mesh(mesh)
            self._viewer.display_mesh(mesh)
            self.lbl_zcut.setText("—")
            self.lbl_debug.setText("")
            # Reset manual adjustment state (previous z_cut is no longer valid)
            self._preview_mesh   = None
            self._detected_z_cut = None
            self._manual_z_cut   = None
            self._enable_manual_controls(False)
            self._state.post_status(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            self._state.post_status(f"Load error: {e}")

    def _on_preview(self):
        mesh = self._state.current_mesh
        if mesh is None:
            self._log("No mesh loaded for preview.")
            return
        params = self._get_params()
        try:
            is_cusp = (params["method"] == "Cusp-bounded minimum")

            if is_cusp:
                z_cut, debug = self._logic.compute_cusp_bounded_cut_z(
                    mesh, **params
                )
            else:
                z_cut, debug = self._logic.compute_occlusal_cut_z(
                    mesh.points, **params
                )

            self.lbl_zcut.setText(f"{z_cut:.4f}")
            note = debug.get("method_note", "")

            if is_cusp:
                self.lbl_debug.setText(
                    f"zBase={debug['zBase']:.4f}  "
                    f"{note}"
                )
            else:
                self.lbl_debug.setText(
                    f"zBase={debug['zBase']:.4f}  "
                    f"top={debug['topCount']}pts  "
                    f"low={debug['lowCount']}pts  "
                    f"focus={debug['focusCount']}pts"
                    + (f"\n{note}" if note else "")
                )

            # Store state for manual adjustment
            self._preview_mesh    = mesh
            self._detected_z_cut  = z_cut
            self._manual_z_cut    = z_cut

            # Configure manual z_cut spinbox (bounded by mesh Z range)
            self._enable_manual_controls(True)
            bounds = mesh.bounds  # (xmin,xmax, ymin,ymax, zmin,zmax)
            self.spn_manual_z.blockSignals(True)
            self.spn_manual_z.setRange(bounds[4] + 1e-6, bounds[5] - 1e-6)
            self.spn_manual_z.setValue(z_cut)
            self.spn_manual_z.blockSignals(False)

            self._apply_and_show_cut(mesh, z_cut, params["keep_largest"])
            self._log(
                f"Preview OK  z_cut={z_cut:.4f}  "
                f"zBase={debug['zBase']:.4f}"
                + (f"  [{note}]" if note else "")
            )
        except Exception as e:
            self._log(f"Preview error: {e}")
            self._enable_manual_controls(False)

    # ── Manual Z-plane adjustment ─────────────────────────

    def _enable_manual_controls(self, enabled: bool):
        self.spn_manual_z.setEnabled(enabled)
        self.btn_z_down.setEnabled(enabled)
        self.btn_z_up.setEnabled(enabled)
        self.btn_z_reset.setEnabled(enabled)
        self.btn_apply.setEnabled(enabled)

    def _apply_and_show_cut(self, mesh: pv.PolyData, z_cut: float,
                            keep_largest: bool):
        """Clip and display — shared by preview and manual adjustments."""
        cut = self._logic.clip_keep_above(mesh, z_cut)
        if keep_largest:
            cut = self._logic.keep_largest(cut)
        self._viewer.display_mesh(cut)

    def _on_z_down(self):
        """Lower the cut plane → cut_z decreases → MORE mesh is kept."""
        if self._preview_mesh is None:
            return
        step = self.spn_manual_step.value()
        new_z = self.spn_manual_z.value() - step
        self.spn_manual_z.setValue(new_z)  # triggers _on_manual_z_changed

    def _on_z_up(self):
        """Raise the cut plane → cut_z increases → LESS mesh is kept."""
        if self._preview_mesh is None:
            return
        step = self.spn_manual_step.value()
        new_z = self.spn_manual_z.value() + step
        self.spn_manual_z.setValue(new_z)

    def _on_z_reset(self):
        if self._detected_z_cut is None or self._preview_mesh is None:
            return
        self.spn_manual_z.setValue(self._detected_z_cut)

    def _on_manual_z_changed(self, new_z: float):
        if self._preview_mesh is None:
            return
        self._manual_z_cut = new_z
        keep_largest = self.chk_keep.isChecked()
        try:
            self._apply_and_show_cut(self._preview_mesh, new_z, keep_largest)
            delta = new_z - (self._detected_z_cut or new_z)
            sign = "+" if delta >= 0 else ""
            self.lbl_zcut.setText(
                f"{new_z:.4f}  (auto: {self._detected_z_cut:.4f}, {sign}{delta:.4f})"
            )
        except Exception as e:
            self._log(f"Manual cut error: {e}")

    def _on_apply_cut(self):
        """Save the currently previewed mesh with the current z_cut
        (auto or manually adjusted) to the batch output folder.
        The viewer is left as-is (showing the cut mesh)."""
        if self._preview_mesh is None or self._manual_z_cut is None:
            self._log("Nothing to apply — run Preview first.")
            return

        current_path = self._state.current_path
        if current_path is None:
            self._log("Cannot apply: no current file path.")
            return

        # Resolve output folder (same logic as batch)
        output_dir = self.le_output.text().strip()
        if not output_dir:
            input_dir = os.path.dirname(current_path)
            output_dir = os.path.join(input_dir, "output")

        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            self._log(f"Apply cut error: cannot create output dir: {e}")
            return

        z_cut  = self._manual_z_cut
        keep   = self.chk_keep.isChecked()
        suffix = self.le_suffix.text() or "_cut"

        try:
            cut = self._logic.clip_keep_above(self._preview_mesh, z_cut)
            if keep:
                cut = self._logic.keep_largest(cut)

            fname    = os.path.basename(current_path)
            out_name = os.path.splitext(fname)[0] + suffix + ".ply"
            out_path = os.path.join(output_dir, out_name)
            cut.save(out_path)

            is_manual = abs(z_cut - (self._detected_z_cut or z_cut)) > 1e-6
            tag = "manual" if is_manual else "auto"
            self._log(
                f"Applied ({tag})  z_cut={z_cut:.4f}  "
                f"{cut.n_points} pts / {cut.n_cells} cells  →  {out_path}"
            )
            self._state.post_status(f"Saved: {out_name}")
        except Exception as e:
            self._log(f"Apply cut error: {e}")

    def _on_run(self):
        input_dir = self.le_input.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            self._log("Please select a valid input folder.")
            return

        ext   = f".{self.cmb_format.currentText().lower()}"
        files = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(ext)
        ])
        if not files:
            self._log(f"No {ext} files found in {input_dir}")
            return

        output_dir = self.le_output.text().strip() or \
                     os.path.join(input_dir, "output")
        params     = self._get_params()

        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_abort.setEnabled(True)

        self._worker        = _CutWorker(files, params, output_dir)
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker_thread.start()

    def _on_abort(self):
        if self._worker:
            self._worker.abort()

    def _on_progress(self, current: int, total: int, msg: str):
        self.progress_bar.setValue(int(current / total * 100))
        self._log(f"[{current}/{total}] {msg}")

    def _on_done(self):
        self.btn_run.setEnabled(True)
        self.btn_abort.setEnabled(False)
        self.progress_bar.setValue(100)
        self._log("Done.")
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread.wait()
        self._state.post_status("AutoPlaneCut batch complete.")

    def _log(self, msg: str):
        self.log_box.appendPlainText(msg)
        self.log_box.ensureCursorVisible()