"""
DAMOS – Morphology Engine
=========================
DetectionResult dataclass and MorphologyEngine: automatic detection of cusps,
intercuspal points, central point, and border semilandmarks on dental meshes.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyvista as pv


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    cusps: np.ndarray
    intercuspal: np.ndarray
    center: np.ndarray
    border_semilandmarks: np.ndarray
    border_curve: np.ndarray

    def copy(self) -> "DetectionResult":
        return DetectionResult(
            cusps=self.cusps.copy(),
            intercuspal=self.intercuspal.copy(),
            center=self.center.copy(),
            border_semilandmarks=self.border_semilandmarks.copy(),
            border_curve=self.border_curve.copy(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class MorphologyEngine:

    def __init__(self):
        self.mesh: Optional[pv.PolyData] = None
        self.points: Optional[np.ndarray] = None
        self._ordering_warnings: list = []

    def load_mesh(self, filepath: str) -> pv.PolyData:
        self.mesh = pv.read(filepath)
        self.points = self.mesh.points
        return self.mesh

    def set_mesh(self, mesh: pv.PolyData) -> None:
        """Assign an already-loaded mesh without touching the filesystem."""
        self.mesh = mesh
        self.points = mesh.points

    # ── Anatomical cusp names ─────────────────────────────

    CUSP_NAMES = {
        "M1 lower – Sapiens (5 cusps)": [
            "Protoconid", "Hypoconid", "Entoconid", "Metaconid", "Hypoconulid",
        ],
        "M1 upper – Sapiens (4 cusps)": [
            "Paracone", "Metacone", "Hypocone", "Protocone",
        ],
        "M2 lower – Sapiens (4 cusps)": [
            "Protoconid", "Hypoconid", "Entoconid", "Metaconid",
        ],
        "M2 upper – Sapiens (4 cusps)": [
            "Paracone", "Metacone", "Hypocone", "Protocone",
        ],
        "Bilophodont (4 cusps)": [
            "Protoconid", "Metaconid", "Hypoconid", "Entoconid",
        ],
        "Pm – Sapiens (2 cusps)": [
            "Buccal", "Lingual",
        ],
    }

    def get_cusp_names(self, tooth_type: str, n_cusps: int) -> list:
        if tooth_type in self.CUSP_NAMES:
            return self.CUSP_NAMES[tooth_type]
        return [f"cusp{i+1}" for i in range(n_cusps)]

    def get_detection_parameters(self, tooth_type: str, n_cusps: int) -> dict:
        if tooth_type == "Pm – Sapiens (2 cusps)":
            return {"n_cusps": 2, "min_distance": 6.0, "alpha": 0.25, "refine_radius": 2.2}
        elif tooth_type in ("M2 lower – Sapiens (4 cusps)",
                            "M2 upper – Sapiens (4 cusps)"):
            return {"n_cusps": 4, "min_distance": 5.0, "alpha": 0.30, "refine_radius": 1.8}
        elif tooth_type == "M1 lower – Sapiens (5 cusps)":
            return {"n_cusps": 5, "min_distance": 4.5, "alpha": 0.35, "refine_radius": 1.6}
        elif tooth_type == "M1 upper – Sapiens (4 cusps)":
            return {"n_cusps": 4, "min_distance": 4.5, "alpha": 0.35, "refine_radius": 1.6}
        elif tooth_type == "Bilophodont (4 cusps)":
            return {"n_cusps": 4, "min_distance": 3.0, "alpha": 0.20, "refine_radius": 1.5}
        else:
            return {"n_cusps": n_cusps, "min_distance": 4.5, "alpha": 0.35, "refine_radius": 1.8}

    def order_points_consistently(self, pts: np.ndarray,
                                   tooth_type: str = "Auto (generic)") -> np.ndarray:
        self._ordering_warnings = []

        if pts is None or len(pts) == 0:
            return pts
        pts = np.asarray(pts).copy()
        n = len(pts)

        if tooth_type == "Bilophodont (4 cusps)" and n >= 4:
            return self._order_bilophodont(pts)

        if tooth_type == "Pm – Sapiens (2 cusps)" and n >= 2:
            return self._order_premolar(pts)

        if tooth_type in ("M1 lower – Sapiens (5 cusps)",
                          "M1 upper – Sapiens (4 cusps)",
                          "M2 lower – Sapiens (4 cusps)",
                          "M2 upper – Sapiens (4 cusps)") and n >= 4:
            return self._order_bunodont(pts, tooth_type)

        self._ordering_warnings.append(
            "Using generic CCW ordering — no anatomical quadrant assignment.")
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        ccw = np.argsort(angles)
        ccw_pts = pts[ccw]
        start = int(np.argmax(ccw_pts[:, 2]))
        return np.vstack([ccw_pts[start:], ccw_pts[:start]])

    # ── Ordering helpers ──────────────────────────────────

    def _order_premolar(self, pts: np.ndarray) -> np.ndarray:
        order = np.argsort(pts[:, 1])[::-1]
        y_diff = abs(pts[order[0], 1] - pts[order[1], 1])
        y_range = pts[:, 1].max() - pts[:, 1].min()
        if y_range > 1e-6 and y_diff / y_range < 0.15:
            self._ordering_warnings.append(
                f"Premolar cusps poorly separated in Y "
                f"(ΔY={y_diff:.2f}, {y_diff/y_range*100:.0f}% of range). "
                f"Buccal/lingual assignment may be unreliable.")
        return pts[order[:2]]

    def _order_bunodont(self, pts: np.ndarray, tooth_type: str) -> np.ndarray:
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        rx = pts[:, 0] - cx
        ry = pts[:, 1] - cy

        mesio_buccal  = (rx <= 0) & (ry >= 0)
        disto_buccal  = (rx >  0) & (ry >= 0)
        disto_lingual = (rx >  0) & (ry <  0)
        mesio_lingual = (rx <= 0) & (ry <  0)

        quadrant_names = ["mesio-buccal", "disto-buccal",
                          "disto-lingual", "mesio-lingual"]
        quadrants = [mesio_buccal, disto_buccal, disto_lingual, mesio_lingual]
        ordered = []

        for qi, mask in enumerate(quadrants):
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                self._ordering_warnings.append(
                    f"⚠ No cusp found in {quadrant_names[qi]} quadrant. "
                    f"Check mesh orientation or tooth type selection.")
            elif len(idxs) == 1:
                ordered.append(pts[idxs[0]])
            elif len(idxs) > 1:
                z_sorted = np.argsort(pts[idxs, 2])[::-1]
                best = idxs[z_sorted[0]]
                ordered.append(pts[best])
                z_diff = pts[idxs[z_sorted[0]], 2] - pts[idxs[z_sorted[1]], 2]
                z_range = pts[:, 2].max() - pts[:, 2].min()
                if z_range > 1e-6 and z_diff / z_range < 0.05:
                    self._ordering_warnings.append(
                        f"⚠ Ambiguous {quadrant_names[qi]}: "
                        f"{len(idxs)} cusps in quadrant, top two differ by "
                        f"only {z_diff:.3f} mm ({z_diff/z_range*100:.1f}% of Z range). "
                        f"Consider manual verification.")

        for i, p in enumerate(pts):
            dx = abs(p[0] - cx)
            dy = abs(p[1] - cy)
            x_range = pts[:, 0].max() - pts[:, 0].min()
            y_range = pts[:, 1].max() - pts[:, 1].min()
            if x_range > 1e-6 and dx / x_range < 0.06:
                self._ordering_warnings.append(
                    f"⚠ Cusp {i+1} is near the mesio-distal midline "
                    f"(ΔX={dx:.3f}, {dx/x_range*100:.1f}% of range). "
                    f"Quadrant assignment may flip with small rotations.")
            if y_range > 1e-6 and dy / y_range < 0.06:
                self._ordering_warnings.append(
                    f"⚠ Cusp {i+1} is near the bucco-lingual midline "
                    f"(ΔY={dy:.3f}, {dy/y_range*100:.1f}% of range). "
                    f"Quadrant assignment may flip with small rotations.")

        if tooth_type == "M1 lower – Sapiens (5 cusps)" and len(pts) >= 5:
            assigned_set = set()
            for op in ordered:
                dists = np.linalg.norm(pts - op, axis=1)
                assigned_set.add(int(np.argmin(dists)))

            remaining = [i for i in range(len(pts)) if i not in assigned_set]
            if remaining:
                rem_pts = pts[remaining]
                hypo_idx = remaining[np.argmax(rem_pts[:, 0])]
                ordered.append(pts[hypo_idx])
            else:
                self._ordering_warnings.append(
                    "⚠ Could not identify hypoconulid — "
                    "all cusps consumed by 4-quadrant assignment.")

        if len(ordered) < 4:
            self._ordering_warnings.append(
                "⚠ Quadrant assignment failed — falling back to CCW ordering. "
                "Landmark homology is NOT guaranteed.")
            return self._order_fallback_ccw(pts)

        return np.array(ordered)

    def _order_bilophodont(self, pts: np.ndarray) -> np.ndarray:
        cx = np.mean(pts[:, 0])

        mesial_idxs = np.where(pts[:, 0] <= cx)[0]
        distal_idxs = np.where(pts[:, 0] > cx)[0]

        if len(mesial_idxs) < 2 or len(distal_idxs) < 2:
            self._ordering_warnings.append(
                f"⚠ Loph split unbalanced: {len(mesial_idxs)} mesial, "
                f"{len(distal_idxs)} distal. Using X-rank fallback.")
            sorted_x = np.argsort(pts[:, 0])
            mesial_idxs = sorted_x[:2]
            distal_idxs = sorted_x[2:4]

        mes_pts = pts[mesial_idxs]
        mes_order = np.argsort(mes_pts[:, 1])[::-1]
        mesio_buccal  = pts[mesial_idxs[mes_order[0]]]
        mesio_lingual = pts[mesial_idxs[mes_order[1]]]

        dis_pts = pts[distal_idxs]
        dis_order = np.argsort(dis_pts[:, 1])[::-1]
        disto_buccal  = pts[distal_idxs[dis_order[0]]]
        disto_lingual = pts[distal_idxs[dis_order[1]]]

        y_range = pts[:, 1].max() - pts[:, 1].min()
        for loph_name, bp, lp in [("mesial", mesio_buccal, mesio_lingual),
                                    ("distal", disto_buccal, disto_lingual)]:
            y_diff = abs(bp[1] - lp[1])
            if y_range > 1e-6 and y_diff / y_range < 0.12:
                self._ordering_warnings.append(
                    f"⚠ {loph_name.capitalize()} loph: buccal/lingual cusps "
                    f"poorly separated (ΔY={y_diff:.2f}, "
                    f"{y_diff/y_range*100:.0f}% of range). "
                    f"Assignment may be unreliable.")

        x_range = pts[:, 0].max() - pts[:, 0].min()
        loph_gap = abs(np.mean([mesio_buccal[0], mesio_lingual[0]]) -
                       np.mean([disto_buccal[0], disto_lingual[0]]))
        if x_range > 1e-6 and loph_gap / x_range < 0.15:
            self._ordering_warnings.append(
                f"⚠ Lophs poorly separated in X "
                f"(gap={loph_gap:.2f}, {loph_gap/x_range*100:.0f}% of range). "
                f"Mesial/distal assignment may be unreliable.")

        return np.array([mesio_buccal, mesio_lingual, disto_buccal, disto_lingual])

    def _order_fallback_ccw(self, pts: np.ndarray) -> np.ndarray:
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        ccw = np.argsort(angles)
        ccw_pts = pts[ccw]
        start = int(np.argmax(ccw_pts[:, 2]))
        return np.vstack([ccw_pts[start:], ccw_pts[:start]])

    # ── Cusp detection ────────────────────────────────────

    def detect_cusps(self, points, n_cusps=5, min_distance=4.5, alpha=0.35):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        cx, cy = np.mean(x), np.mean(y)
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r_norm = (r - r.min()) / (r.max() - r.min() + 1e-12)
        score = z - alpha * r_norm * (z.max() - z.min())
        r_limit = np.percentile(r, 95)
        valid_mask = r < r_limit
        valid_points = points[valid_mask]
        valid_score = score[valid_mask]
        order = np.argsort(valid_score)[::-1]
        sorted_points = valid_points[order]
        selected = []
        for p in sorted_points:
            if len(selected) == 0:
                selected.append(p)
            else:
                dists = [np.linalg.norm(p[:2] - q[:2]) for q in selected]
                if min(dists) > min_distance:
                    selected.append(p)
            if len(selected) >= n_cusps:
                break
        selected = np.array(selected)
        if len(selected) < n_cusps:
            raise ValueError(f"Detected {len(selected)} cusps, expected {n_cusps}.")
        return selected

    def refine_cusps_to_local_maxima(self, points, cusps, refine_radius=2.0, max_shift=1.2):
        refined = []
        for cusp in cusps:
            dxy = np.linalg.norm(points[:, :2] - cusp[:2], axis=1)
            local_idx = np.where(dxy <= refine_radius)[0]
            if len(local_idx) == 0:
                refined.append(cusp)
                continue
            local_pts = points[local_idx]
            candidate = local_pts[np.argmax(local_pts[:, 2])]
            if np.linalg.norm(candidate[:2] - cusp[:2]) <= max_shift and candidate[2] >= cusp[2]:
                refined.append(candidate)
            else:
                refined.append(cusp)
        return np.array(refined)

    # ── Intercuspal ───────────────────────────────────────

    def line_minimum_on_mesh(self, points, c1, c2, n_samples=120, t_min=0.20, t_max=0.80):
        sampled_indices = []
        for t in np.linspace(t_min, t_max, n_samples):
            xy = (1.0 - t) * c1[:2] + t * c2[:2]
            dists = np.linalg.norm(points[:, :2] - xy, axis=1)
            sampled_indices.append(np.argmin(dists))
        sampled_indices = list(dict.fromkeys(sampled_indices))
        sampled_pts = points[sampled_indices]
        return sampled_pts[np.argmin(sampled_pts[:, 2])]

    def detect_intercuspal_points(self, points, cusps, tooth_type="Auto (generic)"):
        n = len(cusps)
        pts = []
        if tooth_type == "Pm – Sapiens (2 cusps)" and n >= 2:
            pts.append(self.line_minimum_on_mesh(points, cusps[0], cusps[1]))
            return np.array(pts)
        elif tooth_type in ("M2 lower – Sapiens (4 cusps)",
                            "M2 upper – Sapiens (4 cusps)",
                            "M1 upper – Sapiens (4 cusps)") and n >= 4:
            for i, j in [(0,1),(1,2),(2,3),(3,0)]:
                pts.append(self.line_minimum_on_mesh(points, cusps[i], cusps[j]))
            return np.array(pts)
        elif tooth_type == "M1 lower – Sapiens (5 cusps)" and n >= 5:
            for i, j in [(0,1),(1,2),(2,3),(3,4),(4,0)]:
                pts.append(self.line_minimum_on_mesh(points, cusps[i], cusps[j]))
            return np.array(pts)
        elif tooth_type == "Bilophodont (4 cusps)" and n >= 4:
            for i, j in [(0,1),(1,3),(3,2),(2,0)]:
                pts.append(self.line_minimum_on_mesh(points, cusps[i], cusps[j]))
            return np.array(pts)
        for i in range(n):
            pts.append(self.line_minimum_on_mesh(points, cusps[i], cusps[(i+1)%n]))
        return np.array(pts)

    # ── Central point ──────────────────────────────────────

    def detect_central_point(self, points, cusps=None, border_points=None):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        ref_xy = np.mean(cusps[:, :2], axis=0) if cusps is not None and len(cusps) > 0 \
                 else np.array([np.mean(x), np.mean(y)])
        r = np.linalg.norm(points[:, :2] - ref_xy, axis=1)
        r_limit = np.percentile(r, 22)
        center_pts = points[r < r_limit]
        if len(center_pts) == 0:
            raise ValueError("No central points found.")
        z_border = np.median(border_points[:, 2]) if border_points is not None and len(border_points) > 0 \
                   else np.percentile(z, 5)
        z_range = z.max() - z.min()
        min_above = z_border + max(0.15, 0.03 * z_range)
        valid = center_pts[center_pts[:, 2] > min_above]
        if len(valid) == 0:
            valid = center_pts
        d = np.linalg.norm(valid[:, :2] - ref_xy, axis=1)
        zv = valid[:, 2]
        z_n = (zv - zv.min()) / (zv.max() - zv.min() + 1e-12)
        d_n = (d - d.min()) / (d.max() - d.min() + 1e-12)
        return valid[np.argmin(z_n + 0.6 * d_n)]

    # ── Border semilandmarks ───────────────────────────────

    def polygon_signed_area_xy(self, points):
        x, y = points[:, 0], points[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

    def order_points_closed_loop(self, points):
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        return points[np.argsort(angles)]

    def walk_edge_loop(self, loop_poly):
        pts = np.asarray(loop_poly.points)
        n = len(pts)
        if n < 3:
            return pts

        lines = loop_poly.lines
        adj = [[] for _ in range(n)]
        i = 0
        while i < len(lines):
            nv = lines[i]
            if nv == 2:
                a, b = int(lines[i + 1]), int(lines[i + 2])
                if b not in adj[a]:
                    adj[a].append(b)
                if a not in adj[b]:
                    adj[b].append(a)
            i += 1 + nv

        degrees = [len(a) for a in adj]
        if any(d == 0 for d in degrees):
            self._ordering_warnings.append(
                "⚠ Boundary loop has isolated vertices — "
                "semilandmark ordering may be incomplete.")
        non_manifold = sum(1 for d in degrees if d not in (0, 2))
        if non_manifold > 0:
            self._ordering_warnings.append(
                f"⚠ Boundary loop has {non_manifold} non-manifold vertices "
                f"(T-junctions or spurs). Traversal may skip branches.")

        ordered_idx = [0]
        visited = {0}
        current = 0
        prev = -1
        while True:
            next_v = -1
            for nb in adj[current]:
                if nb != prev and nb not in visited:
                    next_v = nb
                    break
            if next_v == -1:
                if 0 in adj[current] and len(ordered_idx) >= 3:
                    break
                break
            ordered_idx.append(next_v)
            visited.add(next_v)
            prev = current
            current = next_v
            if len(ordered_idx) >= n:
                break

        if len(ordered_idx) < n:
            self._ordering_warnings.append(
                f"⚠ Edge walk visited {len(ordered_idx)}/{n} boundary "
                f"vertices — contour may be fragmented.")

        return pts[ordered_idx]

    def ensure_clockwise(self, points):
        return points[::-1] if self.polygon_signed_area_xy(points) > 0 else points

    def rotate_curve_start(self, points, reference_point):
        dists = np.linalg.norm(points[:, :2] - reference_point[:2], axis=1)
        start = np.argmin(dists)
        return np.vstack([points[start:], points[:start]])

    def resample_closed_curve(self, points, n_samples=50):
        if len(points) < 3:
            raise ValueError("Not enough points to resample curve.")
        closed = np.vstack([points, points[0]])
        seg_lens = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg_lens)])
        total = cum[-1]
        if total == 0:
            raise ValueError("Curve length is zero.")
        targets = np.linspace(0, total, n_samples, endpoint=False)
        sampled = []
        seg_idx = 0
        for td in targets:
            while seg_idx < len(seg_lens) - 1 and cum[seg_idx + 1] < td:
                seg_idx += 1
            p0, p1 = closed[seg_idx], closed[seg_idx + 1]
            sl = seg_lens[seg_idx]
            if sl == 0:
                sampled.append(p0)
                continue
            t = (td - cum[seg_idx]) / sl
            sampled.append((1 - t) * p0 + t * p1)
        return np.array(sampled)

    def extract_lowest_boundary_loop(self, mesh, return_polydata=False):
        boundary = mesh.extract_feature_edges(
            boundary_edges=True, feature_edges=False,
            manifold_edges=False, non_manifold_edges=False
        )
        if boundary.n_points == 0 or boundary.n_cells == 0:
            pts = self._boundary_fallback_closed_mesh(mesh)
            if return_polydata:
                return self._points_to_loop_polydata(pts)
            return pts

        connected = boundary.connectivity()
        sub_loops = []
        z_means   = []

        if "RegionId" in connected.cell_data:
            region_ids = connected.cell_data["RegionId"]
            for rid in np.unique(region_ids):
                cell_ids = np.where(region_ids == rid)[0]
                sub = connected.extract_cells(cell_ids)
                if sub.n_points >= 3:
                    sub_loops.append(sub)
                    z_means.append(float(np.mean(sub.points[:, 2])))
        elif "RegionId" in connected.point_data:
            region_ids = connected.point_data["RegionId"]
            for rid in np.unique(region_ids):
                mask = (region_ids == rid)
                pts  = connected.points[mask]
                if len(pts) >= 3:
                    sub_loops.append(self._points_to_loop_polydata(pts))
                    z_means.append(float(np.mean(pts[:, 2])))
        else:
            pts = self._boundary_fallback_closed_mesh(mesh)
            if return_polydata:
                return self._points_to_loop_polydata(pts)
            return pts

        if not sub_loops:
            pts = self._boundary_fallback_closed_mesh(mesh)
            if return_polydata:
                return self._points_to_loop_polydata(pts)
            return pts

        lowest = sub_loops[int(np.argmin(z_means))]
        if not isinstance(lowest, pv.PolyData):
            lowest = lowest.extract_surface()

        if return_polydata:
            return lowest
        return np.asarray(lowest.points)

    @staticmethod
    def _points_to_loop_polydata(points):
        n = len(points)
        lines = np.empty(n * 3, dtype=np.int64)
        for k in range(n):
            lines[3 * k]     = 2
            lines[3 * k + 1] = k
            lines[3 * k + 2] = (k + 1) % n
        poly = pv.PolyData(np.asarray(points))
        poly.lines = lines
        return poly

    def _boundary_fallback_closed_mesh(self, mesh):
        from scipy.spatial import ConvexHull

        points = mesh.points
        z = points[:, 2]
        z_thresh = np.percentile(z, 15)
        low_pts = points[z <= z_thresh]

        if len(low_pts) < 3:
            raise ValueError("No boundary edge detected and fallback failed: "
                             "not enough low-Z points.")
        try:
            hull = ConvexHull(low_pts[:, :2])
            hull_pts = low_pts[hull.vertices]
        except Exception:
            raise ValueError("No boundary edge detected and convex hull "
                             "fallback failed.")

        for i in range(len(hull_pts)):
            dxy = np.linalg.norm(points[:, :2] - hull_pts[i, :2], axis=1)
            hull_pts[i, 2] = points[np.argmin(dxy), 2]

        return hull_pts

    def detect_semilandmarks_on_lower_border(self, mesh, reference_point, n_samples=50):
        loop_poly = self.extract_lowest_boundary_loop(mesh, return_polydata=True)
        ordered = self.walk_edge_loop(loop_poly)
        if len(ordered) < 3:
            ordered = self.order_points_closed_loop(np.asarray(loop_poly.points))
        ordered = self.ensure_clockwise(ordered)
        ordered = self.rotate_curve_start(ordered, reference_point)
        semis = self.resample_closed_curve(ordered, n_samples=n_samples)
        return semis, ordered

    # ── Full pipeline ──────────────────────────────────────

    def detect_all(self, n_cusps=5, tooth_type="Auto (generic)",
                   n_border_semilandmarks=50, min_distance=4.5, alpha=0.35) -> DetectionResult:
        if self.mesh is None or self.points is None:
            raise ValueError("No mesh loaded.")
        params = self.get_detection_parameters(tooth_type, n_cusps)
        eff_n     = params["n_cusps"]
        eff_dist  = params["min_distance"] if tooth_type != "Auto (generic)" else min_distance
        eff_alpha = params["alpha"]         if tooth_type != "Auto (generic)" else alpha
        refine_r  = params["refine_radius"]

        cusps = self.detect_cusps(self.points, n_cusps=eff_n,
                                   min_distance=eff_dist, alpha=eff_alpha)
        cusps = self.refine_cusps_to_local_maxima(self.points, cusps,
                                                   refine_radius=refine_r, max_shift=1.2)
        cusps = self.order_points_consistently(cusps, tooth_type=tooth_type)
        intercuspal = self.detect_intercuspal_points(self.points, cusps, tooth_type=tooth_type)
        border_semis, border_curve = self.detect_semilandmarks_on_lower_border(
            self.mesh, reference_point=cusps[0], n_samples=n_border_semilandmarks)
        center = self.detect_central_point(self.points, cusps=cusps, border_points=border_semis)

        return DetectionResult(cusps=cusps, intercuspal=intercuspal, center=center,
                               border_semilandmarks=border_semis, border_curve=border_curve)
