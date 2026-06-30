"""
Priority-Flood Watershed — Layer 3
=====================================

Implements the Zuccotti (1998) GIS approach on 3D triangular meshes
using the priority-flood algorithm (Wang & Liu 2006).

Steps
-----
1. Priority-flood fill — floods depressions from mesh boundary inward
   using a min-heap; gives physically correct fill heights and water
   depths without any classification heuristics.
2. Flow accumulation — steepest-descent counting per vertex.
3. BFS basin labelling — connected components of wet vertices.
4. FFEI + volume computation from filled basins.

FFEI = total wet (basin) area / total occlusal area
Primary physical metric: basin_volume_mm3 (total water retained in mm³)
"""

import heapq

import numpy as np
from typing import Optional, NamedTuple


class WatershedResult(NamedTuple):
    fill_height:    np.ndarray   # (N,) priority-flood fill Z (mesh units, >= z)
    water_depth:    np.ndarray   # (N,) fill_height - z  (>= 0)
    flow_acc:       np.ndarray   # (N,) upstream vertex count
    downstream:     np.ndarray   # (N,) steepest downhill neighbour (-1 = sink)
    basin_labels:   np.ndarray   # (N,) -1 = dry, 0..n-1 = basin index
    n_basins:       int
    sink_indices:   np.ndarray   # (n_basins,) deepest vertex per basin
    sink_positions: np.ndarray   # (n_basins, 3)
    face_labels:    np.ndarray   # (F,) majority-vote basin per face (-1 = dry)


class FFEIResult(NamedTuple):
    ffei:               float   # basin_area / total_area
    basin_volume_mm3:   float   # total water volume in mm³ (primary Zuccotti metric)
    basin_area_mm2:     float   # total wet surface area in mm²
    total_area:         float   # total occlusal area in mm²
    n_basins:           int
    basin_areas:        dict    # {basin_id: mm²}
    basin_volumes:      dict    # {basin_id: mm³}
    # Legacy aliases (backward compat with metrics.py and panel)
    retention_area:         float   # = basin_area_mm2
    escape_area:            float   # = total_area - basin_area_mm2
    retention_basin_ids:    list    # = list(range(n_basins))
    escape_basin_ids:       list    # = []
    central_basin_id:       int
    central_area:           float
    peripheral_area:        float   # = escape_area
    per_basin_ffei:         dict    # {basin_id: area/total}
    n_retention_basins:     int     # = n_basins
    n_escape_basins:        int     # = 0


class BasinVolumes(NamedTuple):
    per_basin_mm3:          dict    # {basin_id: mm³}
    total_retention_mm3:    float
    per_basin_spillover_mm: dict    # {basin_id: max fill_h = saddle Z}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_adjacency(faces: np.ndarray, n_vertices: int) -> list:
    adj = [[] for _ in range(n_vertices)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a].append(b); adj[a].append(c)
        adj[b].append(a); adj[b].append(c)
        adj[c].append(a); adj[c].append(b)
    return [list(set(row)) for row in adj]


def _face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _face_labels_from_verts(labels: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-face majority-vote label from vertex labels (-1 = dry)."""
    f0 = labels[faces[:, 0]]
    f1 = labels[faces[:, 1]]
    f2 = labels[faces[:, 2]]
    return np.where(f0 == f1, f0,
           np.where(f1 == f2, f1,
           np.where(f0 == f2, f0, f0)))


def _find_boundary_vertices(faces: np.ndarray, n_vertices: int,
                             vertices: Optional[np.ndarray] = None) -> set:
    """Open-boundary edges (shared by exactly one face).

    Falls back to the lowest Z-band for watertight / closed meshes.
    """
    edge_cnt: dict = {}
    for f in faces:
        for i in range(3):
            e = (min(int(f[i]), int(f[(i + 1) % 3])),
                 max(int(f[i]), int(f[(i + 1) % 3])))
            edge_cnt[e] = edge_cnt.get(e, 0) + 1
    bv: set = set()
    for (a, b), cnt in edge_cnt.items():
        if cnt == 1:
            bv.add(a)
            bv.add(b)
    if len(bv) == 0 and vertices is not None:
        z = vertices[:, 2]
        thr = float(np.percentile(z, 8))
        bv = set(int(v) for v in np.where(z <= thr)[0])
    return bv


# ── Step 1: Priority-flood ────────────────────────────────────────────────────

def priority_flood(z: np.ndarray, adj: list, boundary_verts: set):
    """Fill surface depressions from mesh boundary inward (Wang & Liu 2006).

    Each vertex gets a fill height >= its actual Z.  Water pools inside
    enclosed depressions up to the level of the lowest escape saddle.

    Parameters
    ----------
    z              : (N,) float — actual vertex Z in mesh units (mm)
    adj            : list[list[int]] — vertex adjacency
    boundary_verts : set — mesh boundary vertices (flood seeds)

    Returns
    -------
    fill_h      : (N,) float — flood-fill elevation
    water_depth : (N,) float — fill_h - z  (>= 0)
    """
    n = len(z)
    # Non-boundary vertices start at +inf so the first flood path always wins.
    fill_h = np.full(n, np.inf)
    closed = np.zeros(n, dtype=bool)

    pq = []
    for v in boundary_verts:
        fill_h[v] = float(z[v])
        heapq.heappush(pq, (float(z[v]), int(v)))

    while pq:
        fh, v = heapq.heappop(pq)
        if closed[v]:
            continue
        closed[v] = True
        for nb in adj[v]:
            if closed[nb]:
                continue
            # Water surface at nb must be at least z[nb] (can't go below terrain)
            # and at least as high as the incoming flood level fh.
            new_fh = float(z[nb]) if z[nb] > fh else fh
            if new_fh < fill_h[nb]:
                fill_h[nb] = new_fh
                heapq.heappush(pq, (new_fh, nb))

    # Any vertex not reachable from boundary keeps fill_h = z (no water).
    unreachable = np.isinf(fill_h)
    fill_h[unreachable] = z[unreachable]

    return fill_h, np.maximum(0.0, fill_h - z)


# ── Step 2: Flow accumulation ─────────────────────────────────────────────────

def compute_flow_accumulation(z: np.ndarray, adj: list):
    """Steepest-descent flow accumulation.

    Returns
    -------
    acc        : (N,) int — upstream cell count draining through each vertex
    downstream : (N,) int — next vertex downhill (-1 = local sink)
    """
    n = len(z)
    downstream = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        nbs = adj[i]
        if not nbs:
            continue
        mn = min(nbs, key=lambda j, _z=z: _z[j])
        if z[mn] < z[i]:
            downstream[i] = mn

    acc = np.ones(n, dtype=np.int64)
    for i in np.argsort(-z):       # process high -> low
        d = int(downstream[i])
        if d >= 0:
            acc[d] += acc[i]

    return acc, downstream


# ── Step 3: Basin labelling ───────────────────────────────────────────────────

def label_basins(water_depth: np.ndarray, adj: list,
                 min_depth: float = 1e-4) -> np.ndarray:
    """BFS connected-component labelling of wet vertices.

    Returns
    -------
    labels : (N,) int — -1 = dry, 0..n_basins-1 = basin index
    """
    n = len(water_depth)
    wet = water_depth >= min_depth
    labels = np.full(n, -1, dtype=np.int64)
    current = 0
    for start in np.where(wet)[0]:
        start = int(start)
        if labels[start] >= 0:
            continue
        queue = [start]
        labels[start] = current
        qi = 0
        while qi < len(queue):
            v = queue[qi]; qi += 1
            for nb in adj[v]:
                if wet[nb] and labels[nb] < 0:
                    labels[nb] = current
                    queue.append(nb)
        current += 1
    return labels


# ── FFEI + volume computation ─────────────────────────────────────────────────

def _compute_ffei_and_volumes(
    vertices: np.ndarray,
    faces: np.ndarray,
    fill_h: np.ndarray,
    basin_labels: np.ndarray,
    face_labels: np.ndarray,
    n_basins: int,
    sink_indices: np.ndarray,
) -> tuple:
    """Compute FFEIResult and BasinVolumes from priority-flood output."""
    z = vertices[:, 2]
    fa = _face_areas(vertices, faces)
    total_area = float(fa.sum())

    basin_areas_d: dict = {}
    basin_vols_d:  dict = {}
    spillover_d:   dict = {}

    for k in range(n_basins):
        fmask = face_labels == k
        basin_areas_d[k] = float(fa[fmask].sum())

        fi    = faces[fmask]        # (F_k, 3)
        fi_fa = fa[fmask]           # (F_k,)
        if len(fi) > 0:
            fill_avg = fill_h[fi].mean(axis=1)
            z_avg    = z[fi].mean(axis=1)
            depth    = np.maximum(0.0, fill_avg - z_avg)
            basin_vols_d[k] = float(np.dot(depth, fi_fa))
        else:
            basin_vols_d[k] = 0.0

        verts_k = np.where(basin_labels == k)[0]
        spillover_d[k] = (float(fill_h[verts_k].max()) if len(verts_k) > 0
                          else float(z[int(sink_indices[k])]))

    wet_area = float(sum(basin_areas_d.values()))
    wet_vol  = float(sum(basin_vols_d.values()))
    ffei_val = wet_area / total_area if total_area > 1e-12 else 0.0

    # Central basin: highest volume, tie-broken by centrality
    if n_basins > 0 and len(sink_indices) >= n_basins:
        cxy   = vertices[:, :2].mean(axis=0)
        s_xy  = vertices[sink_indices[:n_basins], :2]
        d_xy  = np.linalg.norm(s_xy - cxy, axis=1)
        vols  = np.array([basin_vols_d[k] for k in range(n_basins)],
                         dtype=np.float64)
        dr = max(float(d_xy.max() - d_xy.min()), 1e-12)
        vr = max(float(vols.max() - vols.min()), 1e-12)
        ds = (d_xy - d_xy.min()) / dr
        vs = 1.0 - (vols - vols.min()) / vr   # 0 = largest volume
        central_id = int(np.argmin(vs + 0.4 * ds))
    else:
        central_id = 0

    ret_ids  = list(range(n_basins))
    per_ffei = {k: basin_areas_d[k] / total_area
                for k in ret_ids if total_area > 1e-12}

    ffei_result = FFEIResult(
        ffei=ffei_val,
        basin_volume_mm3=wet_vol,
        basin_area_mm2=wet_area,
        total_area=total_area,
        n_basins=n_basins,
        basin_areas=basin_areas_d,
        basin_volumes=basin_vols_d,
        retention_area=wet_area,
        escape_area=total_area - wet_area,
        retention_basin_ids=ret_ids,
        escape_basin_ids=[],
        central_basin_id=central_id,
        central_area=basin_areas_d.get(central_id, 0.0),
        peripheral_area=total_area - wet_area,
        per_basin_ffei=per_ffei,
        n_retention_basins=n_basins,
        n_escape_basins=0,
    )
    basin_vol_result = BasinVolumes(
        per_basin_mm3=basin_vols_d,
        total_retention_mm3=wet_vol,
        per_basin_spillover_mm=spillover_d,
    )
    return ffei_result, basin_vol_result


# ── Public entry point ────────────────────────────────────────────────────────

def analyze_basins(vertices: np.ndarray,
                   faces: np.ndarray,
                   min_basin_depth_mm: float = 0.01,
                   boundary_verts: Optional[set] = None):
    """Priority-flood watershed analysis (Zuccotti 1998 in 3D).

    Floods the mesh surface from its boundary inward using a min-heap.
    Every vertex that is lower than its flood-fill level accumulates
    water; connected wet regions form basins.  No merge heuristics or
    classification thresholds are required.

    Parameters
    ----------
    vertices           : (N, 3) float64 — mesh vertices, +Z = occlusal
    faces              : (F, 3) int    — triangle connectivity
    min_basin_depth_mm : float — minimum water depth (mm) to form a basin
    boundary_verts     : set, optional — pre-computed boundary indices

    Returns
    -------
    watershed     : WatershedResult
    ffei_result   : FFEIResult
    basin_volumes : BasinVolumes
    """
    faces_a = np.asarray(faces, dtype=np.int64)
    z = vertices[:, 2].astype(np.float64)
    n = len(vertices)

    adj = _build_adjacency(faces_a, n)

    if boundary_verts is None:
        boundary_verts = _find_boundary_vertices(faces_a, n, vertices)

    # 1. Flood-fill
    fill_h, water_depth = priority_flood(z, adj, boundary_verts)

    # 2. Flow accumulation
    flow_acc, downstream = compute_flow_accumulation(z, adj)

    # 3. Basin labels
    basin_labels = label_basins(water_depth, adj, min_depth=min_basin_depth_mm)
    n_basins = int(basin_labels.max()) + 1 if (basin_labels >= 0).any() else 0

    # 4. Per-basin sink vertices
    if n_basins > 0:
        sinks = []
        for k in range(n_basins):
            vk = np.where(basin_labels == k)[0]
            sinks.append(int(vk[np.argmin(z[vk])]) if len(vk) > 0 else 0)
        sink_indices   = np.array(sinks, dtype=np.int64)
        sink_positions = vertices[sink_indices]
    else:
        sink_indices   = np.zeros(0, dtype=np.int64)
        sink_positions = np.zeros((0, 3))

    # 5. Face labels (majority vote)
    face_labels = _face_labels_from_verts(basin_labels, faces_a)

    ws = WatershedResult(
        fill_height=fill_h,
        water_depth=water_depth,
        flow_acc=flow_acc,
        downstream=downstream,
        basin_labels=basin_labels,
        n_basins=n_basins,
        sink_indices=sink_indices,
        sink_positions=sink_positions,
        face_labels=face_labels,
    )

    # 6. FFEI + volumes
    if n_basins > 0:
        ffei_result, basin_vols = _compute_ffei_and_volumes(
            vertices, faces_a, fill_h, basin_labels, face_labels,
            n_basins, sink_indices,
        )
    else:
        fa = _face_areas(vertices, faces_a)
        total_area = float(fa.sum())
        ffei_result = FFEIResult(
            ffei=0.0, basin_volume_mm3=0.0, basin_area_mm2=0.0,
            total_area=total_area, n_basins=0,
            basin_areas={}, basin_volumes={},
            retention_area=0.0, escape_area=total_area,
            retention_basin_ids=[], escape_basin_ids=[],
            central_basin_id=0, central_area=0.0,
            peripheral_area=total_area,
            per_basin_ffei={}, n_retention_basins=0, n_escape_basins=0,
        )
        basin_vols = BasinVolumes({}, 0.0, {})

    return ws, ffei_result, basin_vols
