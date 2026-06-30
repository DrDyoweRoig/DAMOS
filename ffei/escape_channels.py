"""
Escape Channels — Layer 5
=========================
Gradient-descent paths from cusp vertices to mesh boundary.
Pure NumPy — safe to call from background threads.

Use :func:`paths_to_polydata` on the main thread to get a pv.PolyData.
"""

import numpy as np


# ── Mesh helpers ──────────────────────────────────────────────────────────────

def _build_adjacency(faces: np.ndarray, n_vertices: int) -> list:
    adj = [[] for _ in range(n_vertices)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a].append(b); adj[a].append(c)
        adj[b].append(a); adj[b].append(c)
        adj[c].append(a); adj[c].append(b)
    return [list(set(nbrs)) for nbrs in adj]


def _find_boundary_vertices(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """Vertices on open boundary edges (edges shared by exactly 1 face)."""
    edge_cnt: dict = {}
    for f in faces:
        for i in range(3):
            e = (min(int(f[i]), int(f[(i + 1) % 3])),
                 max(int(f[i]), int(f[(i + 1) % 3])))
            edge_cnt[e] = edge_cnt.get(e, 0) + 1
    bnd: set = set()
    for (a, b), cnt in edge_cnt.items():
        if cnt == 1:
            bnd.add(a)
            bnd.add(b)
    return np.array(sorted(bnd), dtype=np.int64)


# ── Path following ────────────────────────────────────────────────────────────

def _gradient_descent_path(start: int,
                           height: np.ndarray,
                           adj: list,
                           boundary: set,
                           max_steps: int = 5000):
    """Walk steepest descent from *start* until boundary or local minimum.

    Returns
    -------
    path             : list of vertex indices
    reached_boundary : True if the path ended on a boundary vertex
    """
    path = [start]
    cur = start
    seen = {cur}
    for _ in range(max_steps):
        if cur in boundary:
            return path, True
        nbrs = adj[cur]
        if not nbrs:
            break
        best = min(nbrs, key=lambda n: height[n])
        if height[best] >= height[cur]:
            break          # local minimum / flat plateau
        if best in seen:
            break          # cycle guard
        seen.add(best)
        path.append(best)
        cur = best
    return path, (cur in boundary)


def _path_length_3d(path: list, vertices: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    pts = vertices[np.array(path, dtype=np.int64)]
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _tortuosity(path: list, vertices: np.ndarray) -> float:
    """τ = geodesic length L / straight-line distance D."""
    if len(path) < 2:
        return 1.0
    L = _path_length_3d(path, vertices)
    D = float(np.linalg.norm(vertices[path[-1]] - vertices[path[0]]))
    return L / D if D > 1e-10 else 1.0


# ── Public API ────────────────────────────────────────────────────────────────

def compute_escape_channels(vertices: np.ndarray,
                            faces: np.ndarray,
                            height: np.ndarray,
                            cusp_indices: np.ndarray,
                            max_paths: int = 50) -> dict:
    """Compute gradient-descent paths from each cusp (downhill to boundary or sink).

    Paths are classified by their endpoint:
    - **escape**   : reached the mesh boundary  → true escape channels
    - **retention**: ended at an interior sink   → food-retention paths

    Only escape paths contribute to ``mean_tortuosity``.

    Returns
    -------
    dict
        ``paths``              – all paths (escape + retention)
        ``path_types``         – list[str]: ``'escape'`` or ``'retention'``
        ``tortuosities``       – list[float], one per path
        ``mean_tortuosity``    – mean τ of escape paths only (np.nan if none)
        ``n_escape_paths``     – int
        ``n_retention_paths``  – int
        ``adjacency``          – adjacency list (reused by shear computation)
    """
    n = len(vertices)
    adj = _build_adjacency(faces, n)
    bnd = set(_find_boundary_vertices(faces, n).tolist())

    paths: list = []
    path_types: list = []
    taus: list = []

    for idx in cusp_indices[:max_paths]:
        p, is_escape = _gradient_descent_path(int(idx), height, adj, bnd)
        paths.append(p)
        path_types.append("escape" if is_escape else "retention")
        taus.append(_tortuosity(p, vertices))

    escape_taus = [t for t, tp in zip(taus, path_types) if tp == "escape"]
    mean_tau = float(np.mean(escape_taus)) if escape_taus else np.nan

    n_escape    = path_types.count("escape")
    n_retention = path_types.count("retention")

    return {
        "paths":             paths,
        "path_types":        path_types,
        "tortuosities":      taus,
        "mean_tortuosity":   mean_tau,
        "n_escape_paths":    n_escape,
        "n_retention_paths": n_retention,
        "adjacency":         adj,
    }


def paths_to_polydata(vertices: np.ndarray, paths: list,
                      path_types: list = None):
    """Build two ``pv.PolyData`` objects — one per path type.

    Must be called on the **main thread** (PyVista / VTK are not thread-safe).

    Parameters
    ----------
    vertices    : (N, 3) float64
    paths       : list of vertex-index lists
    path_types  : list[str] of ``'escape'`` / ``'retention'`` per path.
                  If None, all paths are treated as escape.

    Returns
    -------
    escape_pd, retention_pd : two pv.PolyData (either may be empty)
    """
    import pyvista as pv

    def _build(selected_paths):
        pts_list, cells, offset = [], [], 0
        for path in selected_paths:
            if len(path) < 2:
                continue
            n = len(path)
            pts_list.append(vertices[np.array(path, dtype=np.int64)])
            cells.extend([n] + list(range(offset, offset + n)))
            offset += n
        if not pts_list:
            return pv.PolyData()
        pd = pv.PolyData()
        pd.points = np.vstack(pts_list).astype(np.float64)
        pd.lines = np.array(cells, dtype=np.int64)
        return pd

    if path_types is None:
        return _build(paths), pv.PolyData()

    escape_paths    = [p for p, t in zip(paths, path_types) if t == "escape"]
    retention_paths = [p for p, t in zip(paths, path_types) if t == "retention"]
    return _build(escape_paths), _build(retention_paths)
