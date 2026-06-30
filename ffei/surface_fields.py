"""
Surface Fields — Layer 1
========================

Compute per-vertex scalar and vector fields on a triangular mesh:
  - Relative height (Z normalized to [0, 1])
  - Gradient vector (face-based, projected to tangent plane)
  - Slope magnitude (degrees)
  - Mean curvature (uniform Laplacian approximation)

All field computation is fully vectorised (no Python loops over vertices).
"""

import numpy as np
from typing import NamedTuple


class SurfaceFields(NamedTuple):
    """Container for all per-vertex surface fields."""
    height: np.ndarray          # (N,) relative height [0, 1]
    gradient: np.ndarray        # (N, 3) gradient vector on surface
    slope: np.ndarray           # (N,) slope magnitude in degrees
    curvature: np.ndarray       # (N,) mean curvature (Laplacian approx)
    normals: np.ndarray         # (N, 3) vertex normals


# ── Vertex normals (area-weighted, vectorised) ─────────────────────────────

def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)          # area-weighted face normals
    vn = np.zeros_like(vertices)
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return vn / norms


# ── Height ─────────────────────────────────────────────────────────────────

def _relative_height(vertices: np.ndarray) -> np.ndarray:
    z = vertices[:, 2].copy()
    zr = z.max() - z.min()
    if zr < 1e-12:
        return np.zeros(len(z))
    return (z - z.min()) / zr


# ── Gradient (fully vectorised, face-based) ────────────────────────────────

def _compute_gradient(vertices: np.ndarray, height: np.ndarray,
                      faces: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    Face-based gradient of the height field (fully vectorised).

    For each triangle we solve the 2×2 linear system that gives the
    gradient of the piecewise-linear height interpolant:

        G @ [a1, a2]ᵀ = [dh1, dh2]ᵀ
        G = [[e1·e1, e1·e2], [e2·e1, e2·e2]]  (metric tensor)

    Face gradients are then area-weighted and averaged to vertices,
    then projected onto each vertex tangent plane.
    """
    n = len(vertices)
    nf = len(faces)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    h0 = height[faces[:, 0]]
    h1 = height[faces[:, 1]]
    h2 = height[faces[:, 2]]

    e1 = v1 - v0                           # (F, 3) first edge
    e2 = v2 - v0                           # (F, 3) second edge
    dh1 = h1 - h0                          # (F,)
    dh2 = h2 - h0                          # (F,)

    # Metric tensor (symmetric 2×2 per face)
    g00 = np.einsum('ij,ij->i', e1, e1)
    g01 = np.einsum('ij,ij->i', e1, e2)
    g11 = np.einsum('ij,ij->i', e2, e2)
    det = g00 * g11 - g01 * g01

    # Solve for gradient coefficients (avoid degenerate faces)
    valid = det > 1e-20
    a1 = np.zeros(nf); a2 = np.zeros(nf)
    a1[valid] = ( g11[valid] * dh1[valid] - g01[valid] * dh2[valid]) / det[valid]
    a2[valid] = (-g01[valid] * dh1[valid] + g00[valid] * dh2[valid]) / det[valid]

    # Face gradient in 3D (lives in the triangle tangent plane)
    fg = a1[:, None] * e1 + a2[:, None] * e2   # (F, 3)

    # Weight by face area for accumulation
    area = 0.5 * np.sqrt(np.maximum(det, 0.0))
    fg_w = fg * area[:, None]

    # Accumulate to vertices
    grad = np.zeros((n, 3))
    w    = np.zeros(n)
    np.add.at(grad, faces[:, 0], fg_w)
    np.add.at(grad, faces[:, 1], fg_w)
    np.add.at(grad, faces[:, 2], fg_w)
    np.add.at(w,    faces[:, 0], area)
    np.add.at(w,    faces[:, 1], area)
    np.add.at(w,    faces[:, 2], area)

    w = np.where(w < 1e-20, 1.0, w)
    grad /= w[:, None]

    # Project each vertex gradient onto its tangent plane
    dot = np.einsum('ij,ij->i', grad, normals)
    grad = grad - dot[:, None] * normals

    return grad


# ── Slope ──────────────────────────────────────────────────────────────────

def _slope_from_gradient(gradient: np.ndarray) -> np.ndarray:
    """Convert gradient magnitude to slope in degrees."""
    return np.degrees(np.arctan(np.linalg.norm(gradient, axis=1)))


# ── Curvature (vectorised uniform Laplacian) ───────────────────────────────

def _mean_curvature_laplacian(vertices: np.ndarray, faces: np.ndarray,
                               normals: np.ndarray) -> np.ndarray:
    """
    Approximate mean curvature via the uniform umbrella operator.

    For each vertex i:  L(i) = mean_j(v_j) − v_i   (j ∈ 1-ring of i)
    Signed curvature = L(i) · n_i

    Fully vectorised using np.add.at.
    Each edge (i,j) contributes v_j to vertex i and v_i to vertex j.
    Edges shared by two faces are counted twice (acceptable approximation).
    """
    n = len(vertices)
    nbr_sum = np.zeros((n, 3))
    nbr_cnt = np.zeros(n)

    for (k0, k1) in ((0, 1), (0, 2), (1, 2)):
        vi = faces[:, k0]
        vj = faces[:, k1]
        np.add.at(nbr_sum, vi, vertices[vj])
        np.add.at(nbr_cnt, vi, 1)
        np.add.at(nbr_sum, vj, vertices[vi])
        np.add.at(nbr_cnt, vj, 1)

    nbr_cnt = np.where(nbr_cnt < 1, 1.0, nbr_cnt)
    mean_pos = nbr_sum / nbr_cnt[:, None]
    laplacian = mean_pos - vertices
    return np.einsum('ij,ij->i', laplacian, normals)


# ── Public entry point ─────────────────────────────────────────────────────

def compute_surface_fields(vertices: np.ndarray,
                           faces: np.ndarray) -> SurfaceFields:
    """Compute all per-vertex surface fields for a dental mesh.

    Parameters
    ----------
    vertices : (N, 3) float64 array
        Vertex positions.  Mesh must be oriented +Z up (occlusal plane ≈ XY).
    faces : (F, 3) int array
        Triangle connectivity.

    Returns
    -------
    SurfaceFields named-tuple.
    """
    normals   = _vertex_normals(vertices, faces)
    height    = _relative_height(vertices)
    gradient  = _compute_gradient(vertices, height, faces, normals)
    slope     = _slope_from_gradient(gradient)
    curvature = _mean_curvature_laplacian(vertices, faces, normals)

    return SurfaceFields(
        height=height,
        gradient=gradient,
        slope=slope,
        curvature=curvature,
        normals=normals,
    )
