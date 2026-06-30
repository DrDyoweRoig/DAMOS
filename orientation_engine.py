"""
DAMOS Mesh Orientation Engine (v3)
===================================
Strategy for orienting different teeth consistently:

1. PCA pre-alignment  – align principal axes so oclusal faces +Z
2. Flip correction    – compare to reference to fix 180° ambiguities
3. ICP refinement     – fine registration against reference
4. Manual fine-tune   – Euler angle sliders on top

Author: Albert Epitíe Dyowe Roig
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum, auto


class OrientationMethod(Enum):
    PCA = auto()
    BEST_FIT_PLANE = auto()
    REFERENCE_ICP = auto()
    MANUAL = auto()
    THREE_POINT = auto()


@dataclass
class OrientationResult:
    rotation_matrix: np.ndarray
    normal_vector: np.ndarray
    residual_angle_deg: float
    centroid: np.ndarray
    method: OrientationMethod
    success: bool = True
    message: str = ""
    plane_origin: Optional[np.ndarray] = None
    icp_iterations: int = 0
    icp_mean_error: float = 0.0


# ══════════════════════════════════════════════════════════
# PCA orientation
# ══════════════════════════════════════════════════════════

def compute_pca_orientation(vertices: np.ndarray) -> OrientationResult:
    """
    PCA: smallest-variance direction = occlusal normal → align to +Z.
    """
    centroid = np.mean(vertices, axis=0)
    centered = vertices - centroid

    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    normal = eigenvectors[:, 0]
    if normal[2] < 0:
        normal = -normal

    R = _rotation_to_align(normal, np.array([0.0, 0.0, 1.0]))
    residual = np.degrees(np.arccos(np.clip(np.dot(normal, [0, 0, 1]), -1.0, 1.0)))

    return OrientationResult(
        rotation_matrix=R,
        normal_vector=normal,
        residual_angle_deg=residual,
        centroid=centroid,
        method=OrientationMethod.PCA,
        plane_origin=centroid,
        message=f"PCA: residual {residual:.1f}°"
    )


# ══════════════════════════════════════════════════════════
# Best-fit plane
# ══════════════════════════════════════════════════════════

def compute_bestfit_plane(vertices: np.ndarray) -> OrientationResult:
    """Least-squares plane fit: z = ax + by + c → normal = (-a, -b, 1)."""
    centroid = np.mean(vertices, axis=0)
    A = np.column_stack([vertices[:, 0], vertices[:, 1], np.ones(len(vertices))])
    result, _, _, _ = np.linalg.lstsq(A, vertices[:, 2], rcond=None)
    a, b, _ = result

    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal

    R = _rotation_to_align(normal, np.array([0.0, 0.0, 1.0]))
    residual = np.degrees(np.arccos(np.clip(np.dot(normal, [0, 0, 1]), -1.0, 1.0)))

    return OrientationResult(
        rotation_matrix=R, normal_vector=normal,
        residual_angle_deg=residual, centroid=centroid,
        method=OrientationMethod.BEST_FIT_PLANE,
        message=f"Best-fit plane: residual {residual:.1f}°"
    )


# ══════════════════════════════════════════════════════════
# Reference-based alignment (PCA + flip + ICP)
# ══════════════════════════════════════════════════════════

def compute_reference_alignment(
    source_vertices: np.ndarray,
    reference_vertices: np.ndarray,
    max_icp_iterations: int = 80,
    icp_tolerance: float = 1e-7,
    subsample: int = 4000,
) -> OrientationResult:
    """
    Full alignment pipeline for different teeth:

    1. PCA-align source so its oclusal faces +Z (like reference)
    2. Fix 180° flip ambiguities by comparing PCA eigenvectors
       with those of the reference
    3. ICP refinement for precise match

    This works well across different tooth types because:
    - PCA gets the broad orientation right (flat surface → +Z)
    - Flip correction resolves the 4 possible PCA sign ambiguities
    - ICP does the fine tuning
    """
    from scipy.spatial import KDTree

    # ── Step 1: PCA both ──────────────────────────────────
    src_c = source_vertices - np.mean(source_vertices, axis=0)
    ref_c = reference_vertices - np.mean(reference_vertices, axis=0)

    src_cov = np.cov(src_c.T)
    ref_cov = np.cov(ref_c.T)

    src_evals, src_evecs = np.linalg.eigh(src_cov)
    ref_evals, ref_evecs = np.linalg.eigh(ref_cov)

    # Eigenvectors sorted ascending by eigenvalue:
    #   col 0 = smallest variance (normal to flat surface)
    #   col 1 = medium
    #   col 2 = largest (longest axis of the tooth)

    # ── Step 2: Align PCA frames ──────────────────────────
    # We want src eigenvectors to point the same direction as ref
    # PCA has sign ambiguity: each eigenvector can be ±
    # Try all 8 sign combinations, pick the one whose rotated
    # source is closest to reference

    best_R = np.eye(3)
    best_error = np.inf

    # Subsample for speed
    if len(ref_c) > subsample:
        ref_sub = ref_c[np.random.choice(len(ref_c), subsample, replace=False)]
    else:
        ref_sub = ref_c
    if len(src_c) > subsample:
        src_sub = src_c[np.random.choice(len(src_c), subsample, replace=False)]
    else:
        src_sub = src_c

    tree = KDTree(ref_sub)

    for sx in [1, -1]:
        for sy in [1, -1]:
            for sz in [1, -1]:
                # Create sign-flipped source frame
                S = np.diag([sz, sy, sx])  # note: col0=normal, col2=longest
                src_frame = src_evecs @ S
                ref_frame = ref_evecs

                # Rotation from src_frame to ref_frame
                R_test = ref_frame @ np.linalg.inv(src_frame)

                # Ensure proper rotation (det = +1)
                if np.linalg.det(R_test) < 0:
                    continue

                # Apply and measure
                rotated = (R_test @ src_sub.T).T
                dists, _ = tree.query(rotated)
                err = np.mean(dists)

                if err < best_error:
                    best_error = err
                    best_R = R_test

    # ── Step 3: ICP refinement ────────────────────────────
    src_aligned = (best_R @ src_c.T).T

    if len(src_aligned) > subsample:
        icp_src = src_aligned[np.random.choice(len(src_aligned), subsample, replace=False)].copy()
    else:
        icp_src = src_aligned.copy()

    cumulative_R = np.eye(3)
    prev_error = np.inf
    final_iter = 0

    for i in range(max_icp_iterations):
        distances, indices = tree.query(icp_src)
        matched = ref_sub[indices]

        H = icp_src.T @ matched
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        R_icp = Vt.T @ np.diag([1, 1, np.sign(d)]) @ U.T

        icp_src = (R_icp @ icp_src.T).T
        cumulative_R = R_icp @ cumulative_R

        mean_error = np.mean(distances)
        final_iter = i + 1
        if abs(prev_error - mean_error) < icp_tolerance:
            break
        prev_error = mean_error

    # Final error
    final_dists, _ = tree.query(icp_src)
    final_mean_error = float(np.mean(final_dists))

    # Total rotation = ICP @ PCA_with_flip
    total_R = cumulative_R @ best_R

    normal = total_R @ np.array([0, 0, 1.0])
    residual = np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0)))

    return OrientationResult(
        rotation_matrix=total_R,
        normal_vector=normal,
        residual_angle_deg=residual,
        centroid=np.mean(source_vertices, axis=0),
        method=OrientationMethod.REFERENCE_ICP,
        icp_iterations=final_iter,
        icp_mean_error=final_mean_error,
        message=f"Aligned: PCA+flip+ICP ({final_iter} iter, error {final_mean_error:.3f} mm)"
    )


# ══════════════════════════════════════════════════════════
# Three-point plane orientation
# ══════════════════════════════════════════════════════════

def compute_three_point_orientation(
    vertices: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> OrientationResult:
    """
    Orient mesh so that the plane defined by 3 points becomes
    transversal to Z (i.e. the plane normal aligns with +Z).
    The occlusal surface (majority of mesh above the plane) faces Z+.
    """
    p1, p2, p3 = np.asarray(p1), np.asarray(p2), np.asarray(p3)
    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-12:
        return OrientationResult(
            rotation_matrix=np.eye(3),
            normal_vector=np.array([0.0, 0.0, 1.0]),
            residual_angle_deg=0.0,
            centroid=np.mean(vertices, axis=0),
            method=OrientationMethod.THREE_POINT,
            success=False,
            message="The 3 points are collinear — cannot define a plane.",
        )
    normal = normal / norm_len

    centroid = np.mean(vertices, axis=0)
    plane_center = (p1 + p2 + p3) / 3.0
    above = np.sum(np.dot(vertices - plane_center, normal) > 0)
    below = len(vertices) - above
    if below > above:
        normal = -normal

    R = _rotation_to_align(normal, np.array([0.0, 0.0, 1.0]))
    residual = np.degrees(np.arccos(np.clip(np.dot(normal, [0, 0, 1]), -1.0, 1.0)))

    return OrientationResult(
        rotation_matrix=R,
        normal_vector=normal,
        residual_angle_deg=residual,
        centroid=centroid,
        method=OrientationMethod.THREE_POINT,
        plane_origin=plane_center,
        message=f"3-point plane: residual {residual:.1f}°",
    )


# ══════════════════════════════════════════════════════════
# Manual rotation
# ══════════════════════════════════════════════════════════

def compute_manual_rotation(
    pitch_deg: float, roll_deg: float, yaw_deg: float
) -> OrientationResult:
    """Euler angles (degrees), intrinsic X-Y-Z convention."""
    pitch, roll, yaw = np.radians(pitch_deg), np.radians(roll_deg), np.radians(yaw_deg)

    Rx = np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
    Ry = np.array([[np.cos(roll),0,np.sin(roll)],[0,1,0],[-np.sin(roll),0,np.cos(roll)]])
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])

    R = Rz @ Ry @ Rx
    normal = R @ np.array([0, 0, 1.0])
    residual = np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0)))

    return OrientationResult(
        rotation_matrix=R, normal_vector=normal,
        residual_angle_deg=residual, centroid=np.zeros(3),
        method=OrientationMethod.MANUAL,
        message=f"Manual: P={pitch_deg:.1f}° R={roll_deg:.1f}° Y={yaw_deg:.1f}°"
    )


# ══════════════════════════════════════════════════════════
# Apply
# ══════════════════════════════════════════════════════════

def apply_orientation(
    vertices: np.ndarray,
    result: OrientationResult,
    center: bool = True
) -> np.ndarray:
    v = vertices.copy()
    if center:
        centroid = result.centroid if result.centroid is not None else np.mean(v, axis=0)
        v -= centroid
    return (result.rotation_matrix @ v.T).T


# ══════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════

def _rotation_to_align(vec_from, vec_to):
    a = vec_from / np.linalg.norm(vec_from)
    b = vec_to / np.linalg.norm(vec_to)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-10:
        return np.eye(3) if c > 0 else np.eye(3) + 2 * _skew(
            np.cross(a, [1,0,0] if abs(a[0])<0.9 else [0,1,0])
        ) @ _skew(np.cross(a, [1,0,0] if abs(a[0])<0.9 else [0,1,0]))
    K = _skew(v)
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))

def _skew(v):
    return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])