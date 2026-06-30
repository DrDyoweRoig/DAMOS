"""
DAMOS Mesh Orientation Engine
=============================
Core algorithms for automatic, semi-automatic and manual mesh orientation.
Aligns the occlusal surface normal to +Z axis.

Methods:
    - PCA eigenvector alignment
    - Best-fit plane (least-squares)
    - 3-landmark plane definition
    - ICP-based reference alignment
    - Manual rotation (Euler angles)

Author: Albert Epitíe Dyowe Roig
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum, auto


class OrientationMethod(Enum):
    """Available orientation methods."""
    PCA = auto()
    BEST_FIT_PLANE = auto()
    THREE_LANDMARKS = auto()
    REFERENCE_ICP = auto()
    MANUAL = auto()


@dataclass
class OrientationResult:
    """Result of an orientation operation."""
    rotation_matrix: np.ndarray
    normal_vector: np.ndarray
    residual_angle_deg: float
    centroid: np.ndarray
    method: OrientationMethod
    success: bool = True
    message: str = ""
    # Store intermediate data for visualization
    plane_origin: Optional[np.ndarray] = None
    plane_vectors: Optional[Tuple[np.ndarray, np.ndarray]] = None
    landmarks_used: Optional[np.ndarray] = None


def compute_pca_orientation(vertices: np.ndarray) -> OrientationResult:
    """
    Compute orientation using PCA on mesh vertices.
    
    The occlusal surface of a tooth is typically the flattest aspect,
    meaning the smallest eigenvalue corresponds to the surface normal.
    We align this normal to +Z.
    
    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) array of vertex coordinates.
    
    Returns
    -------
    OrientationResult
        Rotation matrix aligning the occlusal normal to +Z.
    """
    centroid = np.mean(vertices, axis=0)
    centered = vertices - centroid
    
    # Covariance matrix and eigen decomposition
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Eigenvectors are sorted ascending by eigenvalue.
    # Smallest eigenvalue → direction of least variance → surface normal
    normal = eigenvectors[:, 0]
    
    # Ensure normal points upward (+Z)
    if normal[2] < 0:
        normal = -normal
    
    # Build rotation matrix to align normal with +Z
    rotation_matrix = _rotation_to_align(normal, np.array([0.0, 0.0, 1.0]))
    
    # Compute residual angle
    residual = np.degrees(np.arccos(
        np.clip(np.dot(normal, [0, 0, 1]), -1.0, 1.0)
    ))
    
    # Plane visualization vectors (the two largest eigenvectors)
    plane_v1 = eigenvectors[:, 2]
    plane_v2 = eigenvectors[:, 1]
    
    return OrientationResult(
        rotation_matrix=rotation_matrix,
        normal_vector=normal,
        residual_angle_deg=residual,
        centroid=centroid,
        method=OrientationMethod.PCA,
        plane_origin=centroid,
        plane_vectors=(plane_v1, plane_v2),
        message=f"PCA orientation: residual tilt {residual:.1f}°"
    )


def compute_bestfit_plane(vertices: np.ndarray) -> OrientationResult:
    """
    Compute orientation using least-squares best-fit plane.
    
    Fits z = ax + by + c to vertex coordinates and derives the
    plane normal from the coefficients.
    
    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) array of vertex coordinates.
    
    Returns
    -------
    OrientationResult
    """
    centroid = np.mean(vertices, axis=0)
    
    # Least squares: z = ax + by + c
    A = np.column_stack([vertices[:, 0], vertices[:, 1], np.ones(len(vertices))])
    z = vertices[:, 2]
    result, residuals, rank, sv = np.linalg.lstsq(A, z, rcond=None)
    
    a, b, c = result
    # Normal to plane z = ax + by + c is (-a, -b, 1), normalized
    normal = np.array([-a, -b, 1.0])
    normal = normal / np.linalg.norm(normal)
    
    if normal[2] < 0:
        normal = -normal
    
    rotation_matrix = _rotation_to_align(normal, np.array([0.0, 0.0, 1.0]))
    
    residual = np.degrees(np.arccos(
        np.clip(np.dot(normal, [0, 0, 1]), -1.0, 1.0)
    ))
    
    return OrientationResult(
        rotation_matrix=rotation_matrix,
        normal_vector=normal,
        residual_angle_deg=residual,
        centroid=centroid,
        method=OrientationMethod.BEST_FIT_PLANE,
        plane_origin=centroid,
        message=f"Best-fit plane: residual tilt {residual:.1f}°"
    )


def compute_three_landmark_orientation(
    vertices: np.ndarray,
    landmarks: np.ndarray
) -> OrientationResult:
    """
    Compute orientation from 3 user-selected landmark points.
    
    The three points define a plane; its normal is aligned to +Z.
    Typical usage: user clicks on 3 cusp tips or occlusal landmarks.
    
    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) mesh vertices (for centroid computation).
    landmarks : np.ndarray
        (3, 3) array of three 3D points on the occlusal surface.
    
    Returns
    -------
    OrientationResult
    """
    if landmarks.shape != (3, 3):
        return OrientationResult(
            rotation_matrix=np.eye(3),
            normal_vector=np.array([0, 0, 1.0]),
            residual_angle_deg=0.0,
            centroid=np.mean(vertices, axis=0),
            method=OrientationMethod.THREE_LANDMARKS,
            success=False,
            message="Exactly 3 landmarks required."
        )
    
    p1, p2, p3 = landmarks
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    
    if norm < 1e-10:
        return OrientationResult(
            rotation_matrix=np.eye(3),
            normal_vector=np.array([0, 0, 1.0]),
            residual_angle_deg=0.0,
            centroid=np.mean(vertices, axis=0),
            method=OrientationMethod.THREE_LANDMARKS,
            success=False,
            message="Landmarks are collinear — cannot define a plane."
        )
    
    normal = normal / norm
    if normal[2] < 0:
        normal = -normal
    
    centroid = np.mean(vertices, axis=0)
    rotation_matrix = _rotation_to_align(normal, np.array([0.0, 0.0, 1.0]))
    
    residual = np.degrees(np.arccos(
        np.clip(np.dot(normal, [0, 0, 1]), -1.0, 1.0)
    ))
    
    return OrientationResult(
        rotation_matrix=rotation_matrix,
        normal_vector=normal,
        residual_angle_deg=residual,
        centroid=centroid,
        method=OrientationMethod.THREE_LANDMARKS,
        landmarks_used=landmarks,
        plane_origin=np.mean(landmarks, axis=0),
        message=f"3-landmark plane: residual tilt {residual:.1f}°"
    )


def compute_icp_alignment(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-6
) -> OrientationResult:
    """
    Align source mesh to a reference (target) mesh using ICP.
    
    Simple point-to-point ICP implementation. For production use,
    consider using Open3D or trimesh ICP for better performance.
    
    Parameters
    ----------
    source_vertices : np.ndarray
        (N, 3) vertices to orient.
    target_vertices : np.ndarray
        (M, 3) reference mesh vertices.
    max_iterations : int
        Maximum ICP iterations.
    tolerance : float
        Convergence threshold on mean distance change.
    
    Returns
    -------
    OrientationResult
    """
    from scipy.spatial import KDTree
    
    src = source_vertices.copy()
    src_centroid = np.mean(src, axis=0)
    tgt_centroid = np.mean(target_vertices, axis=0)
    
    # Center both
    src_c = src - src_centroid
    tgt_c = target_vertices - tgt_centroid
    
    tree = KDTree(tgt_c)
    
    cumulative_R = np.eye(3)
    prev_error = np.inf
    
    for i in range(max_iterations):
        # Find closest points
        distances, indices = tree.query(src_c)
        matched = tgt_c[indices]
        
        # Compute optimal rotation (Kabsch algorithm)
        H = src_c.T @ matched
        U, S, Vt = np.linalg.svd(H)
        
        # Handle reflection
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1, 1, np.sign(d)])
        R = Vt.T @ sign_matrix @ U.T
        
        src_c = (R @ src_c.T).T
        cumulative_R = R @ cumulative_R
        
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    
    # Extract the effective normal alignment
    # The Z-axis of the rotated frame
    normal = cumulative_R @ np.array([0, 0, 1.0])
    residual = np.degrees(np.arccos(
        np.clip(abs(normal[2]), 0.0, 1.0)
    ))
    
    return OrientationResult(
        rotation_matrix=cumulative_R,
        normal_vector=normal,
        residual_angle_deg=residual,
        centroid=src_centroid,
        method=OrientationMethod.REFERENCE_ICP,
        message=f"ICP alignment: {i+1} iterations, residual {residual:.1f}°"
    )


def compute_manual_rotation(
    pitch_deg: float,
    roll_deg: float,
    yaw_deg: float
) -> OrientationResult:
    """
    Compute rotation matrix from Euler angles (degrees).
    
    Convention: intrinsic rotations X-Y-Z (pitch, roll, yaw).
    
    Parameters
    ----------
    pitch_deg : float
        Rotation around X axis.
    roll_deg : float
        Rotation around Y axis.
    yaw_deg : float
        Rotation around Z axis.
    
    Returns
    -------
    OrientationResult
    """
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    yaw = np.radians(yaw_deg)
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    Ry = np.array([
        [np.cos(roll), 0, np.sin(roll)],
        [0, 1, 0],
        [-np.sin(roll), 0, np.cos(roll)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    R = Rz @ Ry @ Rx
    
    # Effective normal after rotation
    normal = R @ np.array([0, 0, 1.0])
    residual = np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0)))
    
    return OrientationResult(
        rotation_matrix=R,
        normal_vector=normal,
        residual_angle_deg=residual,
        centroid=np.zeros(3),
        method=OrientationMethod.MANUAL,
        message=f"Manual: pitch={pitch_deg:.1f}° roll={roll_deg:.1f}° yaw={yaw_deg:.1f}°"
    )


def apply_orientation(
    vertices: np.ndarray,
    result: OrientationResult,
    center: bool = True
) -> np.ndarray:
    """
    Apply an OrientationResult to a set of vertices.
    
    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) original vertices.
    result : OrientationResult
        The computed orientation.
    center : bool
        If True, center the mesh at origin before rotating.
    
    Returns
    -------
    np.ndarray
        (N, 3) oriented vertices.
    """
    v = vertices.copy()
    
    if center:
        centroid = result.centroid if result.centroid is not None else np.mean(v, axis=0)
        v = v - centroid
    
    v = (result.rotation_matrix @ v.T).T
    
    return v


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rotation_to_align(vec_from: np.ndarray, vec_to: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix that aligns vec_from to vec_to.
    Uses Rodrigues' rotation formula.
    
    Parameters
    ----------
    vec_from : np.ndarray
        Source unit vector (3,).
    vec_to : np.ndarray
        Target unit vector (3,).
    
    Returns
    -------
    np.ndarray
        (3, 3) rotation matrix.
    """
    a = vec_from / np.linalg.norm(vec_from)
    b = vec_to / np.linalg.norm(vec_to)
    
    v = np.cross(a, b)
    c = np.dot(a, b)
    
    # If vectors are nearly parallel
    if np.linalg.norm(v) < 1e-10:
        if c > 0:
            return np.eye(3)
        else:
            # 180° rotation: find perpendicular axis
            perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(a, perp)
            axis = axis / np.linalg.norm(axis)
            # Rotation by π around axis
            K = _skew(axis)
            return np.eye(3) + 2 * K @ K
    
    # Rodrigues formula: R = I + [v]x + [v]x^2 * (1/(1+c))
    K = _skew(v)
    R = np.eye(3) + K + K @ K * (1.0 / (1.0 + c))
    
    return R


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
