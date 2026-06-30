"""
Feature Detection — Layer 2
============================

Detect anatomically relevant minima (foveae, depressions) and maxima
(cusps) on the occlusal surface, with robust filtering to avoid edge
artifacts and noise.

Key principles:
  - A minimum is a vertex whose Z is lower than ALL its 1-ring neighbors.
  - Minima are filtered by: distance to centroid, relative depth, and
    position within the occlusal basin (not on lateral walls).
  - Nearby minima are clustered into single functional depressions.
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import NamedTuple


class DetectedFeatures(NamedTuple):
    """Container for detected topographic features."""
    indices: np.ndarray          # (K,) vertex indices
    positions: np.ndarray        # (K, 3) 3D positions
    values: np.ndarray           # (K,) height values
    labels: np.ndarray           # (K,) cluster labels (-1 = unclustered)


# ── Per-tooth-type detection presets ──────────────────────

_DETECTION_PRESETS = {
    # Bunodont human molars: deep foveae, prominent cusps, central basins
    "M1 lower – Sapiens (5 cusps)": {
        "centroid_filter_minima": 0.70,
        "depth_threshold": 0.05,
        "slope_max": 60.0,
        "prominence_threshold": 0.10,
        "centroid_filter_maxima": 0.85,
        "cluster_distance_fraction": 0.05,
    },
    "M1 upper – Sapiens (4 cusps)": {
        "centroid_filter_minima": 0.70,
        "depth_threshold": 0.05,
        "slope_max": 60.0,
        "prominence_threshold": 0.10,
        "centroid_filter_maxima": 0.85,
        "cluster_distance_fraction": 0.05,
    },
    "M2 lower – Sapiens (4 cusps)": {
        "centroid_filter_minima": 0.70,
        "depth_threshold": 0.05,
        "slope_max": 60.0,
        "prominence_threshold": 0.10,
        "centroid_filter_maxima": 0.85,
        "cluster_distance_fraction": 0.05,
    },
    "M2 upper – Sapiens (4 cusps)": {
        "centroid_filter_minima": 0.70,
        "depth_threshold": 0.05,
        "slope_max": 60.0,
        "prominence_threshold": 0.10,
        "centroid_filter_maxima": 0.85,
        "cluster_distance_fraction": 0.05,
    },
    "Pm – Sapiens (2 cusps)": {
        "centroid_filter_minima": 0.65,
        "depth_threshold": 0.04,
        "slope_max": 55.0,
        "prominence_threshold": 0.08,
        "centroid_filter_maxima": 0.80,
        "cluster_distance_fraction": 0.06,
    },
    # Bilophodont: shallow transverse depressions between lophs,
    # small lateral foveae that must not be filtered out
    "Bilophodont (4 cusps)": {
        "centroid_filter_minima": 0.82,
        "depth_threshold": 0.02,
        "slope_max": 65.0,
        "prominence_threshold": 0.06,
        "centroid_filter_maxima": 0.88,
        "cluster_distance_fraction": 0.04,
    },
}

# Default preset for unknown tooth types
_DEFAULT_PRESET = {
    "centroid_filter_minima": 0.70,
    "depth_threshold": 0.05,
    "slope_max": 60.0,
    "prominence_threshold": 0.10,
    "centroid_filter_maxima": 0.85,
    "cluster_distance_fraction": 0.05,
}


def get_detection_preset(tooth_type: str) -> dict:
    """Return detection parameters tuned for a given tooth type.

    Parameters
    ----------
    tooth_type : str
        One of the DAMOS tooth type identifiers.

    Returns
    -------
    dict with keys: centroid_filter_minima, depth_threshold, slope_max,
         prominence_threshold, centroid_filter_maxima,
         cluster_distance_fraction.
    """
    return _DETECTION_PRESETS.get(tooth_type, _DEFAULT_PRESET).copy()


def _vertex_min_neighbor_height(faces: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Return, for each vertex, the minimum height among its 1-ring neighbors.
    Fully vectorised using np.minimum.at.
    """
    n = len(height)
    min_nb = np.full(n, np.inf)
    for k0, k1 in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)):
        np.minimum.at(min_nb, faces[:, k0], height[faces[:, k1]])
    return min_nb


def _vertex_max_neighbor_height(faces: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Return, for each vertex, the maximum height among its 1-ring neighbors."""
    n = len(height)
    max_nb = np.full(n, -np.inf)
    for k0, k1 in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)):
        np.maximum.at(max_nb, faces[:, k0], height[faces[:, k1]])
    return max_nb


def detect_minima(vertices: np.ndarray, faces: np.ndarray,
                  height: np.ndarray,
                  slope: np.ndarray = None,
                  centroid_filter: float = 0.7,
                  depth_threshold: float = 0.05,
                  slope_max: float = 60.0,
                  max_features: int = 20) -> DetectedFeatures:
    """Detect local minima (foveae, depressions) on the occlusal surface.

    Fully vectorised: no Python loop over vertices.
    A vertex is a local minimum if its height ≤ all 1-ring neighbor heights.
    """
    n = len(vertices)

    # Vectorised local minimum detection
    min_nb = _vertex_min_neighbor_height(faces, height)
    is_minimum = height <= min_nb

    # Filter by centroid distance
    xy = vertices[:, :2]
    centroid_xy = xy.mean(axis=0)
    radii = np.linalg.norm(xy - centroid_xy, axis=1)
    max_radius = radii.max()
    if max_radius > 1e-12:
        is_minimum &= (radii / max_radius) <= centroid_filter

    # Filter by depth below median
    is_minimum &= height <= (np.median(height) - depth_threshold)

    # Filter by slope
    if slope is not None:
        is_minimum &= slope <= slope_max

    min_indices = np.where(is_minimum)[0]
    if len(min_indices) == 0:
        return DetectedFeatures(
            indices=np.array([], dtype=int),
            positions=np.zeros((0, 3)),
            values=np.array([]),
            labels=np.array([], dtype=int),
        )

    sorted_idx = min_indices[np.argsort(height[min_indices])][:max_features]
    return DetectedFeatures(
        indices=sorted_idx,
        positions=vertices[sorted_idx],
        values=height[sorted_idx],
        labels=np.arange(len(sorted_idx)),
    )


def detect_maxima(vertices: np.ndarray, faces: np.ndarray,
                  height: np.ndarray,
                  centroid_filter: float = 0.85,
                  prominence_threshold: float = 0.1,
                  max_features: int = 10) -> DetectedFeatures:
    """Detect local maxima (cusps) on the occlusal surface.

    Fully vectorised: no Python loop over vertices.
    A vertex is a local maximum if its height ≥ all 1-ring neighbor heights.
    """
    n = len(vertices)

    # Vectorised local maximum detection
    max_nb = _vertex_max_neighbor_height(faces, height)
    is_maximum = height >= max_nb

    # Filter by centroid distance
    xy = vertices[:, :2]
    centroid_xy = xy.mean(axis=0)
    radii = np.linalg.norm(xy - centroid_xy, axis=1)
    max_radius = radii.max()
    if max_radius > 1e-12:
        is_maximum &= (radii / max_radius) <= centroid_filter

    # Filter by prominence above median
    is_maximum &= height >= (np.median(height) + prominence_threshold)

    max_indices = np.where(is_maximum)[0]
    if len(max_indices) == 0:
        return DetectedFeatures(
            indices=np.array([], dtype=int),
            positions=np.zeros((0, 3)),
            values=np.array([]),
            labels=np.array([], dtype=int),
        )

    sorted_idx = max_indices[np.argsort(-height[max_indices])][:max_features]
    return DetectedFeatures(
        indices=sorted_idx,
        positions=vertices[sorted_idx],
        values=height[sorted_idx],
        labels=np.arange(len(sorted_idx)),
    )


def cluster_features(features: DetectedFeatures,
                     min_distance: float = None,
                     mesh_bbox_fraction: float = 0.05) -> DetectedFeatures:
    """Cluster nearby features into single functional units.

    Uses simple agglomerative clustering by Euclidean distance.
    Two features closer than `min_distance` are merged into one
    (keeping the more extreme — lowest for minima, highest for maxima).

    Parameters
    ----------
    features : DetectedFeatures
    min_distance : float or None
        Minimum distance between features. If None, computed as
        `mesh_bbox_fraction` times the bounding box diagonal.
    mesh_bbox_fraction : float
        Fraction of bbox diagonal to use as min_distance if not given.

    Returns
    -------
    DetectedFeatures with merged features
    """
    if len(features.indices) <= 1:
        return features

    positions = features.positions
    values = features.values

    if min_distance is None:
        bbox_diag = np.linalg.norm(positions.max(0) - positions.min(0))
        # Use the full mesh extent would be better; approximate from features
        min_distance = bbox_diag * mesh_bbox_fraction
        if min_distance < 1e-6:
            return features

    # Distance matrix
    dist = cdist(positions, positions)

    # Greedy merge: process features in order, merge close ones
    n = len(features.indices)
    merged = np.zeros(n, dtype=bool)
    clusters = []

    for i in range(n):
        if merged[i]:
            continue
        cluster = [i]
        for j in range(i + 1, n):
            if merged[j]:
                continue
            if dist[i, j] < min_distance:
                cluster.append(j)
                merged[j] = True
        clusters.append(cluster)

    # For each cluster, keep the most extreme feature
    # (We determine direction from the first feature: if low, keep lowest)
    keep_indices = []
    is_minima = np.mean(values) < 0.5  # heuristic

    for cluster in clusters:
        if is_minima:
            best = cluster[np.argmin(values[cluster])]
        else:
            best = cluster[np.argmax(values[cluster])]
        keep_indices.append(best)

    keep = np.array(keep_indices)

    return DetectedFeatures(
        indices=features.indices[keep],
        positions=features.positions[keep],
        values=features.values[keep],
        labels=np.arange(len(keep)),
    )