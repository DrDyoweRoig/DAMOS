"""
FFEI Pipeline
=============

High-level function that runs the complete FFEI analysis on a single
mesh or a batch of meshes, combining all layers.
"""

from pathlib import Path
from typing import Union, List, Optional

import numpy as np

from .surface_fields import compute_surface_fields, SurfaceFields
from .detection import detect_minima, detect_maxima, cluster_features, DetectedFeatures
from .watershed import analyze_basins, WatershedResult, FFEIResult, BasinVolumes
from .metrics import extract_metrics
from .io import load_mesh, export_metrics_csv, batch_load_meshes


class FFEIAnalysis:
    """Container for the full analysis of a single specimen."""

    def __init__(self):
        self.specimen_id: str = ""
        self.vertices: np.ndarray = None
        self.faces: np.ndarray = None
        self.fields: SurfaceFields = None
        self.minima: DetectedFeatures = None
        self.maxima: DetectedFeatures = None
        self.watershed: WatershedResult = None
        self.ffei_result: FFEIResult = None
        self.basin_volumes: Optional[BasinVolumes] = None
        self.metrics: dict = None


def analyze_mesh(vertices: np.ndarray,
                 faces: np.ndarray,
                 specimen_id: str = "unknown",
                 centroid_filter_minima: float = 0.7,
                 depth_threshold: float = 0.05,
                 prominence_threshold: float = 0.1,
                 cluster_distance: float = None,
                 min_basin_depth_mm: float = 0.01,
                 # Legacy parameter names accepted but ignored
                 merge_depth_threshold: float = 0.05,
                 merge_min_area_fraction: float = 0.020) -> FFEIAnalysis:
    """Run the complete FFEI pipeline on a single mesh.

    Parameters
    ----------
    vertices : (N, 3) float array
        Must be oriented with occlusal surface pointing +Z.
    faces : (F, 3) int array
    specimen_id : str
    centroid_filter_minima : float
        Centroid filter for minima detection (0–1). Default 0.7.
    depth_threshold : float
        Minimum relative depth for minima. Default 0.05.
    prominence_threshold : float
        Minimum relative prominence for maxima. Default 0.1.
    cluster_distance : float or None
        Minimum distance for clustering features. Auto if None.
    min_basin_depth_mm : float
        Minimum water depth (mm) to form a basin. Default 0.01.

    Returns
    -------
    FFEIAnalysis with all results.
    """
    result = FFEIAnalysis()
    result.specimen_id = specimen_id
    result.vertices = vertices
    result.faces = faces

    # Layer 1: Surface fields
    result.fields = compute_surface_fields(vertices, faces)

    # Layer 2: Feature detection
    raw_minima = detect_minima(
        vertices, faces, result.fields.height,
        slope=result.fields.slope,
        centroid_filter=centroid_filter_minima,
        depth_threshold=depth_threshold,
    )
    result.minima = cluster_features(raw_minima, min_distance=cluster_distance)

    raw_maxima = detect_maxima(
        vertices, faces, result.fields.height,
        prominence_threshold=prominence_threshold,
    )
    result.maxima = cluster_features(raw_maxima, min_distance=cluster_distance)

    # Layer 3: Priority-flood watershed (Zuccotti 1998 in 3D)
    result.watershed, result.ffei_result, result.basin_volumes = analyze_basins(
        vertices, faces, min_basin_depth_mm=min_basin_depth_mm,
    )

    # Layer 4: Metrics
    result.metrics = extract_metrics(
        vertices, faces, result.fields,
        result.minima, result.maxima,
        result.watershed, result.ffei_result,
        basin_volumes=result.basin_volumes,
        specimen_id=specimen_id,
    )

    return result


def analyze_file(path: Union[str, Path],
                 specimen_id: str = None,
                 **kwargs) -> FFEIAnalysis:
    """Run FFEI analysis on a mesh file.

    Parameters
    ----------
    path : str or Path
    specimen_id : str, optional
        If None, uses the filename stem.
    **kwargs
        Passed to analyze_mesh().

    Returns
    -------
    FFEIAnalysis
    """
    path = Path(path)
    if specimen_id is None:
        specimen_id = path.stem

    vertices, faces = load_mesh(path)
    return analyze_mesh(vertices, faces, specimen_id=specimen_id, **kwargs)


def batch_analyze(directory: Union[str, Path],
                  output_csv: Union[str, Path] = None,
                  **kwargs) -> List[FFEIAnalysis]:
    """Run FFEI analysis on all meshes in a directory.

    Parameters
    ----------
    directory : str or Path
    output_csv : str or Path, optional
        If given, export all metrics to this CSV file.
    **kwargs
        Passed to analyze_mesh().

    Returns
    -------
    list of FFEIAnalysis
    """
    meshes = batch_load_meshes(directory)
    results = []

    for filename, vertices, faces in meshes:
        specimen_id = Path(filename).stem
        print(f"  Analyzing {specimen_id}...")
        try:
            analysis = analyze_mesh(
                vertices, faces, specimen_id=specimen_id, **kwargs
            )
            results.append(analysis)
            ffei = analysis.ffei_result.ffei
            n_basins = analysis.watershed.n_basins
            print(f"    FFEI = {ffei:.4f} | Basins = {n_basins}")
        except Exception as e:
            print(f"    [ERROR] {specimen_id}: {e}")

    if output_csv and results:
        metrics_list = [r.metrics for r in results]
        export_metrics_csv(metrics_list, output_csv, overwrite=True)
        print(f"\n  Metrics exported to {output_csv}")

    return results
