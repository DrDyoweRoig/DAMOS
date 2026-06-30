"""
Metrics Extraction — Layer 4
==============================

Extract per-tooth summary metrics from surface fields, detected
features, and watershed results.

Produces a flat dictionary of named metrics suitable for CSV export
and statistical comparison between specimens.
"""

import numpy as np
from .surface_fields import SurfaceFields
from .detection import DetectedFeatures
from .watershed import WatershedResult, FFEIResult, BasinVolumes


def extract_metrics(vertices: np.ndarray,
                    faces: np.ndarray,
                    fields: SurfaceFields,
                    minima: DetectedFeatures,
                    maxima: DetectedFeatures,
                    watershed: WatershedResult,
                    ffei_result: FFEIResult,
                    basin_volumes: BasinVolumes = None,
                    specimen_id: str = "unknown") -> dict:
    """Extract all FFEI metrics for a single tooth.

    Parameters
    ----------
    vertices : (N, 3) float array
    faces : (F, 3) int array
    fields : SurfaceFields
    minima : DetectedFeatures — detected foveae/depressions
    maxima : DetectedFeatures — detected cusps
    watershed : WatershedResult
    ffei_result : FFEIResult
    specimen_id : str

    Returns
    -------
    dict with named metrics
    """
    height = fields.height
    slope = fields.slope
    curvature = fields.curvature

    # --- Basic topographic stats ---
    z_raw = vertices[:, 2]
    z_range = z_raw.max() - z_raw.min()

    # --- Cusp-fovea relationship ---
    if len(maxima.indices) > 0 and len(minima.indices) > 0:
        mean_cusp_height = height[maxima.indices].mean()
        deepest_fovea_height = height[minima.indices].min()
        cusp_fovea_ratio = mean_cusp_height / max(deepest_fovea_height, 1e-6)
        cusp_fovea_depth = mean_cusp_height - deepest_fovea_height
    else:
        mean_cusp_height = np.nan
        deepest_fovea_height = np.nan
        cusp_fovea_ratio = np.nan
        cusp_fovea_depth = np.nan

    # --- Retention volumes ---
    if basin_volumes is not None:
        retention_vol = basin_volumes.total_retention_mm3
        central_vol = basin_volumes.per_basin_mm3.get(
            ffei_result.central_basin_id, 0.0
        )
    else:
        retention_vol = np.nan
        central_vol = np.nan

    # --- Basin statistics ---
    basin_areas = ffei_result.basin_areas
    n_basins = len(basin_areas)
    areas_list = list(basin_areas.values())

    if n_basins > 1:
        # Gini coefficient of basin areas (0 = equal, 1 = one dominates)
        areas_arr = np.array(areas_list)
        areas_sorted = np.sort(areas_arr)
        n = len(areas_sorted)
        cumulative = np.cumsum(areas_sorted)
        gini = (2 * np.sum((np.arange(1, n+1) * areas_sorted)) / 
                (n * cumulative[-1])) - (n + 1) / n
    else:
        gini = 0.0

    # --- Depth of primary fovea (in real mesh units) ---
    if len(minima.indices) > 0:
        primary_fovea_idx = minima.indices[0]  # deepest
        primary_fovea_depth = z_raw.max() - z_raw[primary_fovea_idx]
    else:
        primary_fovea_depth = np.nan

    metrics = {
        # Identification
        "specimen_id": specimen_id,

        # FFEI core (multi-fovea)
        "ffei": ffei_result.ffei,
        "retention_area": ffei_result.retention_area,
        "escape_area": ffei_result.escape_area,
        "total_occlusal_area": ffei_result.total_area,
        "n_retention_basins": ffei_result.n_retention_basins,
        "n_escape_basins": ffei_result.n_escape_basins,

        # Back-compat: central basin
        "central_basin_area": ffei_result.central_area,

        # Basin topology
        "n_basins": n_basins,
        "n_functional_depressions": len(minima.indices),
        "n_cusps": len(maxima.indices),
        "basin_area_gini": round(gini, 4),

        # Relief
        "z_range_mm": round(z_range, 4),
        "height_sd": round(np.std(height), 4),
        "height_skewness": round(float(_skewness(height)), 4),

        # Slope
        "slope_mean_deg": round(float(np.mean(slope)), 2),
        "slope_max_deg": round(float(np.max(slope)), 2),
        "slope_sd_deg": round(float(np.std(slope)), 2),

        # Curvature
        "curvature_mean": round(float(np.mean(curvature)), 6),
        "curvature_sd": round(float(np.std(curvature)), 6),

        # Cusp-fovea relationship
        "mean_cusp_height_rel": round(float(mean_cusp_height), 4)
            if not np.isnan(mean_cusp_height) else np.nan,
        "deepest_fovea_height_rel": round(float(deepest_fovea_height), 4)
            if not np.isnan(deepest_fovea_height) else np.nan,
        "cusp_fovea_ratio": round(float(cusp_fovea_ratio), 4)
            if not np.isnan(cusp_fovea_ratio) else np.nan,
        "cusp_fovea_depth_rel": round(float(cusp_fovea_depth), 4)
            if not np.isnan(cusp_fovea_depth) else np.nan,
        "primary_fovea_depth_mm": round(float(primary_fovea_depth), 4)
            if not np.isnan(primary_fovea_depth) else np.nan,

        # Water retention volumes
        "retention_volume_mm3": round(float(retention_vol), 4)
            if not np.isnan(float(retention_vol)) else np.nan,
        "central_basin_volume_mm3": round(float(central_vol), 4)
            if not np.isnan(float(central_vol)) else np.nan,

        # Zuccotti primary metric: total water retained in basins
        "basin_volume_mm3": round(float(ffei_result.basin_volume_mm3), 4)
            if hasattr(ffei_result, 'basin_volume_mm3') else np.nan,
    }

    return metrics


def _skewness(x: np.ndarray) -> float:
    """Compute Fisher's skewness."""
    n = len(x)
    if n < 3:
        return 0.0
    mean = x.mean()
    std = x.std()
    if std < 1e-12:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 3))
