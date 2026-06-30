"""
FFEI — Food-Flow Estimation Index
==================================

A topographic-functional analysis module for 3D dental occlusal surfaces.

FFEI quantifies the functional organization of the occlusal relief by
applying discrete watershed analysis on triangular meshes, producing
basin maps and a retention index that describes how much of the occlusal
surface drains toward the central fovea versus lateral margins.

Designed for integration with the DAMOS pipeline.

Author: Dr Albert Epitíe Dyowe Roig
"""

__version__ = "0.1.0"

# Lazy imports — modules are imported on first use, not at package load.
# This avoids requiring trimesh when running inside DAMOS (which uses
# PyVista for mesh I/O instead).
#
# Usage:
#   from ffei.surface_fields import compute_surface_fields
#   from ffei.detection import detect_minima, detect_maxima, cluster_features
#   from ffei.watershed import discrete_watershed, merge_basins, compute_ffei
#   from ffei.metrics import extract_metrics
#   from ffei.io import load_mesh, export_metrics_csv  # requires trimesh
#   from ffei.pipeline import analyze_mesh, batch_analyze  # requires trimesh