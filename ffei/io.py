"""
I/O Module
==========

Load meshes (PLY, STL, OBJ) and export metrics to CSV.
Uses trimesh for robust mesh loading.
"""

import csv
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import trimesh


def load_mesh(path: Union[str, Path]) -> tuple:
    """Load a triangular mesh from file.

    Supported formats: PLY, STL, OBJ, OFF, and others via trimesh.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    vertices : (N, 3) float array
    faces : (F, 3) int array

    Raises
    ------
    ValueError
        If the file cannot be loaded or is not a triangular mesh.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    mesh = trimesh.load(str(path), force='mesh')

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"File {path} did not load as a triangular mesh.")

    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int64)

    return vertices, faces


def export_metrics_csv(metrics_list: List[dict],
                       output_path: Union[str, Path],
                       overwrite: bool = False) -> Path:
    """Export a list of per-specimen metrics dictionaries to CSV.

    Parameters
    ----------
    metrics_list : list of dict
        Each dict is the output of `extract_metrics()` for one specimen.
    output_path : str or Path
    overwrite : bool
        If False and file exists, append rows. If True, overwrite.

    Returns
    -------
    Path to the written CSV file.
    """
    output_path = Path(output_path)

    if not metrics_list:
        raise ValueError("No metrics to export.")

    fieldnames = list(metrics_list[0].keys())

    mode = 'w' if overwrite or not output_path.exists() else 'a'
    write_header = mode == 'w' or not output_path.exists()

    with open(output_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(metrics_list)

    return output_path


def batch_load_meshes(directory: Union[str, Path],
                      extensions: tuple = ('.ply', '.stl', '.obj', '.off')
                      ) -> List[tuple]:
    """Load all mesh files from a directory.

    Parameters
    ----------
    directory : str or Path
    extensions : tuple of str

    Returns
    -------
    list of (filename, vertices, faces) tuples
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    results = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in extensions:
            try:
                vertices, faces = load_mesh(path)
                results.append((path.name, vertices, faces))
            except (ValueError, FileNotFoundError) as e:
                print(f"  [SKIP] {path.name}: {e}")

    return results
