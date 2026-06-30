"""
DAMOS Mesh Exporter
====================
Exports oriented meshes preserving:
- Original file format (.ply, .stl, .obj, .off)
- Suffix convention: _or (oriented)
- Face connectivity from original mesh

Author: Albert Epitíe Dyowe Roig
"""

import numpy as np
from pathlib import Path
from typing import Optional
import struct


SUFFIX = "_or"

SUPPORTED_FORMATS = {".ply", ".stl", ".obj", ".off"}


def export_oriented_mesh(
    original_path: Path,
    oriented_vertices: np.ndarray,
    faces: np.ndarray,
    output_dir: Optional[Path] = None,
    suffix: str = SUFFIX,
) -> Path:
    """
    Export an oriented mesh in the same format as the original.
    
    The output filename is: {original_stem}{suffix}.{ext}
    e.g. LM1_NAT001.ply → LM1_NAT001_or.ply
    
    Parameters
    ----------
    original_path : Path
        Path to the original mesh file (determines format).
    oriented_vertices : np.ndarray
        (N, 3) oriented vertex coordinates.
    faces : np.ndarray
        (M, 3) or (M, 4) face indices from the original mesh.
    output_dir : Path, optional
        Output directory. Defaults to same directory as original.
    suffix : str
        Suffix to append. Default: '_or'.
    
    Returns
    -------
    Path
        Path to the exported file.
    """
    original_path = Path(original_path)
    ext = original_path.suffix.lower()
    
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {ext}. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    out_dir = output_dir if output_dir else original_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_name = f"{original_path.stem}{suffix}{ext}"
    out_path = out_dir / out_name
    
    if ext == ".ply":
        _write_ply(out_path, oriented_vertices, faces)
    elif ext == ".stl":
        _write_stl(out_path, oriented_vertices, faces)
    elif ext == ".obj":
        _write_obj(out_path, oriented_vertices, faces)
    elif ext == ".off":
        _write_off(out_path, oriented_vertices, faces)
    
    return out_path


def get_output_path(
    original_path: Path,
    output_dir: Optional[Path] = None,
    suffix: str = SUFFIX,
) -> Path:
    """Preview what the output path will be without writing."""
    original_path = Path(original_path)
    ext = original_path.suffix.lower()
    out_dir = output_dir if output_dir else original_path.parent
    return Path(out_dir) / f"{original_path.stem}{suffix}{ext}"


# -----------------------------------------------------------------------
# Format writers
# -----------------------------------------------------------------------

def _write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray):
    """Write ASCII PLY file."""
    n_verts = len(vertices)
    n_faces = len(faces)
    n_face_verts = faces.shape[1] if faces.ndim == 2 else 3
    
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment Oriented by DAMOS MeshOrient\n")
        f.write(f"element vertex {n_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {n_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        for v in vertices:
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        
        for face in faces:
            indices = " ".join(str(int(i)) for i in face)
            f.write(f"{n_face_verts} {indices}\n")


def _write_stl(path: Path, vertices: np.ndarray, faces: np.ndarray):
    """Write binary STL file."""
    n_faces = len(faces)
    
    with open(path, 'wb') as f:
        # 80-byte header
        header = b"DAMOS MeshOrient" + b'\0' * 64
        f.write(header[:80])
        
        # Number of triangles
        f.write(struct.pack('<I', n_faces))
        
        for face in faces:
            # Get triangle vertices
            v0 = vertices[int(face[0])]
            v1 = vertices[int(face[1])]
            v2 = vertices[int(face[2])]
            
            # Compute normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            
            # Write: normal (3 floats) + 3 vertices (9 floats) + attribute (2 bytes)
            f.write(struct.pack('<3f', *normal))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray):
    """Write Wavefront OBJ file."""
    with open(path, 'w') as f:
        f.write("# Oriented by DAMOS MeshOrient\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        
        # OBJ faces are 1-indexed
        for face in faces:
            indices = " ".join(str(int(i) + 1) for i in face)
            f.write(f"f {indices}\n")


def _write_off(path: Path, vertices: np.ndarray, faces: np.ndarray):
    """Write Object File Format (OFF)."""
    n_verts = len(vertices)
    n_faces = len(faces)
    n_face_verts = faces.shape[1] if faces.ndim == 2 else 3
    
    with open(path, 'w') as f:
        f.write("OFF\n")
        f.write(f"{n_verts} {n_faces} 0\n")
        
        for v in vertices:
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        
        for face in faces:
            indices = " ".join(str(int(i)) for i in face)
            f.write(f"{n_face_verts} {indices}\n")
