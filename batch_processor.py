"""
DAMOS Batch Orientation Processor
==================================
Processes multiple meshes through the orientation pipeline
with progress tracking, configurable thresholds, and export.

Author: Albert Epitíe Dyowe Roig
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from enum import Enum, auto
import json
import time

from .orientation_engine import (
    OrientationMethod,
    OrientationResult,
    compute_pca_orientation,
    compute_bestfit_plane,
    compute_three_landmark_orientation,
    compute_icp_alignment,
    apply_orientation,
)


class BatchItemStatus(Enum):
    PENDING = auto()
    PROCESSING = auto()
    DONE = auto()
    FLAGGED = auto()  # residual above threshold
    ERROR = auto()
    SKIPPED = auto()


@dataclass
class BatchItem:
    """Single mesh in the batch queue."""
    filepath: Path
    status: BatchItemStatus = BatchItemStatus.PENDING
    result: Optional[OrientationResult] = None
    oriented_vertices: Optional[np.ndarray] = None
    error_message: str = ""
    processing_time: float = 0.0

    @property
    def filename(self) -> str:
        return self.filepath.name

    @property
    def residual(self) -> Optional[float]:
        if self.result:
            return self.result.residual_angle_deg
        return None


@dataclass
class BatchConfig:
    """Configuration for batch orientation processing."""
    method: OrientationMethod = OrientationMethod.PCA
    residual_threshold_deg: float = 5.0  # flag meshes above this
    center_meshes: bool = True
    export_format: str = "ply"  # ply, stl, obj
    output_suffix: str = "_oriented"
    output_directory: Optional[Path] = None
    reference_mesh_path: Optional[Path] = None  # for ICP
    overwrite: bool = False


class BatchProcessor:
    """
    Processes a queue of meshes through orientation pipeline.
    
    Signals progress through callbacks for GUI integration.
    
    Usage
    -----
    processor = BatchProcessor(config)
    processor.add_files([path1, path2, ...])
    processor.on_progress = my_callback
    processor.run()
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.items: List[BatchItem] = []
        self._reference_vertices: Optional[np.ndarray] = None
        self._cancelled = False

        # Callbacks
        self.on_progress: Optional[Callable[[int, int, BatchItem], None]] = None
        self.on_item_complete: Optional[Callable[[BatchItem], None]] = None
        self.on_batch_complete: Optional[Callable[[List[BatchItem]], None]] = None
        self.on_error: Optional[Callable[[BatchItem, str], None]] = None

    def add_files(self, paths: List[Path]):
        """Add mesh files to the queue."""
        for p in paths:
            path = Path(p)
            if path.exists() and path.suffix.lower() in ('.ply', '.stl', '.obj', '.off'):
                self.items.append(BatchItem(filepath=path))

    def remove_item(self, index: int):
        """Remove item from queue by index."""
        if 0 <= index < len(self.items):
            self.items.pop(index)

    def clear(self):
        """Clear the entire queue."""
        self.items.clear()

    def cancel(self):
        """Signal cancellation of running batch."""
        self._cancelled = True

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def completed(self) -> int:
        return sum(1 for i in self.items if i.status in (
            BatchItemStatus.DONE, BatchItemStatus.FLAGGED
        ))

    @property
    def flagged_items(self) -> List[BatchItem]:
        return [i for i in self.items if i.status == BatchItemStatus.FLAGGED]

    def run(self):
        """
        Execute batch orientation on all pending items.
        
        This is designed to be called from a QThread worker
        for non-blocking GUI integration.
        """
        self._cancelled = False

        # Load reference mesh if ICP is selected
        if self.config.method == OrientationMethod.REFERENCE_ICP:
            if not self._load_reference():
                return

        for idx, item in enumerate(self.items):
            if self._cancelled:
                item.status = BatchItemStatus.SKIPPED
                continue

            if item.status not in (BatchItemStatus.PENDING, BatchItemStatus.ERROR):
                continue

            item.status = BatchItemStatus.PROCESSING

            if self.on_progress:
                self.on_progress(idx, self.total, item)

            try:
                t0 = time.time()
                self._process_item(item)
                item.processing_time = time.time() - t0

                # Flag if residual exceeds threshold
                if (item.result and
                        item.result.residual_angle_deg > self.config.residual_threshold_deg):
                    item.status = BatchItemStatus.FLAGGED
                else:
                    item.status = BatchItemStatus.DONE

                if self.on_item_complete:
                    self.on_item_complete(item)

            except Exception as e:
                item.status = BatchItemStatus.ERROR
                item.error_message = str(e)
                if self.on_error:
                    self.on_error(item, str(e))

        if self.on_batch_complete:
            self.on_batch_complete(self.items)

    def _process_item(self, item: BatchItem):
        """Process a single mesh item."""
        vertices, faces = self._load_mesh(item.filepath)

        if self.config.method == OrientationMethod.PCA:
            result = compute_pca_orientation(vertices)
        elif self.config.method == OrientationMethod.BEST_FIT_PLANE:
            result = compute_bestfit_plane(vertices)
        elif self.config.method == OrientationMethod.REFERENCE_ICP:
            result = compute_icp_alignment(vertices, self._reference_vertices)
        else:
            result = compute_pca_orientation(vertices)

        item.result = result
        item.oriented_vertices = apply_orientation(
            vertices, result, center=self.config.center_meshes
        )

    def _load_mesh(self, path: Path) -> tuple:
        """
        Load mesh vertices and faces.
        
        Uses trimesh if available, falls back to basic PLY reader.
        """
        try:
            import trimesh
            mesh = trimesh.load(str(path), process=False)
            return np.array(mesh.vertices), np.array(mesh.faces)
        except ImportError:
            # Fallback: basic PLY reader
            return self._read_ply_basic(path)

    def _read_ply_basic(self, path: Path) -> tuple:
        """Minimal PLY ASCII reader as fallback."""
        vertices = []
        faces = []
        n_verts = 0
        n_faces = 0
        in_header = True
        vert_count = 0

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if in_header:
                    if line.startswith('element vertex'):
                        n_verts = int(line.split()[-1])
                    elif line.startswith('element face'):
                        n_faces = int(line.split()[-1])
                    elif line == 'end_header':
                        in_header = False
                else:
                    parts = line.split()
                    if vert_count < n_verts:
                        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        vert_count += 1
                    else:
                        # Face line: n_verts idx0 idx1 idx2 ...
                        face_indices = [int(x) for x in parts[1:]]
                        faces.append(face_indices)

        return np.array(vertices), np.array(faces)

    def _load_reference(self) -> bool:
        """Load reference mesh for ICP alignment."""
        if self.config.reference_mesh_path is None:
            for item in self.items:
                item.status = BatchItemStatus.ERROR
                item.error_message = "No reference mesh specified for ICP."
            return False

        try:
            verts, _ = self._load_mesh(self.config.reference_mesh_path)
            self._reference_vertices = verts
            return True
        except Exception as e:
            for item in self.items:
                item.status = BatchItemStatus.ERROR
                item.error_message = f"Cannot load reference: {e}"
            return False

    def export_results(self, output_dir: Optional[Path] = None):
        """
        Export oriented meshes and a summary report.
        
        Parameters
        ----------
        output_dir : Path, optional
            Override output directory from config.
        """
        out_dir = output_dir or self.config.output_directory
        if out_dir is None:
            return

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "method": self.config.method.name,
            "threshold_deg": self.config.residual_threshold_deg,
            "total_meshes": self.total,
            "completed": self.completed,
            "flagged": len(self.flagged_items),
            "items": []
        }

        for item in self.items:
            entry = {
                "filename": item.filename,
                "status": item.status.name,
                "residual_deg": round(item.residual, 2) if item.residual else None,
                "processing_time_s": round(item.processing_time, 3),
                "error": item.error_message or None,
            }
            report["items"].append(entry)

        # Save report
        report_path = out_dir / "orientation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

    def get_summary(self) -> dict:
        """Return batch processing summary."""
        residuals = [i.residual for i in self.items if i.residual is not None]
        return {
            "total": self.total,
            "completed": self.completed,
            "flagged": len(self.flagged_items),
            "errors": sum(1 for i in self.items if i.status == BatchItemStatus.ERROR),
            "mean_residual": float(np.mean(residuals)) if residuals else 0.0,
            "max_residual": float(np.max(residuals)) if residuals else 0.0,
            "min_residual": float(np.min(residuals)) if residuals else 0.0,
        }
