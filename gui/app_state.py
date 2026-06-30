"""
DAMOS app state – shared mesh list, selection, and Qt signals.
"""

from PySide6.QtCore import QObject, Signal
import numpy as np


class AppState(QObject):
    """
    Central observable state shared across all modules.
    Every panel/module receives a reference to this object
    and connects to its signals instead of calling each other directly.
    """

    # File list changed (new batch loaded)
    files_changed = Signal(list)          # list[str] – absolute paths

    # Current selection changed
    current_index_changed = Signal(int)   # index into files list

    # Current mesh changed (after load or processing)
    mesh_changed = Signal(object)         # pyvista.PolyData or None

    # Landmarks changed (added / moved / deleted)
    landmarks_changed = Signal(object)    # dict { label: np.ndarray(3,) } or None

    # Status / log message for the bottom bar
    status_message = Signal(str)

    # Processing started / finished (bool = True → busy)
    processing_state = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._files: list[str] = []
        self._current_index: int = -1
        self._current_mesh = None          # pyvista.PolyData
        self._landmarks: dict = {}         # { label: np.ndarray(3,) }

    # ── files ──────────────────────────────────────────────

    @property
    def files(self) -> list[str]:
        return self._files

    def set_files(self, paths: list[str]):
        self._files = list(paths)
        self.files_changed.emit(self._files)
        if self._files:
            self.set_current_index(0)
        else:
            self.set_current_index(-1)

    # ── current index ──────────────────────────────────────

    @property
    def current_index(self) -> int:
        return self._current_index

    def set_current_index(self, idx: int):
        if idx < -1 or (self._files and idx >= len(self._files)):
            return
        self._current_index = idx
        self.current_index_changed.emit(idx)

    @property
    def current_path(self) -> str | None:
        if 0 <= self._current_index < len(self._files):
            return self._files[self._current_index]
        return None

    def go_previous(self):
        if self._current_index > 0:
            self.set_current_index(self._current_index - 1)

    def go_next(self):
        if self._current_index < len(self._files) - 1:
            self.set_current_index(self._current_index + 1)

    # ── mesh ───────────────────────────────────────────────

    @property
    def current_mesh(self):
        return self._current_mesh

    def set_mesh(self, mesh):
        self._current_mesh = mesh
        self.mesh_changed.emit(mesh)

    # ── landmarks ──────────────────────────────────────────

    @property
    def landmarks(self) -> dict:
        return self._landmarks

    def set_landmarks(self, lmks: dict):
        self._landmarks = lmks if lmks is not None else {}
        self.landmarks_changed.emit(self._landmarks)

    def clear_landmarks(self):
        self._landmarks = {}
        self.landmarks_changed.emit(self._landmarks)

    # ── helpers ────────────────────────────────────────────

    def post_status(self, msg: str):
        self.status_message.emit(msg)

    def set_busy(self, busy: bool):
        self.processing_state.emit(busy)
