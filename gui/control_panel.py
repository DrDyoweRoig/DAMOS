"""
DAMOS Orientation Control Panel
================================
Side panel with:
- Mode tabs (Manual / Semi-automatic / Automatic)
- Method selection dropdown
- Fine-adjustment sliders (Pitch, Roll, Yaw)
- Batch queue list with status indicators
- Action buttons

Author: Albert Epitíe Dyowe Roig
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QGroupBox, QListWidget, QListWidgetItem,
    QFileDialog, QDoubleSpinBox, QTabBar, QProgressBar,
    QAbstractItemView, QSizePolicy,
)
from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtGui import QColor, QIcon
from pathlib import Path
from typing import List, Optional

from ..core.orientation_engine import OrientationMethod


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
STYLE_SHEET = """
QWidget {
    font-family: "Segoe UI", "SF Pro Text", system-ui, sans-serif;
    font-size: 13px;
}
QGroupBox {
    font-weight: 500;
    font-size: 12px;
    color: #666;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    margin-top: 14px;
    padding: 16px 12px 10px 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
QPushButton {
    background: #f8f8f8;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 6px 14px;
    color: #333;
}
QPushButton:hover { background: #eee; }
QPushButton#primary {
    background: #534AB7;
    color: white;
    border-color: #534AB7;
}
QPushButton#primary:hover { background: #3C3489; }
QComboBox {
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 5px 8px;
    background: white;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #e0e0e0;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 16px; height: 16px;
    margin: -6px 0;
    background: #534AB7;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: #AFA9EC;
    border-radius: 2px;
}
QListWidget {
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    background: white;
    outline: none;
}
QListWidget::item {
    padding: 6px 8px;
    border-bottom: 1px solid #f0f0f0;
}
QListWidget::item:selected {
    background: #EEEDFE;
    color: #333;
}
QProgressBar {
    border: none;
    border-radius: 3px;
    background: #e0e0e0;
    height: 6px;
    text-align: center;
}
QProgressBar::chunk {
    background: #534AB7;
    border-radius: 3px;
}
QTabBar::tab {
    padding: 8px 16px;
    border: none;
    background: transparent;
    color: #888;
    font-size: 13px;
}
QTabBar::tab:selected {
    color: #333;
    font-weight: 500;
    border-bottom: 2px solid #534AB7;
}
QTabBar::tab:hover:!selected { color: #555; }
"""

# Method options per mode
METHOD_OPTIONS = {
    "Manual": [
        ("Free rotation (drag)", OrientationMethod.MANUAL),
        ("Snap to axis", OrientationMethod.MANUAL),
        ("Align by 3 clicks", OrientationMethod.THREE_LANDMARKS),
    ],
    "Semi-automatic": [
        ("Best-fit plane (PCA)", OrientationMethod.PCA),
        ("Best-fit plane (LSQ)", OrientationMethod.BEST_FIT_PLANE),
        ("3 landmark points", OrientationMethod.THREE_LANDMARKS),
        ("Reference mesh (ICP)", OrientationMethod.REFERENCE_ICP),
    ],
    "Automatic": [
        ("PCA eigenvector alignment", OrientationMethod.PCA),
        ("Best-fit plane (LSQ)", OrientationMethod.BEST_FIT_PLANE),
        ("Iterative closest point (ICP)", OrientationMethod.REFERENCE_ICP),
    ],
}


class ControlPanel(QWidget):
    """
    Side panel controlling orientation parameters and batch queue.
    
    Signals
    -------
    mode_changed(str)
        Emitted when user switches mode tab.
    method_changed(OrientationMethod)
        Emitted when orientation method dropdown changes.
    slider_changed(float, float, float)
        Emitted when any Euler angle slider moves (pitch, roll, yaw).
    apply_clicked()
        Emitted when "Apply" is pressed.
    reset_clicked()
        Emitted when "Reset" is pressed.
    batch_run_clicked()
        Emitted when "Run batch" is pressed.
    files_added(list[Path])
        Emitted when user selects files via dialog.
    threshold_changed(float)
        Emitted when residual threshold spinbox changes.
    """

    mode_changed = Signal(str)
    method_changed = Signal(object)  # OrientationMethod
    slider_changed = Signal(float, float, float)
    apply_clicked = Signal()
    reset_clicked = Signal()
    batch_run_clicked = Signal()
    files_added = Signal(list)
    threshold_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(260)
        self.setStyleSheet(STYLE_SHEET)
        self._current_mode = "Semi-automatic"
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Mode tabs ---
        self.tab_bar = QTabBar()
        self.tab_bar.addTab("Manual")
        self.tab_bar.addTab("Semi-auto")
        self.tab_bar.addTab("Automatic")
        self.tab_bar.setCurrentIndex(1)
        self.tab_bar.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tab_bar)

        # --- Method group ---
        method_group = QGroupBox("Orientation method")
        mg_layout = QVBoxLayout(method_group)

        self.method_combo = QComboBox()
        self._populate_methods("Semi-automatic")
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        mg_layout.addWidget(self.method_combo)

        # Reference mesh selector (only for ICP)
        self.ref_row = QHBoxLayout()
        self.ref_label = QLabel("Reference:")
        self.ref_label.setStyleSheet("font-size: 11px; color: #888;")
        self.ref_btn = QPushButton("Select .ply")
        self.ref_btn.setFixedHeight(26)
        self.ref_btn.clicked.connect(self._select_reference)
        self.ref_row.addWidget(self.ref_label)
        self.ref_row.addWidget(self.ref_btn)
        mg_layout.addLayout(self.ref_row)
        self._set_ref_visible(False)

        layout.addWidget(method_group)

        # --- Fine adjustment group ---
        adj_group = QGroupBox("Fine adjustment")
        ag_layout = QVBoxLayout(adj_group)

        self.pitch_slider, self.pitch_value = self._make_slider("Pitch (X)", ag_layout)
        self.roll_slider, self.roll_value = self._make_slider("Roll (Y)", ag_layout)
        self.yaw_slider, self.yaw_value = self._make_slider("Yaw (Z)", ag_layout, range_deg=180)

        layout.addWidget(adj_group)

        # --- Action buttons ---
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(8, 6, 8, 6)

        self.btn_reset = QPushButton("Reset")
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setObjectName("primary")

        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_apply.clicked.connect(self.apply_clicked.emit)

        btn_row.addWidget(self.btn_reset)
        btn_row.addWidget(self.btn_apply)
        layout.addLayout(btn_row)

        # --- Batch group ---
        batch_group = QGroupBox("Batch queue")
        bg_layout = QVBoxLayout(batch_group)

        # Threshold
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Flag threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.5, 45.0)
        self.threshold_spin.setValue(5.0)
        self.threshold_spin.setSuffix("°")
        self.threshold_spin.setDecimals(1)
        self.threshold_spin.valueChanged.connect(self.threshold_changed.emit)
        thresh_row.addWidget(self.threshold_spin)
        bg_layout.addLayout(thresh_row)

        # File list
        self.batch_list = QListWidget()
        self.batch_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.batch_list.setMinimumHeight(100)
        bg_layout.addWidget(self.batch_list)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        bg_layout.addWidget(self.progress_bar)

        # Batch buttons
        batch_btn_row = QHBoxLayout()
        self.btn_add_files = QPushButton("Add files")
        self.btn_run_batch = QPushButton("Run batch")
        self.btn_run_batch.setObjectName("primary")

        self.btn_add_files.clicked.connect(self._add_files_dialog)
        self.btn_run_batch.clicked.connect(self.batch_run_clicked.emit)

        batch_btn_row.addWidget(self.btn_add_files)
        batch_btn_row.addWidget(self.btn_run_batch)
        bg_layout.addLayout(batch_btn_row)

        layout.addWidget(batch_group)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Slider factory
    # ------------------------------------------------------------------

    def _make_slider(self, label: str, parent_layout, range_deg: int = 45):
        """Create a labeled slider with value readout."""
        row = QHBoxLayout()

        lbl = QLabel(label)
        lbl.setFixedWidth(65)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(-range_deg * 10, range_deg * 10)  # 0.1° steps
        slider.setValue(0)

        value_label = QLabel("0.0°")
        value_label.setFixedWidth(45)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value_label.setStyleSheet("font-family: monospace; font-size: 12px; color: #666;")

        def on_change(val):
            deg = val / 10.0
            value_label.setText(f"{deg:.1f}°")
            self.slider_changed.emit(
                self.pitch_slider.value() / 10.0 if hasattr(self, 'pitch_slider') else 0,
                self.roll_slider.value() / 10.0 if hasattr(self, 'roll_slider') else 0,
                self.yaw_slider.value() / 10.0 if hasattr(self, 'yaw_slider') else 0,
            )

        slider.valueChanged.connect(on_change)

        row.addWidget(lbl)
        row.addWidget(slider, stretch=1)
        row.addWidget(value_label)
        parent_layout.addLayout(row)

        return slider, value_label

    # ------------------------------------------------------------------
    # Batch list management
    # ------------------------------------------------------------------

    def add_batch_items(self, filepaths: List[Path]):
        """Add files to the batch list widget."""
        for fp in filepaths:
            item = QListWidgetItem(f"  ⏳  {fp.name}")
            item.setData(Qt.UserRole, str(fp))
            item.setToolTip(str(fp))
            self.batch_list.addItem(item)

    def update_batch_item(self, index: int, status: str, residual: Optional[float] = None):
        """
        Update visual status of a batch item.
        
        Parameters
        ----------
        index : int
            Item index in the list.
        status : str
            One of: 'pending', 'processing', 'done', 'flagged', 'error'.
        residual : float, optional
            Residual angle to display.
        """
        if index >= self.batch_list.count():
            return

        item = self.batch_list.item(index)
        name = Path(item.data(Qt.UserRole)).name
        residual_str = f" ({residual:.1f}°)" if residual is not None else ""

        icons = {
            "pending": "⏳",
            "processing": "▶️",
            "done": "✅",
            "flagged": "⚠️",
            "error": "❌",
        }
        icon = icons.get(status, "⏳")
        item.setText(f"  {icon}  {name}{residual_str}")

        colors = {
            "pending": QColor("#888"),
            "processing": QColor("#378ADD"),
            "done": QColor("#1D9E75"),
            "flagged": QColor("#BA7517"),
            "error": QColor("#E24B4A"),
        }
        item.setForeground(colors.get(status, QColor("#888")))

    def set_progress(self, current: int, total: int):
        """Update progress bar."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        if current >= total:
            self.progress_bar.setVisible(False)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_tab_changed(self, index: int):
        modes = ["Manual", "Semi-automatic", "Automatic"]
        self._current_mode = modes[index]
        self._populate_methods(self._current_mode)
        self.mode_changed.emit(self._current_mode)

    def _on_method_changed(self, index: int):
        options = METHOD_OPTIONS.get(self._current_mode, [])
        if 0 <= index < len(options):
            method = options[index][1]
            self.method_changed.emit(method)
            self._set_ref_visible(method == OrientationMethod.REFERENCE_ICP)

    def _populate_methods(self, mode: str):
        self.method_combo.blockSignals(True)
        self.method_combo.clear()
        for label, _ in METHOD_OPTIONS.get(mode, []):
            self.method_combo.addItem(label)
        self.method_combo.blockSignals(False)
        self._on_method_changed(0)

    def _on_reset(self):
        """Reset sliders to zero and emit."""
        self.pitch_slider.setValue(0)
        self.roll_slider.setValue(0)
        self.yaw_slider.setValue(0)
        self.reset_clicked.emit()

    def _add_files_dialog(self):
        """Open file dialog to add meshes."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select mesh files",
            "",
            "Mesh files (*.ply *.stl *.obj *.off);;All files (*)",
        )
        if files:
            paths = [Path(f) for f in files]
            self.add_batch_items(paths)
            self.files_added.emit(paths)

    def _select_reference(self):
        """Select reference mesh for ICP."""
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Select reference mesh",
            "",
            "Mesh files (*.ply *.stl *.obj);;All files (*)",
        )
        if f:
            self.ref_btn.setText(Path(f).name)
            self.ref_btn.setToolTip(f)

    def _set_ref_visible(self, visible: bool):
        self.ref_label.setVisible(visible)
        self.ref_btn.setVisible(visible)
