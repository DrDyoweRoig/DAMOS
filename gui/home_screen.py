"""
DAMOS – Home / Welcome screen
Larger title, larger cards, centered layout (no sidebar).
"""

import os as _os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QGridLayout, QSizePolicy, QSpacerItem
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap


class ModuleCard(QFrame):
    clicked = Signal(str)

    _COLORS = {
        "meshorient":  "#4A8A6A",
        "autoplancut": "#1D6FA4",
        "polytrim":    "#2A7A5A",
        "autolmk":     "#7A4FA0",
        "automorph":   "#A06030",
        "ffei":        "#0D7A8A",
    }

    _ICONS = {
        "meshorient":  "⟲",
        "autoplancut": "✂",
        "polytrim":    "◈",
        "autolmk":     "◎",
        "automorph":   "∿",
        "ffei":        "⌇",
    }

    def __init__(self, key: str, title: str, description: str, parent=None):
        super().__init__(parent)
        self._key    = key
        self._accent = self._COLORS.get(key, "#1D6FA4")
        self.setObjectName("CardWidget")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(250, 210)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(10)

        # Accent bar
        bar = QFrame()
        bar.setFixedHeight(4)
        bar.setStyleSheet(
            f"background-color: {self._accent}; border-radius: 2px; border: none;"
        )
        layout.addWidget(bar)

        # Icon
        icon_lbl = QLabel(self._ICONS.get(key, "◆"))
        icon_font = QFont()
        icon_font.setPointSize(48)
        icon_lbl.setFont(icon_font)
        icon_lbl.setStyleSheet(f"color: {self._accent}; background: transparent;")
        layout.addWidget(icon_lbl)

        # Title
        title_lbl = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet("color: #E8F0FF; background: transparent;")
        layout.addWidget(title_lbl)

        # Description
        desc_lbl = QLabel(description)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("color: #7A9AB8; font-size: 11px; background: transparent;")
        layout.addWidget(desc_lbl)

        layout.addStretch(1)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._key)
        super().mousePressEvent(event)

    def enterEvent(self, event):
        self.setStyleSheet(
            f"QFrame#CardWidget {{ background-color: #2D3F5E; "
            f"border: 1px solid {self._accent}; border-radius: 8px; }}"
        )
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet("")
        super().leaveEvent(event)



class HomeScreen(QWidget):
    navigate = Signal(str)

    _MODULES = [
        {
            "key":         "meshorient",
            "title":       "MeshOrient",
            "description": "Automatic and manual mesh orientation "
                           "with occlusal plane alignment to +Z",
        },
        {
            "key":         "autoplancut",
            "title":       "AutoPlaneCut",
            "description": "Automatic occlusal plane detection and batch mesh cutting",
        },
        {
            "key":         "polytrim",
            "title":       "PolyTrim",
            "description": "Batch polygon decimation and mesh standardisation",
        },
        {
            "key":         "autolmk",
            "title":       "AutoLMK",
            "description": "Automatic and manual 3D landmark placement",
        },
        {
            "key":         "automorph",
            "title":       "AutoMorph",
            "description": "Slope, relief, RFI, DNE, OPCR and bounding-box metrics",
        },
        {
            "key":         "ffei",
            "title":       "FFEI",
            "description": "Food-Flow Estimation Index: watershed-based "
                           "functional analysis of occlusal drainage basins",
        },
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_logo_pix') and self._logo_pix is not None:
            max_w = int(self.width() * 0.40)
            max_h = 250
            pix = self._logo_pix.scaled(
                max_w, max_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self._logo_label.setPixmap(pix)
            self._logo_label.setFixedHeight(pix.height())

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Centered content area ─────────────────────────
        center_widget = QWidget()
        center_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        cv = QVBoxLayout(center_widget)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)

        # Vertical centering
        cv.addStretch(1)

        # ── Title block ───────────────────────────────────
        title_block = QWidget()
        tbl = QVBoxLayout(title_block)
        tbl.setContentsMargins(0, 0, 0, 0)
        tbl.setSpacing(8)
        tbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        logo_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "assets", "logo_damos.png"
        )
        title = QLabel()
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet("background: transparent;")
        self._logo_label = title
        if _os.path.exists(logo_path):
            self._logo_pix = QPixmap(logo_path)
            pix = self._logo_pix.scaled(
                560, 150,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            title.setPixmap(pix)
            title.setMinimumHeight(pix.height())
        else:
            self._logo_pix = None
            font = QFont("Segoe UI"); font.setBold(True); font.setPointSize(72)
            title.setFont(font)
            title.setText("DAMOS")
            title.setStyleSheet("color: #F0F6FF; background: transparent;")
        title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        tbl.addWidget(title)

        # Thin accent line under title
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFixedWidth(120)
        line.setStyleSheet("color: #1D6FA4; background-color: #1D6FA4; "
                           "border: none; height: 2px; margin-top: 8px;")
        line.setFixedHeight(2)
        line_wrap = QHBoxLayout()
        line_wrap.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        line_wrap.addWidget(line)
        tbl.addLayout(line_wrap)

        cv.addWidget(title_block)
        cv.addSpacing(56)

        # ── Module cards ──────────────────────────────────

        # Section label helper
        def _section_label(text):
            lbl = QLabel(text)
            lbl_font = QFont()
            lbl_font.setPointSize(9)
            lbl_font.setBold(True)
            lbl.setFont(lbl_font)
            lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            lbl.setStyleSheet(
                "color: #4A6A88; letter-spacing: 2px; background: transparent;"
            )
            return lbl

        # Card row helper
        def _card_row(modules):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            row_layout.setSpacing(20)
            row_layout.setContentsMargins(0, 0, 0, 0)
            for mod in modules:
                card = ModuleCard(
                    key=mod["key"],
                    title=mod["title"],
                    description=mod["description"],
                )
                card.clicked.connect(self.navigate)
                row_layout.addWidget(card)
            return row_widget

        # Preprocessing row (first 3 modules)
        cv.addWidget(_section_label("PREPROCESSING"))
        cv.addSpacing(10)
        cv.addWidget(_card_row(self._MODULES[:3]))
        cv.addSpacing(24)

        # Analysis row (last 3 modules)
        cv.addWidget(_section_label("ANALYSIS"))
        cv.addSpacing(10)
        cv.addWidget(_card_row(self._MODULES[3:]))
        cv.addSpacing(32)

        # ── Footer ────────────────────────────────────────
        note = QLabel(
            "Load meshes via  File → Load Files / Load Folder  "
            "and navigate specimens with  ◀ Prev  /  Next ▶"
        )
        note.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        note.setStyleSheet("color: #3A5A78; font-size: 12px; background: transparent;")
        cv.addWidget(note)

        cv.addStretch(2)

        root.addWidget(center_widget)