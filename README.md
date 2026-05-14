# DAMOS — Dental Analysis and Morphometry Open Suite

**DAMOS** is a standalone desktop application for the quantitative analysis of 3D dental surface meshes. It integrates six modules covering the full pipeline from mesh orientation and preparation to morphometric characterisation and functional analysis.

---

## Modules

| Module | Category | Function |
|---|---|---|
| **MeshOrient** | Preprocessing | Automatic, semi-automatic and manual mesh orientation |
| **AutoPlaneCut** | Preprocessing | Occlusal plane detection and mesh trimming |
| **PolyTrim** | Preprocessing | Polygon decimation and mesh standardisation |
| **AutoLMK** | Analysis | Automatic and manual 3D landmark placement |
| **AutoMorph** | Analysis | Topographic metrics: DNE, OPCR, RFI, PCV, slope |
| **FFEI** | Analysis | Food-Flow Estimation Index — watershed-based functional analysis |

---

## Requirements

- Python ≥ 3.9
- PySide6 ≥ 6.5
- PyVista ≥ 0.43
- pyvistaqt ≥ 0.11
- NumPy ≥ 1.24

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Installation and Run

```bash
# Clone or unzip the repository
cd DAMOS

# Install dependencies
pip install -r requirements.txt

# Launch
python damos.py
```

Or use the standalone executable (Windows):

```
dist/DAMOS/DAMOS.exe
```

---

## Mesh requirements

- **Format**: PLY or OBJ
- **Orientation**: occlusal surface facing **+Z** (required by AutoPlaneCut, AutoLMK, AutoMorph, FFEI)
- **Coverage**: isolated crown; no root, mandible, or partial scans
- **Coordinate units**: millimetres (recommended)

---

## Typical workflow

```
Load meshes
    │
    ▼
MeshOrient ──► AutoPlaneCut ──► PolyTrim
                                    │
                                    ▼
                              AutoLMK  ──►  AutoMorph
                                                │
                                                ▼
                                              FFEI
```

---

## Output files

| Module | Output |
|---|---|
| MeshOrient | Oriented mesh (`*_oriented.ply`) |
| AutoPlaneCut | Trimmed mesh (`*_cut.ply`) |
| PolyTrim | Decimated mesh (`*_trimmed.ply`) |
| AutoLMK | Landmark CSV per specimen + batch combined CSV |
| AutoMorph | Metric CSV per specimen + `ALL_Topography.csv` |
| FFEI | FFEI metric CSV per specimen + batch combined CSV |

---

## Citation

If you use DAMOS in your research, you are kindly requested to cite it.  
Proper citation allows the author to track the software's use and justify its continued development.

> Dyowe Roig, A. E., Martínez, L. M., & Estebaranz Sánchez, F. (2025). *DAMOS — Dental Analysis and Morphometry Open Suite* (v0.1). Zenodo. https://doi.org/10.5281/zenodo.20189381

A suggested BibTeX entry:

```bibtex
@software{epitie_damos,
  author    = {Dyowe Roig, Albert Epitie and Martínez, Laura M. and Estebaranz Sánchez, Ferran},
  title     = {{DAMOS} -- {D}ental {A}nalysis and {M}orphometry {O}pen {S}uite},
  year      = {2025},
  version   = {0.1},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20189381},
  url       = {https://doi.org/10.5281/zenodo.20189381}
}
```


---

## License

© Albert Epitie. All rights reserved.  
Contact the author for permissions regarding redistribution or use in commercial research.

---

## Project structure

```
DAMOS/
├── damos.py               # Entry point
├── ofc_engine.py          # Topography engine (DNE, OPCR, RFI, PCV, slopes)
├── morphology_engine.py   # Landmark detection engine
├── orientation_engine.py  # Mesh orientation engine
├── mesh_exporter.py       # Mesh export utilities
├── batch_processor.py     # Batch processing utilities
├── requirements.txt
├── assets/                # Logo and icons
├── gui/
│   ├── app_state.py       # Shared observable state (Qt signals)
│   ├── home_screen.py     # Welcome screen
│   ├── main_window.py     # Main window
│   ├── style.py           # Global dark stylesheet
│   └── viewer3d.py        # PyVista 3D viewer widget
├── modules/
│   ├── meshorient_panel.py
│   ├── autoplancut_panel.py
│   ├── polytrim_panel.py
│   ├── autolmk_panel.py
│   ├── automorph_panel.py
│   └── ffei_panel.py
└── ffei/                  # FFEI analysis engine
    ├── pipeline.py
    ├── surface_fields.py
    ├── watershed.py
    ├── metrics.py
    ├── escape_channels.py
    ├── detection.py
    └── io.py
```
