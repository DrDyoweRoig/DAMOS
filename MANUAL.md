# DAMOS — User Manual

**Dental Analysis and Morphometry Open Suite**  
Author: Albert Epitie  
Version: 0.1

---

## Table of Contents

1. [Interface Overview](#1-interface-overview)
2. [Loading Meshes](#2-loading-meshes)
3. [MeshOrient](#3-meshorient)
4. [AutoPlaneCut](#4-autoplancut)
5. [PolyTrim](#5-polytrim)
6. [AutoLMK](#6-autolmk)
7. [AutoMorph](#7-automorph)
8. [FFEI](#8-ffei)
9. [Keyboard Shortcuts](#9-keyboard-shortcuts)
10. [Recommended Workflow](#10-recommended-workflow)
11. [File Formats](#11-file-formats)
12. [Citation](#12-citation)

---

## 1. Interface Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  [Home] [MeshOrient] [AutoPlaneCut] [PolyTrim] [AutoLMK]        │  ← Tab bar
│         [AutoMorph]  [FFEI]                      specimen [1/N]  │
├──────────────┬─────────────────────────────┬────────────────────┤
│              │                             │                    │
│  File list   │     3D Viewer               │  Control panel     │
│              │                             │  (active module)   │
│  [Load]      │  [Reset][Occlusal][Front]   │                    │
│  [Folder]    │  [Side][Axes][Edges]        │                    │
│              │                             │                    │
│  file1.ply   │                             │                    │
│  file2.ply   │                             │                    │
│  ...         │                             │                    │
│  [◀][▶]      │                             │                    │
├──────────────┴─────────────────────────────┴────────────────────┤
│  Status bar                                                      │
└──────────────────────────────────────────────────────────────────┘
```

### Panels

| Panel | Description |
|---|---|
| **Tab bar** | Switch between modules. The active tab is highlighted in blue. |
| **File list** | Lists all loaded meshes. Click to select; use ◀ ▶ or arrow keys to navigate. |
| **3D Viewer** | Interactive PyVista viewport. Left-drag to rotate, right-drag to pan, scroll to zoom. |
| **Control panel** | Right-side panel — changes content depending on the active module. |
| **Status bar** | Shows the current file name, face count, and module messages. |

### Viewer toolbar buttons

| Button | Action |
|---|---|
| Reset | Reset camera to default view |
| Occlusal | Top-down view along –Z (standard dental view) |
| Front | Front view (XZ plane) |
| Side | Side view (YZ plane) |
| Axes | Toggle world-axis indicator |
| Edges | Toggle mesh edge wireframe overlay |

---

## 2. Loading Meshes

**File › Load Files…** (`Ctrl+O`) — open one or multiple mesh files.  
**File › Load Folder…** (`Ctrl+Shift+O`) — load all meshes in a folder at once.

Supported formats: **.ply**, **.obj**, **.stl**, **.vtk**, **.vtp**

Loaded meshes appear in the file list on the left. The first mesh is selected and displayed automatically. Use ◀ ▶ buttons or the Left/Right arrow keys to navigate between specimens.

> **Important:** All modules expect the occlusal surface facing **+Z** and crown geometry only (no root, mandible or partial scan). Run MeshOrient first if your meshes are not already in standard orientation.

---

## 3. MeshOrient

**Purpose:** Orient meshes so that the occlusal surface faces +Z.  
This step is required before running AutoPlaneCut, AutoLMK, AutoMorph and FFEI.

### Orientation modes

#### Manual
Rotate the mesh freely using three sliders (X, Y, Z rotation in degrees).  
Use the 3D viewer to inspect the result.  
When satisfied, click **Set as reference** to save the orientation as the reference for automatic alignment.

#### Semi-automatic
Applies PCA + ICP alignment to the loaded reference, then allows fine-tuning with the sliders.  
Useful when meshes are roughly oriented but need minor correction.

#### Automatic (batch)
Aligns each mesh in the file list to the reference using PCA followed by ICP (Iterative Closest Point) minimising the RMS distance to the reference surface.

### Controls

| Control | Description |
|---|---|
| Mode selector | Manual / Semi-auto / Automatic |
| X / Y / Z sliders | Manual rotation around each axis (degrees) |
| Set as reference | Save current mesh orientation as alignment target |
| Load reference | Load a previously saved reference mesh (.ply) |
| Threshold (mm) | ICP convergence threshold; specimens exceeding this error are flagged |
| Output folder | Destination for oriented meshes |
| Run Batch | Process all loaded meshes |
| Abort | Stop batch at next specimen |

### Output
Oriented meshes are saved with the suffix `_oriented.ply` in the selected output folder.  
Specimens whose ICP error exceeds the threshold are **flagged** (highlighted in orange in the file list) for manual review.

### Quality indicator
After batch, a summary shows how many specimens were oriented successfully and how many were flagged.

---

## 4. AutoPlaneCut

**Purpose:** Detect the occlusal plane and trim the mesh at that plane, removing sub-occlusal geometry.

### Detection methods

| Method | Description |
|---|---|
| **Robust** | Fits a robust plane to the upper percentile of Z values. Works well for most morphologies. Adjust the percentile parameter if the cut is too high or too low. |
| **Cusp-bounded minimum** | Detects the N principal cusps using MorphologyEngine, builds their convex hull in XY, finds the lowest Z inside that polygon, and cuts there. Adapts to any morphology. |
| **Custom Z** | Enter a Z-value directly and cut at that plane. Useful when you know the exact crown height. |

### Morphology presets

The **Preset** selector adjusts internal detection parameters for the expected cusp count and morphology:

| Preset | Typical use |
|---|---|
| Human bunodont | *Homo sapiens* molars (4–5 cusps) |
| Cercopithecid | Bilophodont molars (cercopithecines, colobines) |
| Hominoid | Complex ape molars |
| Custom | Manual parameter entry |

### Preview and manual adjustment

1. Click **Preview** to see the detected cut plane (shown as a translucent disc in the viewer).
2. Use **+** / **–** buttons or edit the Z value directly to nudge the plane up or down.
3. Click **Apply cut** to save the trimmed mesh.

### Batch processing

Set an **Input folder** and **Output folder**, select method and preset, and click **Run Batch**.  
Each mesh is cut and saved with the suffix `_cut.ply`.

---

## 5. PolyTrim

**Purpose:** Reduce polygon count to a standardised target before analysis, ensuring all specimens have comparable mesh resolution.

### Controls

| Control | Description |
|---|---|
| Target faces | Desired number of triangular faces after decimation |
| Smooth iterations | Laplacian smoothing passes applied before decimation (0 = none) |
| Preview | Apply decimation to the current mesh and show the result |
| Apply & Save | Save the decimated mesh |
| Run Batch | Process all loaded meshes |

### Recommendations

- **10 000–20 000 faces**: sufficient for DNE, OPCR, RFI and most shape metrics.  
- **5 000 faces**: faster batch processing; acceptable for comparative studies.  
- **> 50 000 faces**: maximum precision; appropriate for fine-detail PCV or landmark placement.  
- Smoothing should be applied sparingly (0–3 iterations). Excessive smoothing eliminates biological signal.

### Output
Files are saved with the suffix `_trimmed.ply`. A batch log records face counts before and after for each specimen.

---

## 6. AutoLMK

**Purpose:** Automatically detect and manually edit 3D morphological landmarks — cusps, intercuspal points, the geometric centre, and semilandmarks along the occlusal margin.

### Landmark types

| Type | Description |
|---|---|
| **Cusps** | Local maxima of the occlusal surface (primary cusps). Detected automatically. |
| **Intercuspal** | Points between adjacent cusps at minimum Z. Detected automatically. |
| **Centre** | Geometric centroid of the cusp set. Computed automatically. |
| **Semilandmarks** | Equally spaced points along the occlusal perimeter. Count defined by user. |

### Detection parameters

| Parameter | Description |
|---|---|
| Min cusp height (mm) | Minimum elevation above the occlusal plane for a peak to qualify as a cusp |
| Min cusp separation (mm) | Minimum XY distance between two cusps |
| Smooth before detect | Apply light smoothing before detection (avoids noise peaks) |
| Semilandmark count | Number of equidistant semilandmarks to place around the crown margin |

### Manual editing

After automatic detection:
- **Click a landmark** in the list to highlight it in the viewer.
- **Click "Pick"** then click a new location in the 3D viewer to move the landmark manually.
- **Delete** removes a selected landmark.
- **Add cusp** enters pick mode — click anywhere on the mesh to add a new cusp point.

### Batch processing

Set an **Input folder** and **Output folder** and click **Run Batch**.  
Each specimen produces:
- `*_landmarks.csv` — X, Y, Z coordinates for all landmarks
- `ALL_Landmarks.csv` — combined table for all specimens

### Output CSV columns

```
file, label, type, x, y, z
```
Where `type` is `cusp`, `intercusp`, `center`, or `semi_N`.

---

## 7. AutoMorph

**Purpose:** Compute a comprehensive set of occlusal topographic metrics for single meshes or batch folders.

### Mesh preparation controls

| Control | Description |
|---|---|
| Smooth iterations | Laplacian passes before metric computation (0–100) |
| Target faces | Optionally decimate before computing (0 = use full mesh) |

### Metrics computed

| Metric | Description | Reference |
|---|---|---|
| **Mean slope** (°) | Mean angle of each face normal relative to the vertical (+Z). Measures surface steepness. | Ungar & M'Kirera 2003 |
| **Slope SD** (°) | Standard deviation of face slopes. Measures topographic heterogeneity. | |
| **Simple Relief** (mm) | Max Z − Min Z. Vertical height of the occlusal surface. | |
| **Area 3D** (mm²) | True 3D surface area. | |
| **Area 2D** (mm²) | Projected area onto the XY plane. | |
| **Relief** | 100 × (Area 3D / Area 2D). Percentage surface complexity. | M'Kirera & Ungar 2003 |
| **RFI₁** | Area 3D / Area 2D. Relief Feature Index. | Boyer et al. 2010 |
| **RFI₂** | ln(√RFI₁). Log-scaled relief index. | Boyer et al. 2010 |
| **DNE** | Dirichlet Normal Energy. Measures overall surface curvature. High = sharp cusps; Low = flat/worn. | Bunn et al. 2011 |
| **OPCR** | Orientation Patch Count Rotated. Counts distinct orientation regions. Reflects dietary breadth. | Boyer et al. 2010 |
| **MD** (mm) | Mesiodistal bounding-box length. Requires oriented mesh. | |
| **BL** (mm) | Buccolingual bounding-box width. Requires oriented mesh. | |
| **PCV Mean** | Mean Projected Cavity Visibility. 0 = all foveas; 1 = all cusps. Predicts contact probability. | Berthaume et al. 2019 |
| **PCV Min** | Minimum PCV value. Depth of the most protected fovea. | Berthaume et al. 2019 |

### Display styles

| Style | Description |
|---|---|
| Elevation | Height colourmap (gist_earth). Default view. |
| Slope | Face slope in degrees (turbo colourmap). |
| Shaded | Neutral bone-like shading. |
| DNE | Per-face DNE energy density (inferno colourmap). |
| Orientation | OPCR orientation bins — each colour is a distinct orientation patch (tab10). |
| PCV Skyline | Per-vertex PCV value (RdYlGn: red = fovea, green = cusp). |

### Single-mesh workflow

1. Load a mesh and navigate to it in the file list.
2. Set smooth/decimate parameters if desired.
3. Click **Compute Metrics**.
4. Results appear in the table; change display style to visualise different scalar fields.
5. Click **Save current CSV** to export the metric table for this specimen.

### Batch workflow

1. Set **Input folder** (folder containing *.ply / *.obj files).
2. Set **Output folder** (where CSVs will be saved).
3. Click **Run Batch**.
4. Progress is shown in the log. Each specimen produces a `*_topography.csv`.
5. A combined `ALL_Topography.csv` is written when the batch completes.

### Interpreting metrics

| Observation | Likely interpretation |
|---|---|
| High DNE, low PCV mean | Fresh, unworn tooth with sharp cusps |
| Low DNE, high PCV mean | Worn tooth — cusps flattened, foveas exposed |
| High OPCR | Diverse dietary item processing (many orientation patches) |
| Low OPCR | Specialised processing (few dominant surfaces) |
| RFI₁ >> 1 | High topographic relief — effective at fracturing hard food |

---

## 8. FFEI

**Purpose:** Compute the Food-Flow Estimation Index and associated watershed-based drainage metrics. The FFEI quantifies what proportion of the occlusal area directs food material toward the central fovea.

### Theoretical background

FFEI is based on the hydrological analogy of Ungar & M'Kirera (2003): occlusal topography directs food items downslope. Basins draining to the central fovea contribute to food retention and processing efficiency. The FFEI is the fraction of total occlusal area covered by these "productive" basins.

### Controls

| Control | Description |
|---|---|
| Min basin area (mm²) | Minimum watershed basin size to be included |
| Smooth before compute | Apply smoothing passes before watershed segmentation |
| Central fovea radius | Search radius (mm) for the central fovea detection |
| Run | Compute FFEI for the current mesh |
| Run Batch | Process all loaded meshes |

### Metrics computed

| Metric | Description |
|---|---|
| **FFEI** | Fraction of occlusal area draining to the central fovea (0–1) |
| **n_basins** | Total number of watershed drainage basins |
| **n_cusps** | Detected cusp count |
| **n_depressions** | Detected fovea/depression count |
| **slope_mean** (°) | Mean face slope across the occlusal surface |
| **slope_sd** (°) | Standard deviation of face slopes |
| **z_range** (mm) | Vertical range (max Z − min Z) |
| **cusp_fovea_depth** | Mean cusp Z − deepest fovea Z. Measures occlusal relief depth. |
| **basin_area_gini** | Gini coefficient of basin sizes (0 = all basins equal; 1 = one basin dominates) |
| **primary_fovea_depth** (mm) | Depth of the deepest detected fovea |

### Display styles

| Style | Description |
|---|---|
| Basins | Watershed regions colour-coded by basin ID |
| Height | Z-elevation colourmap |
| Slope | Face slope magnitude colourmap |
| Shaded | Neutral shading |

### Interpreting FFEI

| FFEI value | Interpretation |
|---|---|
| > 0.6 | High food-retention capacity; efficient at processing soft or sticky foods |
| 0.3 – 0.6 | Moderate retention; generalist processing |
| < 0.3 | Low retention; food escapes rapidly — effective at slicing hard or fibrous items |

### Batch output

Each specimen produces a `*_ffei.csv`. A combined `ALL_FFEI.csv` is written at batch completion.

---

## 9. Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `←` / `→` | Navigate to previous / next specimen |
| `Ctrl+O` | Load files |
| `Ctrl+Shift+O` | Load folder |
| `Ctrl+1` | Occlusal view |
| `Ctrl+R` | Reset camera |
| `Ctrl+Q` | Quit |

---

## 10. Recommended Workflow

The standard pipeline for a comparative morphometric study:

### Step 1 — Prepare meshes (MeshOrient)
Orient all specimens to +Z occlusal. Use a well-oriented representative specimen as reference. Run batch automatic alignment. Review flagged specimens manually.

### Step 2 — Trim occlusal surface (AutoPlaneCut)
Remove sub-occlusal geometry. Use the **Cusp-bounded minimum** method for complex morphologies or **Robust** for homogeneous samples. Verify cut plane visually with Preview before batch.

### Step 3 — Standardise polygon count (PolyTrim)
Decimate all meshes to the same target face count (e.g. 10 000). This ensures that metrics such as DNE and OPCR are comparable across specimens of different original resolution.

### Step 4 — Place landmarks (AutoLMK)
Run automatic detection. Verify landmark placement on a representative subset. Correct any misplaced landmarks manually. Export batch CSV.

### Step 5 — Compute topographic metrics (AutoMorph)
Run batch with standardised smoothing (0–2 iterations recommended). Export `ALL_Topography.csv` for statistical analysis.

### Step 6 — Functional analysis (FFEI)
Run batch FFEI on the same prepared meshes. Export `ALL_FFEI.csv`.

### Step 7 — Statistical analysis
Combine `ALL_Topography.csv` and `ALL_FFEI.csv` in R or Python for multivariate analysis, PCA, discriminant analysis, etc.

---

## 11. File Formats

### Input
| Format | Extension | Notes |
|---|---|---|
| Stanford Polygon | `.ply` | Recommended. Binary or ASCII. |
| Wavefront OBJ | `.obj` | Supported. |
| STL | `.stl` | Supported. |
| VTK Legacy | `.vtk` | Supported. |
| VTK XML | `.vtp` | Supported. |

### Output
| File | Content |
|---|---|
| `*_oriented.ply` | Mesh after MeshOrient |
| `*_cut.ply` | Mesh after AutoPlaneCut |
| `*_trimmed.ply` | Mesh after PolyTrim |
| `*_landmarks.csv` | Landmark coordinates (AutoLMK) |
| `*_topography.csv` | All topographic metrics (AutoMorph) |
| `*_ffei.csv` | All FFEI metrics |
| `ALL_Landmarks.csv` | Combined landmark table (batch) |
| `ALL_Topography.csv` | Combined topography table (batch) |
| `ALL_FFEI.csv` | Combined FFEI table (batch) |

---

## 12. Citation

If you use DAMOS in published research, please cite it. Proper attribution helps the author track the software's use and supports its continued development.

> **Epitie, A.** (*year*). *DAMOS — Dental Analysis and Morphometry Open Suite* (version X.X).  
> [Software]. Available from: [repository URL]

BibTeX:

```bibtex
@software{epitie_damos,
  author  = {Epitie, Albert},
  title   = {{DAMOS} -- {D}ental {A}nalysis and {M}orphometry {O}pen {S}uite},
  year    = {2025},
  note    = {Available at: [repository URL]}
}
```

### Citing the underlying metrics

When reporting specific metrics, please also cite the original methodological papers:

| Metric | Citation |
|---|---|
| DNE | Bunn, J.M., Boyer, D.M., Lipman, Y., St. Clair, E.M., Jernvall, J., Daubechies, I. (2011). Comparing Dirichlet normal surface energy of tooth crowns. *American Journal of Physical Anthropology*, 145(2), 247–261. |
| OPCR | Boyer, D.M., Evans, A.R., Jernvall, J. (2010). Evidence of dietary differentiation among insectivorous micromammals. *American Journal of Physical Anthropology*, 141(3), 376–388. |
| RFI | Boyer, D.M. (2008). Relief index of second mandibular molars. *American Journal of Physical Anthropology*, 136(3), 307–328. |
| PCV | Berthaume, M.A. et al. (2019). Dental topography and textural analysis. *Journal of Human Evolution*, 129, 109–125. |
| FFEI | Ungar, P.S. & M'Kirera, F. (2003). A solution to the worn tooth conundrum. *PNAS*, 100(7), 3874–3877. |

---

*DAMOS — Dental Analysis and Morphometry Open Suite*  
*© Albert Epitie. All rights reserved.*
