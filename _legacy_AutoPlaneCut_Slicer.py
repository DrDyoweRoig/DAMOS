import os
import math
import vtk
import qt
import ctk
import slicer

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)


#
# Module
#
class AutoPlaneCut(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)

        parent.title = "AutoPlaneCut"

        # Icon
        iconPath = os.path.join(os.path.dirname(__file__), "AutoPlaneCut.png")
        if os.path.exists(iconPath):
            parent.icon = qt.QIcon(iconPath)

        parent.categories = ["Surface Models"]
        parent.dependencies = []
        parent.contributors = ["Albert"]
        parent.helpText = """
Batch cutting of isolated oriented tooth meshes.

Assumptions
- Meshes are already isolated.
- The occlusal surface is oriented towards +Z.
- The module estimates a cutting plane from the lowest occlusal depression.
- The module keeps the part above the cutting plane.

Parameters
- Input folder: Folder containing the meshes to process.
- Input format: File format of the input meshes (PLY or OBJ).
- Occlusal band: Percentage of total crown height taken from the top of the mesh to define the occlusal search region.
- Low-Z candidate band: Percentage of the lowest points within the occlusal band used as depression candidates.
- Central area around depression: Percentage of the closest points around the detected occlusal center or depression center used to refine the cut estimate.
- Method: Strategy used to estimate the cutting level.
  - Absolute lowest fovea: Uses the single lowest point detected within the occlusal candidate region. This may correspond to a central or lateral fovea.
  - Lowest central fovea: Uses the lowest point within the central region of the occlusal band to avoid peripheral depressions.
  - Robust low depression: Uses a low percentile of clustered depression points to reduce sensitivity to isolated outliers.
- Percentile: Percentile used only in Robust low depression mode.
- Z offset downward: Distance subtracted from the detected depression level so the cutting plane is moved slightly downward.
- Keep largest component: Keeps only the largest connected mesh component after cutting.
- Output suffix: Text added to the output filename.
"""
        parent.acknowledgementText = (
            "AutoPlaneCut for batch tooth cropping in 3D Slicer."
        )


#
# Widget
#
class AutoPlaneCutWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.logic = AutoPlaneCutLogic()
        self.inputFolderPath = ""

        #
        # INPUT
        #
        inputCollapsibleButton = ctk.ctkCollapsibleButton()
        inputCollapsibleButton.text = "Input"
        inputCollapsibleButton.collapsed = False
        self.layout.addWidget(inputCollapsibleButton)

        inputFormLayout = qt.QFormLayout(inputCollapsibleButton)

        # Input folder button + path display
        inputFolderWidget = qt.QWidget()
        inputFolderLayout = qt.QHBoxLayout(inputFolderWidget)
        inputFolderLayout.setContentsMargins(0, 0, 0, 0)
        inputFolderLayout.setSpacing(6)

        self.inputFolderSelectButton = qt.QPushButton("Select input folder")
        self.inputFolderSelectButton.toolTip = "Choose the folder containing the meshes to process"

        self.inputFolderLineEdit = qt.QLineEdit()
        self.inputFolderLineEdit.readOnly = True
        self.inputFolderLineEdit.placeholderText = "No folder selected"

        inputFolderLayout.addWidget(self.inputFolderSelectButton)
        inputFolderLayout.addWidget(self.inputFolderLineEdit)

        inputFormLayout.addRow("Input folder:", inputFolderWidget)

        # Input format
        self.formatComboBox = qt.QComboBox()
        self.formatComboBox.addItem("PLY")
        self.formatComboBox.addItem("OBJ")
        inputFormLayout.addRow("Input format:", self.formatComboBox)

        #
        # CUT DETECTION
        #
        cutCollapsibleButton = ctk.ctkCollapsibleButton()
        cutCollapsibleButton.text = "Cut detection"
        cutCollapsibleButton.collapsed = False
        self.layout.addWidget(cutCollapsibleButton)

        cutFormLayout = qt.QFormLayout(cutCollapsibleButton)

        # Occlusal band
        self.occlusalBandSpinBox = qt.QDoubleSpinBox()
        self.occlusalBandSpinBox.minimum = 1.0
        self.occlusalBandSpinBox.maximum = 100.0
        self.occlusalBandSpinBox.singleStep = 1.0
        self.occlusalBandSpinBox.value = 35.0
        self.occlusalBandSpinBox.suffix = " %"
        cutFormLayout.addRow("Occlusal band:", self.occlusalBandSpinBox)

        # Low-Z candidate region
        self.lowBandSpinBox = qt.QDoubleSpinBox()
        self.lowBandSpinBox.minimum = 1.0
        self.lowBandSpinBox.maximum = 100.0
        self.lowBandSpinBox.singleStep = 1.0
        self.lowBandSpinBox.value = 25.0
        self.lowBandSpinBox.suffix = " %"
        cutFormLayout.addRow("Low-Z candidate band:", self.lowBandSpinBox)

        # Central area
        self.centralAreaSpinBox = qt.QDoubleSpinBox()
        self.centralAreaSpinBox.minimum = 1.0
        self.centralAreaSpinBox.maximum = 100.0
        self.centralAreaSpinBox.singleStep = 1.0
        self.centralAreaSpinBox.value = 20.0
        self.centralAreaSpinBox.suffix = " %"
        cutFormLayout.addRow("Central area around depression:", self.centralAreaSpinBox)

        # Method section
        self.methodDescriptionLabel = qt.QLabel("Strategy used to estimate the cutting level")
        self.methodDescriptionLabel.wordWrap = True
        cutFormLayout.addRow("Method:", self.methodDescriptionLabel)

        self.absoluteLowestRadioButton = qt.QRadioButton("Absolute lowest fovea")
        self.lowestCentralRadioButton = qt.QRadioButton("Lowest central fovea")
        self.robustLowRadioButton = qt.QRadioButton("Robust low depression")
        self.robustLowRadioButton.checked = True

        self.methodButtonGroup = qt.QButtonGroup()
        self.methodButtonGroup.addButton(self.absoluteLowestRadioButton)
        self.methodButtonGroup.addButton(self.lowestCentralRadioButton)
        self.methodButtonGroup.addButton(self.robustLowRadioButton)

        methodWidget = qt.QWidget()
        methodLayout = qt.QVBoxLayout(methodWidget)
        methodLayout.setContentsMargins(0, 0, 0, 0)
        methodLayout.setSpacing(2)
        methodLayout.addWidget(self.absoluteLowestRadioButton)
        methodLayout.addWidget(self.lowestCentralRadioButton)
        methodLayout.addWidget(self.robustLowRadioButton)
        cutFormLayout.addRow("", methodWidget)

        # Percentile
        self.percentileSpinBox = qt.QDoubleSpinBox()
        self.percentileSpinBox.minimum = 0.0
        self.percentileSpinBox.maximum = 50.0
        self.percentileSpinBox.singleStep = 1.0
        self.percentileSpinBox.value = 5.0
        self.percentileSpinBox.suffix = " %"
        cutFormLayout.addRow("Percentile:", self.percentileSpinBox)

        # Offset
        self.offsetSpinBox = qt.QDoubleSpinBox()
        self.offsetSpinBox.minimum = 0.0
        self.offsetSpinBox.maximum = 1000.0
        self.offsetSpinBox.singleStep = 0.01
        self.offsetSpinBox.decimals = 4
        self.offsetSpinBox.value = 0.1
        cutFormLayout.addRow("Z offset downward:", self.offsetSpinBox)

        #
        # OUTPUT
        #
        outputCollapsibleButton = ctk.ctkCollapsibleButton()
        outputCollapsibleButton.text = "Output"
        outputCollapsibleButton.collapsed = False
        self.layout.addWidget(outputCollapsibleButton)

        outputFormLayout = qt.QFormLayout(outputCollapsibleButton)

        # Keep largest component
        self.keepLargestCheckBox = qt.QCheckBox()
        self.keepLargestCheckBox.checked = True
        outputFormLayout.addRow("Keep largest component:", self.keepLargestCheckBox)

        # Output suffix
        self.suffixLineEdit = qt.QLineEdit("_cut")
        outputFormLayout.addRow("Output suffix:", self.suffixLineEdit)

        #
        # ACTIONS
        #
        actionsWidget = qt.QWidget()
        actionsLayout = qt.QHBoxLayout(actionsWidget)
        actionsLayout.setContentsMargins(0, 0, 0, 0)
        actionsLayout.setSpacing(8)

        self.resetButton = qt.QPushButton("Reset to defaults")
        self.resetButton.toolTip = "Restore default parameter values"

        self.applyButton = qt.QPushButton("Process")
        self.applyButton.toolTip = "Process all meshes in the selected folder"

        actionsLayout.addWidget(self.resetButton)
        actionsLayout.addWidget(self.applyButton)

        self.layout.addWidget(actionsWidget)

        self.inputFolderSelectButton.connect("clicked(bool)", self.onSelectInputFolder)
        self.resetButton.connect("clicked(bool)", self.onResetDefaults)
        self.applyButton.connect("clicked(bool)", self.onApplyButton)

        self.absoluteLowestRadioButton.connect("toggled(bool)", self.onMethodChanged)
        self.lowestCentralRadioButton.connect("toggled(bool)", self.onMethodChanged)
        self.robustLowRadioButton.connect("toggled(bool)", self.onMethodChanged)

        self.onMethodChanged()
        self.layout.addStretch(1)

    def onSelectInputFolder(self):
        selectedFolder = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(),
            "Select input folder",
            self.inputFolderPath if self.inputFolderPath else qt.QDir.homePath()
        )

        if selectedFolder:
            self.inputFolderPath = selectedFolder
            self.inputFolderLineEdit.text = selectedFolder

    def onResetDefaults(self):
        self.inputFolderPath = ""
        self.inputFolderLineEdit.text = ""
        self.formatComboBox.setCurrentText("PLY")
        self.occlusalBandSpinBox.value = 35.0
        self.lowBandSpinBox.value = 25.0
        self.centralAreaSpinBox.value = 20.0
        self.robustLowRadioButton.checked = True
        self.percentileSpinBox.value = 5.0
        self.offsetSpinBox.value = 0.1
        self.keepLargestCheckBox.checked = True
        self.suffixLineEdit.text = "_cut"
        self.onMethodChanged()

    def getSelectedMethod(self):
        if self.absoluteLowestRadioButton.checked:
            return "Absolute lowest fovea"
        elif self.lowestCentralRadioButton.checked:
            return "Lowest central fovea"
        return "Robust low depression"

    def onMethodChanged(self, checked=None):
        method = self.getSelectedMethod()

        if method == "Absolute lowest fovea":
            self.percentileSpinBox.enabled = False
            self.centralAreaSpinBox.enabled = False
        elif method == "Lowest central fovea":
            self.percentileSpinBox.enabled = False
            self.centralAreaSpinBox.enabled = True
        else:
            self.percentileSpinBox.enabled = True
            self.centralAreaSpinBox.enabled = True

    def onApplyButton(self):
        inputFolder = self.inputFolderPath
        fileFormat = self.formatComboBox.currentText
        occlusalBandPercent = self.occlusalBandSpinBox.value
        lowBandPercent = self.lowBandSpinBox.value
        centralAreaPercent = self.centralAreaSpinBox.value
        method = self.getSelectedMethod()
        percentile = self.percentileSpinBox.value
        offsetZ = self.offsetSpinBox.value
        keepLargest = self.keepLargestCheckBox.checked
        suffix = self.suffixLineEdit.text.strip()

        if not inputFolder or not os.path.isdir(inputFolder):
            slicer.util.errorDisplay("Please select a valid input folder.")
            return

        if not suffix:
            slicer.util.errorDisplay(
                "Output suffix cannot be empty. "
                "A non-empty suffix is required to avoid overwriting the original files."
            )
            return

        try:
            results, outputFolder, canceled = self.logic.processFolder(
                inputFolder=inputFolder,
                fileFormat=fileFormat,
                occlusalBandPercent=occlusalBandPercent,
                lowBandPercent=lowBandPercent,
                centralAreaPercent=centralAreaPercent,
                method=method,
                percentile=percentile,
                offsetZ=offsetZ,
                keepLargest=keepLargest,
                suffix=suffix,
            )

            okCount = sum(1 for r in results if r["status"] == "OK")
            errCount = sum(1 for r in results if r["status"] != "OK")

            if canceled:
                msg = (
                    f"Processing canceled.\n\n"
                    f"Output folder:\n{outputFolder}\n\n"
                    f"Processed OK: {okCount}\n"
                    f"Errors: {errCount}"
                )
            else:
                msg = (
                    f"Finished.\n\n"
                    f"Output folder:\n{outputFolder}\n\n"
                    f"Processed OK: {okCount}\n"
                    f"Errors: {errCount}"
                )

            if errCount > 0:
                msg += "\n\nCheck Python Interactor for details."

            slicer.util.infoDisplay(msg)

        except Exception as e:
            slicer.util.errorDisplay(f"Error: {str(e)}")


#
# Logic
#
class AutoPlaneCutLogic(ScriptedLoadableModuleLogic):

    def processFolder(
        self,
        inputFolder,
        fileFormat="PLY",
        occlusalBandPercent=35.0,
        lowBandPercent=25.0,
        centralAreaPercent=20.0,
        method="Robust low depression",
        percentile=5.0,
        offsetZ=0.1,
        keepLargest=True,
        suffix="_cut",
    ):
        ext = ".ply" if fileFormat.upper() == "PLY" else ".obj"

        outputFolder = os.path.join(inputFolder, "output")
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        # Exclude files inside the output subfolder to prevent reprocessing
        # previously cut meshes in subsequent runs.
        outputFolderNorm = os.path.normcase(os.path.normpath(outputFolder))
        files = [
            f for f in os.listdir(inputFolder)
            if f.lower().endswith(ext)
            and os.path.normcase(os.path.normpath(os.path.join(inputFolder, f))) != outputFolderNorm
        ]
        files.sort()

        if not files:
            raise ValueError(f"No {ext} files found in input folder.")

        results = []
        totalFiles = len(files)
        canceled = False

        progress = slicer.util.createProgressDialog(
            labelText="Processing meshes...",
            maximum=totalFiles
        )
        progress.minimumDuration = 0
        progress.value = 0

        try:
            for i, fname in enumerate(files):
                if progress.wasCanceled:
                    canceled = True
                    print("[INFO] Processing canceled by user.")
                    break

                progress.labelText = f"Processing {fname} ({i + 1}/{totalFiles})"
                progress.value = i + 1
                slicer.app.processEvents()

                inPath = os.path.join(inputFolder, fname)
                baseName = os.path.splitext(fname)[0]
                outPath = os.path.join(outputFolder, baseName + suffix + ext)

                try:
                    polyData = self.loadMesh(inPath)

                    if polyData is None or polyData.GetNumberOfPoints() == 0:
                        raise ValueError("Empty mesh.")

                    zCut, debugInfo = self.computeOcclusalCutZ(
                        polyData=polyData,
                        occlusalBandPercent=occlusalBandPercent,
                        lowBandPercent=lowBandPercent,
                        centralAreaPercent=centralAreaPercent,
                        method=method,
                        percentile=percentile,
                        offsetZ=offsetZ,
                    )

                    clipped = self.clipKeepAbove(polyData, zCut)

                    if keepLargest:
                        clipped = self.keepLargestRegion(clipped)

                    self.saveMesh(clipped, outPath, fileFormat)

                    print(
                        f"[OK] {fname} | "
                        f"method={method} | "
                        f"zCut={zCut:.4f} | "
                        f"zBase={debugInfo['zBase']:.4f} | "
                        f"zMin={debugInfo['zMin']:.4f} | "
                        f"zMax={debugInfo['zMax']:.4f} | "
                        f"height={debugInfo['height']:.4f} | "
                        f"zTopThr={debugInfo['zThresholdTop']:.4f} | "
                        f"zLowThr={debugInfo['zThresholdLow']:.4f} | "
                        f"topPts={debugInfo['topCount']} | "
                        f"lowPts={debugInfo['lowCount']} | "
                        f"focusPts={debugInfo['focusCount']} | "
                        f"saved: {outPath}"
                    )

                    results.append({
                        "file": fname,
                        "status": "OK",
                        "zCut": zCut
                    })

                except Exception as e:
                    print(f"[ERROR] {fname}: {e}")
                    results.append({
                        "file": fname,
                        "status": f"ERROR: {str(e)}",
                        "zCut": None
                    })

                slicer.app.processEvents()

        finally:
            progress.close()

        return results, outputFolder, canceled

    def loadMesh(self, path):
        ext = os.path.splitext(path)[1].lower()

        if ext == ".ply":
            reader = vtk.vtkPLYReader()
        elif ext == ".obj":
            reader = vtk.vtkOBJReader()
        else:
            raise ValueError(f"Unsupported format: {ext}")

        reader.SetFileName(path)
        reader.Update()

        polyData = vtk.vtkPolyData()
        polyData.DeepCopy(reader.GetOutput())

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(polyData)
        cleaner.Update()

        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(cleaner.GetOutput())
        triangulate.Update()

        output = vtk.vtkPolyData()
        output.DeepCopy(triangulate.GetOutput())

        if output.GetNumberOfPoints() == 0:
            raise ValueError("Loaded mesh has no points.")

        return output

    def saveMesh(self, polyData, path, fileFormat):
        if polyData is None or polyData.GetNumberOfPoints() == 0:
            raise ValueError("Cannot save empty mesh.")

        if fileFormat.upper() == "PLY":
            writer = vtk.vtkPLYWriter()
            writer.SetFileName(path)
            writer.SetInputData(polyData)
            writer.SetFileTypeToBinary()
            writer.Write()

        elif fileFormat.upper() == "OBJ":
            writer = vtk.vtkOBJWriter()
            writer.SetFileName(path)
            writer.SetInputData(polyData)
            writer.Write()

        else:
            raise ValueError(f"Unsupported output format: {fileFormat}")

    def computeOcclusalCutZ(
        self,
        polyData,
        occlusalBandPercent=35.0,
        lowBandPercent=25.0,
        centralAreaPercent=20.0,
        method="Robust low depression",
        percentile=5.0,
        offsetZ=0.1,
    ):
        points = polyData.GetPoints()
        n = polyData.GetNumberOfPoints()

        if n < 10:
            raise ValueError("Mesh has too few points.")

        coords = []
        zValues = []

        for i in range(n):
            p = points.GetPoint(i)
            coords.append((p[0], p[1], p[2]))
            zValues.append(p[2])

        zMin = min(zValues)
        zMax = max(zValues)
        height = zMax - zMin

        if height <= 0:
            raise ValueError("Invalid Z range.")

        # 1) Top occlusal band
        bandFrac = occlusalBandPercent / 100.0
        zThresholdTop = zMax - (height * bandFrac)
        topPoints = [p for p in coords if p[2] >= zThresholdTop]

        if len(topPoints) < 10:
            raise ValueError("Too few points in occlusal band.")

        # 2) Low-Z candidate points within occlusal band
        topZ = [p[2] for p in topPoints]
        zThresholdLow = self.percentileValue(topZ, lowBandPercent)
        lowPoints = [p for p in topPoints if p[2] <= zThresholdLow]

        if len(lowPoints) < 5:
            lowPoints = topPoints

        zBase = None
        focusCount = 0

        if method == "Absolute lowest fovea":
            candidateZ = [p[2] for p in lowPoints]
            # NOTE: min() is sensitive to mesh artefacts or outlier vertices.
            # For noisy meshes, prefer "Robust low depression" instead.
            zBase = min(candidateZ)
            focusCount = len(lowPoints)

        elif method == "Lowest central fovea":
            centerX = self.median([p[0] for p in topPoints])
            centerY = self.median([p[1] for p in topPoints])

            distData = []
            for p in topPoints:
                d = math.sqrt((p[0] - centerX) ** 2 + (p[1] - centerY) ** 2)
                distData.append((p, d))

            distances = [item[1] for item in distData]
            if len(distances) == 0:
                raise ValueError("No distances computed for central fovea mode.")

            distanceThreshold = self.percentileValue(distances, centralAreaPercent)
            focusPoints = [p for p, d in distData if d <= distanceThreshold]

            if len(focusPoints) < 5:
                focusPoints = topPoints

            zBase = min([p[2] for p in focusPoints])
            focusCount = len(focusPoints)

        elif method == "Robust low depression":
            centerX = self.median([p[0] for p in lowPoints])
            centerY = self.median([p[1] for p in lowPoints])

            distData = []
            for p in lowPoints:
                d = math.sqrt((p[0] - centerX) ** 2 + (p[1] - centerY) ** 2)
                distData.append((p, d))

            distances = [item[1] for item in distData]
            if len(distances) == 0:
                raise ValueError("No distances computed.")

            distanceThreshold = self.percentileValue(distances, centralAreaPercent)
            focusPoints = [p for p, d in distData if d <= distanceThreshold]

            if len(focusPoints) < 5:
                focusPoints = lowPoints

            focusZ = [p[2] for p in focusPoints]
            zBase = self.percentileValue(focusZ, percentile)
            focusCount = len(focusPoints)

        else:
            raise ValueError(f"Unknown method: {method}")

        zCut = zBase - offsetZ

        debugInfo = {
            "zMin": zMin,
            "zMax": zMax,
            "height": height,
            "zThresholdTop": zThresholdTop,
            "zThresholdLow": zThresholdLow,
            "topCount": len(topPoints),
            "lowCount": len(lowPoints),
            "focusCount": focusCount,
            "zBase": zBase,
            "zCut": zCut,
        }

        return zCut, debugInfo

    def clipKeepAbove(self, polyData, zCut):
        plane = vtk.vtkPlane()
        plane.SetOrigin(0.0, 0.0, zCut)
        plane.SetNormal(0.0, 0.0, 1.0)

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(polyData)
        clipper.SetClipFunction(plane)
        clipper.InsideOutOff()
        clipper.GenerateClippedOutputOff()
        clipper.Update()

        clipped = vtk.vtkPolyData()
        clipped.DeepCopy(clipper.GetOutput())

        if clipped.GetNumberOfPoints() == 0:
            # The cut plane is outside the mesh Z range: raise a descriptive error
            # instead of silently inverting the plane, which would return the
            # sub-cervical portion rather than the crown.
            bounds = polyData.GetBounds()  # (xMin, xMax, yMin, yMax, zMin, zMax)
            raise ValueError(
                f"Clipping produced an empty mesh. "
                f"zCut={zCut:.4f} is outside the mesh Z range "
                f"[{bounds[4]:.4f}, {bounds[5]:.4f}]. "
                f"Check occlusal band parameters or mesh orientation."
            )

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(clipped)
        cleaner.Update()

        cleaned = vtk.vtkPolyData()
        cleaned.DeepCopy(cleaner.GetOutput())

        if cleaned.GetNumberOfPoints() == 0:
            raise ValueError("Cleaned clipped mesh is empty.")

        return cleaned

    def keepLargestRegion(self, polyData):
        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputData(polyData)
        connectivity.SetExtractionModeToLargestRegion()
        connectivity.Update()

        largest = vtk.vtkPolyData()
        largest.DeepCopy(connectivity.GetOutput())

        if largest.GetNumberOfPoints() == 0:
            return polyData

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(largest)
        cleaner.Update()

        output = vtk.vtkPolyData()
        output.DeepCopy(cleaner.GetOutput())
        return output

    def median(self, values):
        if not values:
            raise ValueError("Cannot compute median of empty list.")
        s = sorted(values)
        m = len(s) // 2
        if len(s) % 2 == 0:
            return 0.5 * (s[m - 1] + s[m])
        return s[m]

    def percentileValue(self, values, percentile):
        if not values:
            raise ValueError("Cannot compute percentile of empty list.")

        if percentile <= 0:
            return min(values)
        if percentile >= 100:
            return max(values)

        s = sorted(values)
        k = (len(s) - 1) * (percentile / 100.0)
        f = math.floor(k)
        c = math.ceil(k)

        if f == c:
            return s[int(k)]

        d0 = s[int(f)] * (c - k)
        d1 = s[int(c)] * (k - f)
        return d0 + d1