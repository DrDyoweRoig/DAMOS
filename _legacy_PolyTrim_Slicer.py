import os
import slicer
import vtk
import qt
import ctk

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)

#
# Module
#
class PolyTrim(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)

        parent.title = "PolyTrim"

        # Icon (optional): put PolyTrim.png next to this .py file
        iconPath = os.path.join(os.path.dirname(__file__), "PolyTrim.png")
        if os.path.exists(iconPath):
            parent.icon = qt.QIcon(iconPath)

        parent.categories = ["Surface Models"]
        parent.dependencies = []
        parent.contributors = ["Albert"]
        parent.helpText = (
            "PolyTrim: batch decimation of PLY/OBJ meshes in a folder to a target polygon count.\n"
            "Outputs are saved in an 'output' subfolder as PLY."
        )
        parent.acknowledgementText = ""


#
# Widget (UI)
#
class PolyTrimWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.logic = PolyTrimLogic()

        # --- Parameters UI ---
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)

        formLayout = qt.QFormLayout(parametersCollapsibleButton)

        # Input folder
        self.inputDirButton = ctk.ctkDirectoryButton()
        self.inputDirButton.caption = "Select input folder"
        formLayout.addRow("Input folder:", self.inputDirButton)

        # Format
        self.formatCombo = qt.QComboBox()
        self.formatCombo.addItems(["PLY", "OBJ"])
        formLayout.addRow("Input format:", self.formatCombo)

        # Target polygons
        self.targetSpin = qt.QSpinBox()
        self.targetSpin.minimum = 1000
        self.targetSpin.maximum = 2000000
        self.targetSpin.singleStep = 1000
        self.targetSpin.value = 20000
        formLayout.addRow("Target polygons:", self.targetSpin)

        # Preserve topology (recommended)
        self.preserveTopologyCheck = qt.QCheckBox()
        self.preserveTopologyCheck.checked = True
        formLayout.addRow("Preserve topology:", self.preserveTopologyCheck)

        # Run button
        self.runButton = qt.QPushButton("Run")
        self.runButton.enabled = False
        formLayout.addRow(self.runButton)

        # Progress bar
        self.progressBar = qt.QProgressBar()
        self.progressBar.minimum = 0
        self.progressBar.maximum = 100
        self.progressBar.value = 0
        formLayout.addRow("Progress:", self.progressBar)

        # Log
        self.logText = qt.QPlainTextEdit()
        self.logText.setReadOnly(True)
        self.logText.setMinimumHeight(160)
        formLayout.addRow("Log:", self.logText)

        self.layout.addStretch(1)

        # Connections
        self.inputDirButton.connect("directoryChanged(QString)", self.onInputDirChanged)
        self.runButton.connect("clicked()", self.onRun)

    def onInputDirChanged(self, _):
        self.runButton.enabled = bool(self.inputDirButton.directory)

    def appendLog(self, msg: str):
        self.logText.appendPlainText(msg)
        self.logText.ensureCursorVisible()
        slicer.app.processEvents()

    def onRun(self):
        inputDir = self.inputDirButton.directory
        if not inputDir or not os.path.isdir(inputDir):
            qt.QMessageBox.warning(slicer.util.mainWindow(), "PolyTrim", "Please select a valid input folder.")
            return

        fmt = self.formatCombo.currentText
        target = int(self.targetSpin.value)
        preserve = bool(self.preserveTopologyCheck.checked)

        ext = ".ply" if fmt == "PLY" else ".obj"

        outputDir = os.path.join(inputDir, "output")
        os.makedirs(outputDir, exist_ok=True)

        files = sorted([f for f in os.listdir(inputDir) if f.lower().endswith(ext)])
        if not files:
            self.appendLog(f"No {ext} files found in: {inputDir}")
            return

        self.logText.clear()
        self.progressBar.value = 0

        self.appendLog(f"Input:  {inputDir}")
        self.appendLog(f"Output: {outputDir}")
        self.appendLog(f"Format: {fmt} ({ext})")
        self.appendLog(f"Target polygons: {target}")
        self.appendLog(f"Preserve topology: {preserve}")
        self.appendLog(f"Files found: {len(files)}")
        self.appendLog("----")

        for i, f in enumerate(files, 1):
            inPath = os.path.join(inputDir, f)
            base = os.path.splitext(f)[0]
            outPath = os.path.join(outputDir, f"{base}_{target}.ply")  # always save as PLY

            modelNode = None
            outNode = None

            try:
                modelNode = slicer.util.loadModel(inPath)
                if modelNode is None:
                    raise RuntimeError("Could not load model")

                poly = modelNode.GetPolyData()
                n0 = poly.GetNumberOfPolys()

                outPoly = self.logic.decimateToTarget(poly, targetPolys=target, preserveTopology=preserve)
                n1 = outPoly.GetNumberOfPolys()

                outNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{base}_{target}")
                outNode.SetAndObservePolyData(outPoly)
                outNode.CreateDefaultDisplayNodes()

                saved = slicer.util.saveNode(outNode, outPath)
                if not saved:
                    raise RuntimeError("Could not save output file")

                self.appendLog(f"[{i}/{len(files)}] OK  {f}: {n0} -> {n1}")

            except Exception as e:
                self.appendLog(f"[{i}/{len(files)}] FAIL {f}: {e}")

            # Clean scene nodes to avoid memory growth
            if modelNode is not None:
                slicer.mrmlScene.RemoveNode(modelNode)
            if outNode is not None:
                slicer.mrmlScene.RemoveNode(outNode)

            self.progressBar.value = int((i / len(files)) * 100)

        self.appendLog("----")
        self.appendLog("Done.")


#
# Logic
#
class PolyTrimLogic(ScriptedLoadableModuleLogic):
    def triangulate(self, poly):
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(poly)
        tri.Update()
        return tri.GetOutput()

    def decimateToTarget(self, poly, targetPolys=20000, preserveTopology=True):
        poly = self.triangulate(poly)
        n0 = poly.GetNumberOfPolys()

        if n0 == 0:
            raise ValueError("Input mesh has 0 polygons.")
        if n0 <= targetPolys:
            return poly

        reduction = 1.0 - (targetPolys / float(n0))
        reduction = max(0.0, min(0.99, reduction))

        dec = vtk.vtkDecimatePro()
        dec.SetInputData(poly)
        dec.SetTargetReduction(float(reduction))

        if preserveTopology:
            dec.PreserveTopologyOn()
        else:
            dec.PreserveTopologyOff()

        dec.BoundaryVertexDeletionOff()
        dec.SplittingOff()
        dec.Update()

        return self.triangulate(dec.GetOutput())