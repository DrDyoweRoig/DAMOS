# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect all data files, binaries, and hidden imports for key libraries
pyvista_datas, pyvista_binaries, pyvista_hidden = collect_all('pyvista')
pyvistaqt_datas, pyvistaqt_binaries, pyvistaqt_hidden = collect_all('pyvistaqt')
vtk_hidden = collect_submodules('vtkmodules')
trimesh_datas, trimesh_binaries, trimesh_hidden = collect_all('trimesh')

a = Analysis(
    ['damos.py'],
    pathex=['.'],
    binaries=pyvista_binaries + pyvistaqt_binaries + trimesh_binaries,
    datas=(
        pyvista_datas
        + pyvistaqt_datas
        + trimesh_datas
        + [
            ('assets/logo_damos.png', 'assets'),
            ('icons/damos_logo.ico', 'icons'),
            ('icons/damos_logo_exe.png', 'icons'),
        ]
    ),
    hiddenimports=(
        pyvista_hidden
        + pyvistaqt_hidden
        + vtk_hidden
        + trimesh_hidden
        + [
            'PySide6.QtSvg',
            'PySide6.QtOpenGL',
            'PySide6.QtOpenGLWidgets',
            'PySide6.QtPrintSupport',
            'scipy.special._cdflib',
            'scipy._lib.messagestream',
            'gui',
            'gui.main_window',
            'gui.home_screen',
            'gui.control_panel',
            'gui.viewer3d',
            'gui.viewport_widget',
            'gui.file_list_panel',
            'gui.app_state',
            'gui.style',
            'gui.orientation_engine',
            'ffei',
            'ffei.detection',
            'ffei.watershed',
            'ffei.pipeline',
            'ffei.surface_fields',
            'ffei.metrics',
            'ffei.escape_channels',
            'ffei.io',
            'modules',
            'modules.autolmk_panel',
            'modules.autoplancut_panel',
            'modules.automorph_panel',
            'modules.ffei_panel',
            'modules.meshorient_panel',
            'modules.polytrim_panel',
            'batch_processor',
            'mesh_exporter',
            'ofc_engine',
            'orientation_engine',
        ]
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib.tests',
        'numpy.tests',
        'scipy.tests',
        'PyQt5',
        'PyQt6',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DAMOS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icons/damos_logo.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='DAMOS',
)