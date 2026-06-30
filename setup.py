from setuptools import setup, find_packages

setup(
    name="ffei",
    version="0.1.0",
    description="Food-Flow Estimation Index — Topographic-functional analysis of 3D dental occlusal surfaces",
    author="Albert Epitíe Dyowe Roig",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.8",
        "trimesh>=3.15",
    ],
    extras_require={
        "gui": ["pyvista>=0.38", "pyvistaQt>=0.9", "PySide6>=6.4"],
        "viz": ["matplotlib>=3.5"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
    ],
)
