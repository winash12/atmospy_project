#!/bin/tcsh -f
# =============================================================================
#     PRAMANA VAYU (PV) REPOSITORY CLEANUP & JOSS STANDARDIZATION SCRIPT
# =============================================================================

echo "========================================================================="
echo "--> Initializing Pristine JOSS Repository Structure Sweep..."
echo "========================================================================="

# 1. Structure canonical JOSS directory paths layout safely
mkdir -p doc
mkdir -p vayu
mkdir -p src
mkdir -p tests
mkdir -p fortran_test
mkdir -p .github/workflows

# Create the dedicated high-performance C++ sub-structure trees
mkdir -p cpp/include
mkdir -p cpp/src

# 2. Touch future Python solver modules inside core vayu/ package domain
if ( ! -e "vayu/finite_diff_solvers.py" ) then
    touch vayu/finite_diff_solvers.py
    echo "[STRATEGY] Initialized solver file: vayu/finite_diff_solvers.py"
endif

if ( ! -e "vayu/spectral_grid_solvers.py" ) then
    touch vayu/spectral_grid_solvers.py
    echo "[STRATEGY] Initialized solver file: vayu/spectral_grid_solvers.py"
endif

# 3. Relocate active Python source files down to core vayu/ package domain
foreach src_file (config_loader.py coordinate_transformers.py thermostatics.py thermodynamics.py library_strategies.py physics_strategies.py strategy_factory.py strategy_interface.py PV.py)
    if ( -e "$src_file" ) then
        mv "$src_file" vayu/
        echo "[MIGRATE] Core source moved: $src_file -> vayu/"
    endif
end

# Initialize the explicit package tracking index if not present
if ( ! -e "vayu/__init__.py" ) then
    touch vayu/__init__.py
    echo "[INIT] Package index file initialized: vayu/__init__.py"
endif

# 4. Relocate active operational scripts down to the clean src/ directory
foreach script_file (fetch_core.csh merge.py CORe_DailyMeanPV.py XDailyMeanPV.py DaskXDailyMeanPV_Core.py XTensorDailyMeanPv_Core.py config_service.py print2.py)
    if ( -e "$script_file" ) then
        mv "$script_file" src/
        echo "[MIGRATE] Script moved: $script_file -> src/"
    endif
end

# 5. Relocate historical Fortran references down to fortran_test/ directory
foreach f90_file (legacy_s2thta.f90 p2thta.f90 main_test.f90 legacy_ddy.f90 p2thta_cut_old.f90 p2thta_cut.f90)
    if ( -e "$f90_file" ) then
        mv "$f90_file" fortran_test/
        echo "[MIGRATE] Fortran code moved: $f90_file -> fortran_test/"
    endif
end

# 6. AGGRESSIVE SWEEP: Purge transient data files, graphs, and backup files
echo "--> Purging junk, text editor temporary files, NetCDF/GRIB data caches..."

# Remove text editor backup files completely
rm -f *~
rm -f vayu/*~
rm -f src/*~
rm -f fortran_test/*~

# Remove explicit file backups and text scraps
rm -f coordinate_transformers.py.backup
rm -f strategy_factory.py~ strategy_interface.py~ physics_strategies.py~ merge.py_backup
rm -f crap crap~

# Remove large datasets, GRIB binary tracking indices (.idx) and compiled .so maps
rm -f *.nc
rm -f *.grb
rm -f *.idx
rm -f *.png
rm -f debug_solver
rm -f ff_core.cpython-312-x86_64-linux-gnu.so

# Scrub python and pytest runtime registers cache memory grids
find . -name "__pycache__" -exec rm -rf {} +
find . -name ".pytest_cache" -exec rm -rf {} +

# 7. INSTANTIATE JOSS DOCUMENTATION AND REPOSITORY ESSENTIALS
if ( ! -e "LICENSE" ) then
    echo "Copyright (c) 2026 Pramana Vayu Developers." > LICENSE
    echo "All rights reserved. Provided under the open-source MIT License terms." >> LICENSE
    echo "[DOC] Initialized LICENSE file."
endif

if ( ! -e "doc/paper.md" ) then
    echo "---" > doc/paper.md
    echo "title: 'Pramana Vayu: A Hybrid High-Performance Atmospheric Isentropic Potential Vorticity Transforms Engine'" >> doc/paper.md
    echo "tags: [meteorology, dynamics, potential-vorticity, isentropic, cpp, openmp]" >> doc/paper.md
    echo "authors: [Aswin]" >> doc/paper.md
    echo "---" >> doc/paper.md
    echo "[DOC] Template compiled for JOSS paper layout: doc/paper.md"
endif

# 8. FIXED: COMPILING MANDATORY PYPROJECT.TOML PACKAGING HANDBOOK
if ( ! -e "pyproject.toml" ) then
    cat <<EOF > pyproject.toml
[build-system]
requires = ["meson-python", "ninja", "numpy", "xtensor"]
build-backend = "mesonpy"

[project]
name = "vayu"
version = "1.0.0"
description = "Hybrid High-Performance Atmospheric Isentropic Potential Vorticity Transforms Engine"
authors = [{name = "Aswin"}]
license = {text = "MIT"}
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "xarray",
    "scipy",
    "matplotlib",
    "cartopy",
    "dask",
    "distributed"
]
EOF
    echo "[PACKAGING] Successfully compiled production-ready: pyproject.toml"
else
    echo "[PACKAGING] Retained existing pyproject.toml at root layout position."
endif

# Ensure existing config resources are explicitly recognized at root
if ( -e "config.toml" ) then
    echo "[RETAIN] Preserved active runtime framework state parameters: config.toml"
endif
if ( -e "config.yaml" ) then
    echo "[RETAIN] Preserved alternative layout properties: config.yaml"
endif

# 9. GENERATE UNIVERSAL .GITIGNORE FILE BLOCK
cat <<EOF > .gitignore
# Compiled Binary Shared Objects Tracks
*.so
build/
subprojects/

# Dataset Caches and Plot Output Graphics
*.nc
*.grb
*.grib
*.idx
*.png
*.log

# Workspace registers and local system overrides
__pycache__/
.venv/
.pytest_cache/
.vscode/
.idea/
.DS_Store
EOF
git add .gitignore
echo "[GIT] Universal project .gitignore file rules compiled successfully."

echo "========================================================================="
echo "--> STATUS: COMPLETE! DIRECTORY PURGED AND STRUCTURALLY READY WITH TOMLS!"
echo "========================================================================="
