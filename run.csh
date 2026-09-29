mkdir -p .github/workflows

# 2. Write the Continuous Integration YAML file directly inside that path
cat << 'EOF' > .github/workflows/ci.yml
name: Pramana Vayu High-Performance Engine CI Suite

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout Code Repository State
      uses: actions/checkout@v4

    - name: Set Up Python Application Environment
      uses: actions/setup-python@v5
      with:
        python-version: '3.12'
        cache: 'pip'

    - name: Install System Packaging Tools and OpenMP Core Compiler Runtime
      run: |
        sudo apt-get update
        sudo apt-get install -y libopenmp-dev ninja-build gfortran libspatialindex-dev libgeos-dev

    - name: Install High-Performance Scientific Dependencies
      run: |
        python -m pip install --upgrade pip
        pip install numpy xarray scipy matplotlib cartopy dask distributed meson ninja pybind11 xtensor pytest

    - name: Configure and Compile Fused Multi-Threaded C++ Binary Extensions Via Meson
      run: |
        meson setup build
        meson compile -C build
        meson install -C build

    - name: Run Modular Tests Verification Suites Framework
      run: |
        pytest tests/ --ignore=.venv/
EOF

