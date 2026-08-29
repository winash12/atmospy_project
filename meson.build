project('pramanavayu', 'cpp',
    version: '0.1.0',
    default_options: [
        'cpp_std=c++20',      # Matches your project requirement
        'optimization=3',     # Max performance for voxel interpolation
        'warning_level=2'
    ]
)

# --- 1. Find Python and its Development Dependency ---
# py_dep is critical; without it, you get "undefined symbol" errors for Py... symbols
py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency(embed:true)

# --- 2. Dynamic NumPy Include Path ---
# Required for xtensor-python to interface with the NumPy C-API
numpy_inc_path = run_command(py, [
    '-c', 'import numpy; print(numpy.get_include())'
], check: true).stdout().strip()

# --- 3. Header and Source Locations ---
# inc_vayu tells the compiler where to find vayu_ops.hpp
inc_vayu = include_directories('cpp/include')

sources = [
    'cpp/vayu_kernels.cpp', 
    'cpp/vayu_bindings.cpp'
]

# --- 4. Define Dependencies ---
# Meson handles finding the system paths for these libraries
deps = [
    py_dep,
    dependency('pybind11'),
    dependency('xtensor'),
    dependency('xtensor-python'),
    dependency('openmp')
]

# --- 5. Build the Extension Module ---
py.extension_module(
    'vayu_core',
    sources: sources,
    dependencies: deps,
    include_directories: [inc_vayu],
    cpp_args: [
        '-march=native',
        '-mavx2',
        '-fno-fast-math',
        '-fopenmp',
        '-I' + numpy_inc_path,
        '-D_GLIBCXX_ASSERTIONS=1'
    ],
    # Add link_args to prevent the linker from dropping "unused" symbols
    link_args: ['-Wl,--no-as-needed'], 
    gnu_symbol_visibility: 'default',
    install: true
)






