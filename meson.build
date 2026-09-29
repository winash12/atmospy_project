project('pramanavayu', 'cpp',
    version: '1.0.0',
    default_options: [
        'cpp_std=c++20',      # Enforces C++20 standard compliance rules
        'optimization=3',     # Leverages full O3 compiler loop optimizations
        'warning_level=2'
    ]
)

# --- 1. Find Python and its Development Dependency ---
py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency(embed: true)

# --- 2. Dynamic NumPy Include Path Check ---
numpy_inc_path = run_command(py, [
    '-c', 'import numpy; print(numpy.get_include())'
], check: true).stdout().strip()

# --- 3. Synchronized Header and Source JOSS Paths ---
# Tells the compiler where to locate vayu_ops.hpp
inc_vayu = include_directories('cpp/include')

# FIXED: Aligned source paths straight into your clean JOSS cpp/src/ directory layout!
sources = [
    'cpp/src/isobaric_to_isentropic_pressure.cpp', 
    'cpp/src/isobaric_to_isentropic_velocity.cpp', 
    'cpp/src/vayu_core_bindings.cpp'
]

# --- 4. Resolve System Packaging Dependencies ---
deps = [
    py_dep,
    dependency('pybind11'),
    dependency('xtensor'),
    dependency('xtensor-python'),
    dependency('openmp')
]

# --- 5. Build the Shared Extension C++ Binary Module ---
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
    link_args: ['-Wl,--no-as-needed'], 
    gnu_symbol_visibility: 'default',
    install: true
)
