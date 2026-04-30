project('pramanavayu', 'cpp', 
        version: '0.1.0', 
        default_options: ['cpp_std=c++20', 'optimization=3'])


py = import('python').find_installation(pure: false)


# --- 2. Dynamic NumPy Path (As String) ---
numpy_inc_path = run_command(py, [
  '-c', 'import numpy; print(numpy.get_include())'
], check: true).stdout().strip()


# Dependencies
# Ensure xtensor-python is installed via your package manager or conda
deps = [
    dependency('pybind11'),
    dependency('xtensor'),
    dependency('xtensor-python'),
    dependency('openmp')
]

# 1. Define the source files first
sources = [
    'cpp/vayu_kernels.cpp', 
    'cpp/vayu_bindings.cpp'
]

# CRITICAL: -fno-fast-math prevents the compiler from reordering 
# floating-point ops, which is essential to match F77 at 1e-15.
py.extension_module(
    'vayu_core',
    sources: sources,
    dependencies: [
        dependency('pybind11'),
        dependency('xtensor'),
        dependency('xtensor-python'),
        dependency('openmp')
    ],
    cpp_args: [
        '-march=native',
        '-mavx2',
        '-fno-fast-math', # Reminder: Use no-fast-math for 1e-15 precision matching
        '-fopenmp',
        '-I' + numpy_inc_path, # Manually inject the NumPy path as requested
        '-D_GLIBCXX_ASSERTIONS=1'
    ],
    install: true
)