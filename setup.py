from __future__ import annotations

import sys

from setuptools import setup

# pybind11 provides build helpers that correctly set up include paths and
# compiler flags for Python extension modules.
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Single source of truth for the version. Passed to C++ via the VERSION_INFO
# macro and stringified there (see pybind_module.cpp).
VERSION = "0.1.0"


def compile_args():
    """Return compiler args for high-performance builds.

    Requirement mapping:
    - C++17
    - -O3
    - -mavx2 for SIMD

    Note (Windows/MSVC):
    - MSVC doesn't understand -O3/-mavx2, so we translate to /O2 and /arch:AVX2.
    """

    if sys.platform.startswith("win"):
        # MSVC flags
        return ["/O2", "/arch:AVX2", "/openmp"]

    # GCC/Clang flags
    return [
        "-O3",
        "-mavx2",
        "-mfma",
        "-fopenmp",
    ]

def link_args():
    if sys.platform.startswith("win"):
        return []
    return ["-fopenmp"]


ext_modules = [
    Pybind11Extension(
        "vectorcore",  # module name: import vectorcore
        [
            "src/pybind_module.cpp",
            "src/bruteforce_index.cpp",
            "src/distance.cpp",
            "src/hnsw_index.cpp",
            "src/pq_index.cpp",
        ],
        include_dirs=[
            "include",
        ],
        cxx_std=17,
        define_macros=[("VERSION_INFO", VERSION)],
        extra_compile_args=compile_args(),
        extra_link_args=link_args(),
    )
]


setup(
    name="vectorcore",
    version=VERSION,
    description="VectorCore (prototype) - pybind11 extension",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
