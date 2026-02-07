from __future__ import annotations

import sys

from setuptools import setup

# pybind11 provides build helpers that correctly set up include paths and
# compiler flags for Python extension modules.
from pybind11.setup_helpers import Pybind11Extension, build_ext


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
        return ["/O2", "/arch:AVX2"]

    # GCC/Clang flags
    return [
        "-O3",
        "-mavx2",
        "-mfma",
    ]


ext_modules = [
    Pybind11Extension(
        "vectorcore",  # module name: import vectorcore
        [
            "src/main.cpp",
            "src/VectorStore.cpp",
        ],
        include_dirs=[
            "include",
            "src",
        ],
        cxx_std=17,
        extra_compile_args=compile_args(),
    )
]


setup(
    name="vectorcore",
    version="0.0.0",
    description="VectorCore (prototype) - pybind11 extension",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
