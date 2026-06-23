cmake_minimum_required(VERSION 3.20)

project(vectorcore LANGUAGES CXX)

# VectorCore core constraints:
# - C++17 core engine
# - Python bindings via pybind11
# - High performance: flat memory, SIMD distance kernels

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release for performance testing when the user doesn't specify.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(VECTORCORE_ENABLE_AVX2 "Enable AVX2 distance kernels when supported" ON)

include(FetchContent)

# Pull pybind11 at configure time. This keeps the repo lightweight for interviews.
# If you prefer vendoring/submodules, swap this for add_subdirectory(pybind11).
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

add_library(vectorcore_core
  src/bruteforce_index.cpp
  src/distance.cpp
  src/hnsw_index.cpp
)

target_include_directories(vectorcore_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(vectorcore_core PUBLIC cxx_std_17)

# Enable AVX2 where available. This is deliberately explicit so you can explain
# it in an interview: we want the compiler to actually emit AVX2 instructions.
if(VECTORCORE_ENABLE_AVX2)
  if(MSVC)
    target_compile_options(vectorcore_core PRIVATE /arch:AVX2)
  else()
    target_compile_options(vectorcore_core PRIVATE -mavx2 -mfma)
  endif()
endif()

# OpenMP parallelizes the brute-force scan. Without this the #pragma omp lines
# are silently ignored and the CMake build runs the search single-threaded
# (the pip/setup.py build already passes /openmp | -fopenmp).
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(vectorcore_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Python extension module
pybind11_add_module(vectorcore
  src/pybind_module.cpp
)

target_link_libraries(vectorcore PRIVATE vectorcore_core)

# Passed as a bare token; pybind_module.cpp stringifies it (cross-platform safe).
target_compile_definitions(vectorcore PRIVATE VERSION_INFO=0.1.0)

# ---- Tests ----
enable_testing()

option(VECTORCORE_BUILD_TESTS "Build the GoogleTest unit-test suite" ON)

# Minimal smoke test (no external dependencies).
add_executable(vectorcore_smoke_test tests/test_smoke.cpp)
target_link_libraries(vectorcore_smoke_test PRIVATE vectorcore_core)
add_test(NAME vectorcore_smoke COMMAND vectorcore_smoke_test)

if(VECTORCORE_BUILD_TESTS)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  # Windows: make GoogleTest use the same shared CRT as the rest of the build.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(vectorcore_unit_tests
    tests/test_distance.cpp
    tests/test_bruteforce.cpp
    tests/test_hnsw.cpp
  )
  target_link_libraries(vectorcore_unit_tests PRIVATE vectorcore_core GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(vectorcore_unit_tests)
endif()
