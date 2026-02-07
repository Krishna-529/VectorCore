cmake_minimum_required(VERSION 3.20)

project(vectrax LANGUAGES CXX)

# Vectra-X core constraints:
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

option(VECTRAX_ENABLE_AVX2 "Enable AVX2 distance kernels when supported" ON)

include(FetchContent)

# Pull pybind11 at configure time. This keeps the repo lightweight for interviews.
# If you prefer vendoring/submodules, swap this for add_subdirectory(pybind11).
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

add_library(vectrax_core
  src/bruteforce_index.cpp
  src/distance.cpp
  src/hnsw_index.cpp
  src/VectorStore.cpp
)

target_include_directories(vectrax_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(vectrax_core PUBLIC cxx_std_17)

# Enable AVX2 where available. This is deliberately explicit so you can explain
# it in an interview: we want the compiler to actually emit AVX2 instructions.
if(VECTRAX_ENABLE_AVX2)
  if(MSVC)
    target_compile_options(vectrax_core PRIVATE /arch:AVX2)
  else()
    target_compile_options(vectrax_core PRIVATE -mavx2 -mfma)
  endif()
endif()

# Python extension module
pybind11_add_module(vectrax
  src/pybind_module.cpp
)

target_link_libraries(vectrax PRIVATE vectrax_core)

target_compile_definitions(vectrax PRIVATE VECTRAX_VERSION=\"0.1.0\")

# Simple C++ smoke test (optional but helpful for quick validation)
enable_testing()
add_executable(vectrax_smoke_test tests/test_smoke.cpp)

target_link_libraries(vectrax_smoke_test PRIVATE vectrax_core)
add_test(NAME vectrax_smoke COMMAND vectrax_smoke_test)
