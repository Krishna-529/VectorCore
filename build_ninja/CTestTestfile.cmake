# CMake generated Testfile for 
# Source directory: C:/Users/HPW/Desktop/Vectra-X/Vectra-X
# Build directory: C:/Users/HPW/Desktop/Vectra-X/Vectra-X/build_ninja
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[vectrax_smoke]=] "C:/Users/HPW/Desktop/Vectra-X/Vectra-X/build_ninja/vectrax_smoke_test.exe")
set_tests_properties([=[vectrax_smoke]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/HPW/Desktop/Vectra-X/Vectra-X/CMakeLists.txt;70;add_test;C:/Users/HPW/Desktop/Vectra-X/Vectra-X/CMakeLists.txt;0;")
subdirs("_deps/pybind11-build")
