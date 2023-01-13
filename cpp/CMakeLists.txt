cmake_minimum_required(VERSION 3.22.1)
project(pca)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -Wall -fopenmp -pthread")
include_directories(./include/eigen)
include_directories(./include/)
include_directories(${PYTHON_INCLUDE_DIRS})
add_subdirectory(./include/pybind11)

pybind11_add_module(_oja ./src/oja.cpp)
