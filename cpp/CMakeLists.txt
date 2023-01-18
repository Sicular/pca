cmake_minimum_required(VERSION 3.22.1)
project(pca LANGUAGES CXX CUDA)

find_package(CUDA QUIET REQUIRED)

# find_package(PythonLibs REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -Wall -pthread -L/usr/local/cuda-11.7/lib64 -lcusparse -lcudart")

include_directories(./include/eigen)
include_directories(./include/)
include_directories(${PYTHON_INCLUDE_DIRS})
add_subdirectory(./include/pybind11)

set(CMAKE_CUDA_FLAGS "-O3 -Wall")

cuda_add_library(_pca SHARED
    ./src/test.cpp
    ./include/core/test.cu)

target_link_libraries(_pca
    ${PYTHON_LIBRARIES}
    cudart
    cusparse
)

target_include_directories(_pca PUBLIC ${CMAKE_SOURCE_DIR}/include/pybind11/include)
set_target_properties(_pca PROPERTIES PREFIX "")
