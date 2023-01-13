# try to use lto
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3 -pthread")

FILE(GLOB UtilsCppFiles ./src/utils/*.cpp)
pybind11_add_module(_mars ./src/core/mars.cpp ${UtilsCppFiles})

