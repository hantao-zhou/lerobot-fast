cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pi0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_library(pi0 STATIC
    src/pi0.cpp
)

target_include_directories(pi0 PUBLIC include)
target_link_libraries(pi0 PUBLIC ${TORCH_LIBRARIES})

pybind11_add_module(pi0_cpp src/bindings.cpp)
target_link_libraries(pi0_cpp PRIVATE pi0 ${TORCH_LIBRARIES})

