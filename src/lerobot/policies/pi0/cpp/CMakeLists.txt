cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pi0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_library(pi0 STATIC
    src/pi0.cpp
    src/model.cpp
)

target_include_directories(pi0 PUBLIC include)
target_link_libraries(pi0 PUBLIC ${TORCH_LIBRARIES})

pybind11_add_module(pi0_cpp src/bindings.cpp)
target_link_libraries(pi0_cpp PRIVATE pi0 ${TORCH_LIBRARIES})

# FlexAttention CUDA extension
pybind11_add_module(flex_attention_cpp
    src/flex_attention.cpp
    src/cuda/flex_attention.cu
)
target_link_libraries(flex_attention_cpp PRIVATE ${TORCH_LIBRARIES})
set_target_properties(flex_attention_cpp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET flex_attention_cpp PROPERTY CUDA_ARCHITECTURES "70;75;80;86")

