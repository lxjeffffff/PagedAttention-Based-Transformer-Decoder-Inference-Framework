cmake_minimum_required(VERSION 3.18)
project(LLMDecoder)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# Find pybind11 and Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/decoder
    ${PROJECT_SOURCE_DIR}/attention
    ${PROJECT_SOURCE_DIR}/attention_cpu
    ${PROJECT_SOURCE_DIR}/kv_cache
)

# Source files
file(GLOB_RECURSE SRC_CPP CONFIGURE_DEPENDS
    src/*.cpp
    decoder/*.cpp
    decoder/*.cu
    attention_cpu/*.cpp
    kv_cache/*.cpp
)

# Build the pybind11 module
pybind11_add_module(llm_decoder ${SRC_CPP})

# Link CUDA runtime
target_link_libraries(llm_decoder PRIVATE CUDA::cudart)
