cmake_minimum_required(VERSION 2.8.3)
project(separate_compilation)

# CUDA PACKAGE
find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)

# COMPILE CU FILES
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
set(CUDA_NVCC_FLAGS "-arch=sm_20")
CUDA_COMPILE(app main.cu particle.cu v3.cu)

# COMPILE AND LINK
cuda_add_executable(app main.cu particle.cu v3.cu)
