cmake_minimum_required(VERSION 2.8.3)
project(eus_cuda_matrix)

find_package(CUDA REQUIRED)
#set(CUDA_SEPARABLE_COMPILATION ON) # separate compile is very difficult ...

include_directories("/usr/local/cuda/samples/common/inc")

set(CUDA_NVCC_FLAGS "-arch=sm_50; -Xcompiler -fPIC")
cuda_compile(matrixMul_obj src/matrixMul.cu)

cuda_add_library(eus_cuda_matrix SHARED ${matrixMul_obj})
#cuda_add_library(eus_cuda_matrix SHARED eus_cuda_matrix.cu)

add_executable(test_matrixMul src/test_matrixMul.c)
target_link_libraries(test_matrixMul eus_cuda_matrix)
