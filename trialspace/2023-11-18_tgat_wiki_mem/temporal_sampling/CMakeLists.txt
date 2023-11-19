cmake_minimum_required(VERSION 3.23)
project(temporal_sampling)

set(CMAKE_CXX_STANDARD 14)

include_directories(src/cpu)
include_directories(src/cuda)

add_executable(temporal_sampling
        src/cpu/searchsorted_cpu_wrapper.cpp
        src/cpu/searchsorted_cpu_wrapper.h
        src/cuda/searchsorted_cuda_kernel.cu
        src/cuda/searchsorted_cuda_kernel.h
        src/cuda/searchsorted_cuda_wrapper.cpp
        src/cuda/searchsorted_cuda_wrapper.h)

# set(CMAKE_PREFIX_PATH /Users/synapse/opt/anaconda3/envs/pytorch-graph/lib/python3.9/site-packages/torch/share/cmake)
set(CMAKE_PREFIX_PATH /home/yuxin/anaconda3/envs/atgnn/bin/cmake)
find_package(Torch REQUIRED)
include_directories(.)
target_link_libraries(temporal_sampling "${TORCH_LIBRARIES}")