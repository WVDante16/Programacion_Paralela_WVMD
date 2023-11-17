cmake_minimum_required(VERSION 3.26)
project(MemoriaCompartidaCUDA CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(MemoriaCompartidaCUDA main.cu)

set_target_properties(MemoriaCompartidaCUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
