cmake_minimum_required(VERSION 3.20)
project(ECE408_in_thrust LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(cxxopts REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(MP1)
add_subdirectory(MP2)
add_subdirectory(MP3)
add_subdirectory(MP4)
