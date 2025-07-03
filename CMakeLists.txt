cmake_minimum_required(VERSION 3.18)
project(ds LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined. Defaulting to 80 (A100).")
    set(CMAKE_CUDA_ARCHITECTURES 80)
else()
    message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Create the main executable
add_executable(ds main.cu)

target_include_directories(ds PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"  # So src/ can be found
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(ds PRIVATE
    CUDA::cublas
    CUDA::cudart
    CUDA::curand
)

target_compile_definitions(ds PRIVATE
    CUDA_ARCH=${CMAKE_CUDA_ARCHITECTURES}
)

set_target_properties(ds PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)