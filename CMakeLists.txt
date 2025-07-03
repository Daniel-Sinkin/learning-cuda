cmake_minimum_required(VERSION 3.18)
project(ds LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(ZLIB REQUIRED)
message(STATUS "Zlib include: ${ZLIB_INCLUDE_DIRS}")
message(STATUS "Zlib library: ${ZLIB_LIBRARIES}")

include(FetchContent)
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/external")

FetchContent_Declare(
  cnpy
  GIT_REPOSITORY https://github.com/rogersce/cnpy.git
  GIT_TAG        master
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(cnpy)

target_compile_definitions(cnpy PUBLIC CNPY_USE_ZLIB)

target_include_directories(cnpy
  PUBLIC
    $<BUILD_INTERFACE:${cnpy_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cnpy
  ZLIB::ZLIB
)

find_package(CUDAToolkit REQUIRED)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(STATUS "Defaulting CUDA arch to 80 (A100)")
  set(CMAKE_CUDA_ARCHITECTURES 80)
else()
  message(STATUS "CUDA arch = ${CMAKE_CUDA_ARCHITECTURES}")
endif()

add_executable(ds main.cu)

target_include_directories(ds PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(ds PRIVATE
  cnpy
  ZLIB::ZLIB
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