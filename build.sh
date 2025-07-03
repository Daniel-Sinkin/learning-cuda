#!/bin/bash

echo "Loading required modules..."

module load CMake/3.29.3
echo "Loaded CMake: $(cmake --version | head -n1)"

module load GCC/13.3.0
echo "Loaded GCC: $(which g++)"
echo "G++ version: $(g++ --version | head -n1)"

module load CUDA/12
echo "Loaded CUDA: $(nvcc --version | grep release)"

echo ""

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)

case "$GPU_NAME" in
  *"A100"*) ARCH=80 ;;
  *"V100"*) ARCH=70 ;;
  *"P100"*) ARCH=60 ;;
  *"T4"*)   ARCH=75 ;;
  *"RTX 8000"*) ARCH=75 ;;
  *) echo "Unknown or unsupported GPU: $GPU_NAME" >&2; exit 1 ;;
esac

echo "Detected GPU: $GPU_NAME"
echo "Using CUDA architecture: $ARCH"
echo ""

rm -rf build
mkdir build
cd build

echo "Running CMake..."
cmake -DCMAKE_CUDA_ARCHITECTURES=${ARCH} ..

echo "Building project..."
make -j