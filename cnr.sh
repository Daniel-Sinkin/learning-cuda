#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load modules if not already loaded
load_module_if_not_loaded() {
    local module_name=$1
    if ! module is-loaded "$module_name" &>/dev/null; then
        module load "$module_name"
    fi
}

load_module_if_not_loaded CMake/3.29.3
load_module_if_not_loaded GCC/13.3.0
load_module_if_not_loaded CUDA/12

# Use paths relative to script location
BUILD_DIR="${SCRIPT_DIR}/build"
BINARY="ds"  # Changed from batched_gemm to ds

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    echo "Running CMake configuration..."
    cmake ..
    if [ $? -ne 0 ]; then
        echo "CMake configuration failed."
        exit 1
    fi
else
    cd "$BUILD_DIR"
fi

# Run make and time it
echo "Building project..."
start_time=$(date +%s%N)

if ! make; then
    echo "Build failed. Aborting."
    exit 1
fi

end_time=$(date +%s%N)
elapsed_ns=$((end_time - start_time))
elapsed_sec=$(awk "BEGIN {printf \"%.3f\", $elapsed_ns / 1000000000}")

echo "Build completed in $elapsed_sec seconds."

# Check if binary exists and is executable
if [ ! -x "$BINARY" ]; then
    echo "Error: binary '$BINARY' not found after building."
    echo "Available executables in build directory:"
    ls -la *.exe 2>/dev/null || ls -la * | grep -E '^-rwx'
    exit 1
fi

# Run the binary
echo "Running ./$BINARY..."
./$BINARY