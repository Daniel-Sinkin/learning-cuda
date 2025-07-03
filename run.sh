#!/bin/bash
set -e

echo "Starting the run data generation."

echo "Activating python virtual env"
source .venv/bin/activate

echo "Displaying Hardware info"
python3 util/print_hardware_info.py

echo "Building the project"
./build.sh

echo "Generating the Matrices"
mkdir -p build
python3 util/generate_matrices.py -o build/matrices.npz

echo "Compiling and running the project"
./cnr.sh

echo "Finished running the project."