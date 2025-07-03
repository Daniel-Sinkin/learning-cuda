#!/bin/bash

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps

echo "Attempting to stop MPS..."

# Try graceful shutdown
if [ -e "$CUDA_MPS_PIPE_DIRECTORY/control" ]; then
    echo "Sending quit to MPS control daemon..."
    echo quit | nvidia-cuda-mps-control 2>/dev/null
    sleep 1
fi

# Kill anything left over
MPS_PIDS=$(pgrep -u "$USER" -f nvidia-cuda-mps-control)

if [ -n "$MPS_PIDS" ]; then
    echo "Force killing leftover MPS processes: $MPS_PIDS"
    kill -9 $MPS_PIDS
else
    echo "No active MPS processes found."
fi

# Cleanup
rm -rf /tmp/nvidia-mps /tmp/nvidia-log

echo "MPS daemon fully stopped and cleaned up."
