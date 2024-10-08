#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh;

if [ "$MODE" = "preprocessing" ]; then
    echo "Running preprocessing"
    cd /workspace/; conda activate fishcluster_toolbox; python -m processing.data_processing 
elif [ "$MODE" = "training" ]; then
    echo "Running training"
    cd /workspace/; conda activate fishcluster_toolbox; python train.py
elif [ "$MODE" = "downstream" ]; then
    echo "Running downstream analyses"
    cd /workspace/; conda activate fishcluster_toolbox; python -m downstream_analyses
else
    echo "Unknown MODE: $MODE"
    exit 1
fi