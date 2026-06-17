#!/bin/bash

# Find Conda base path and source it so 'conda activate' works in non-interactive shells
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate vit_env
    echo "Activated vit_env Conda environment."
else
    echo "Could not find Conda base. Using default system python..."
fi

# Run the Flask server
echo "Starting the VQA Flask server on port 5005..."
python app.py
