#!/bin/bash

# Overcooked AI Evaluation Script
# This script provides different ways to run evaluations on your JSON experiment logs

python_dir=$(which python)

echo "==============================================="
echo "Overcooked AI Evaluation Options"
echo "==============================================="

# Check if custom directory is provided as argument
if [ $# -eq 1 ]; then
    CUSTOM_DIR="$1"
    echo "Running evaluation on custom directory: $CUSTOM_DIR"
    echo "-----------------------------------------------"
    
    # Option 1: Custom directory with auto-detection (handles 3-level structure)
    echo "1. Running evaluation with auto order detection..."
    /usr/bin/env ${python_dir} -- evaluation.py --test_mode custom_dir --log_dir "$CUSTOM_DIR" --order AUTO
    
    echo "2. Processing results..."
    /usr/bin/env ${python_dir} -- organize_result.py --custom_dir "$CUSTOM_DIR"
    
    echo "3. Converting results by levels..."
    /usr/bin/env ${python_dir} -- convert_result.py --custom_dir "$CUSTOM_DIR"
    
else
    echo "No custom directory specified. Running default evaluation..."
    echo "Usage: ./evaluation.sh [custom_directory_path]"
    echo "-----------------------------------------------"
    
    # Original default behavior
    echo "1. Running built-in evaluation..."
    /usr/bin/env ${python_dir} -- evaluation.py --test_mode build_in
    
    echo "2. Processing results..."
    /usr/bin/env ${python_dir} -- organize_result.py
    
    echo "3. Converting results by levels..."
    /usr/bin/env ${python_dir} -- convert_result.py
fi

echo "==============================================="
echo "Evaluation Complete!"
echo "===============================================" 
