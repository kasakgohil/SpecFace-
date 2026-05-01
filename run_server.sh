#!/bin/bash

# Exit on error
set -e

echo "======================================"
echo " Setting up environment for Capstone"
echo "======================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "python3 could not be found. Please install Python 3."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "======================================"
echo " Setup Complete! Select an option:"
echo " 1) Run Data Setup (setup_data.py)"
echo " 2) Train Baseline ResNet"
echo " 3) Train Global FFT Models"
echo " 4) Train Blockwise FFT Models"
echo " 5) Train Blockwise FFT Models (with PCA)"
echo " 6) Train Grad-CAM Enriched Blockwise FFT Models"
echo " 7) Run ALL Options Sequentially (Overnight Run)"
echo "======================================"

if [ -z "$1" ]; then
    read -p "Enter choice (1-7): " choice
else
    choice=$1
    echo "Running option $choice..."
fi

if [ "$choice" != "1" ]; then
    read -p "Which GPU ID would you like to use? (e.g. 6): " gpu_id
    # Default to 0 if empty
    gpu_id=${gpu_id:-0}
fi

case $choice in
    1)
        CMD="python -u scripts/setup_data.py"
        ;;
    2)
        CMD="python -u scripts/train_resnet.py"
        ;;
    3)
        CMD="python -u scripts/train_fft_global.py"
        ;;
    4)
        CMD="python -u scripts/train_fft_blockwise.py"
        ;;
    5)
        CMD="python -u scripts/train_fft_blockwise.py --use-pca"
        ;;
    6)
        CMD="python -u scripts/train_fft_blockwise.py --use-gradcam"
        ;;
    7)
        CMD="python -u scripts/setup_data.py && python -u scripts/train_resnet.py && python -u scripts/train_fft_global.py && python -u scripts/train_fft_blockwise.py && python -u scripts/train_fft_blockwise.py --use-pca && python -u scripts/train_fft_blockwise.py --use-gradcam"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

if [ "$choice" == "1" ]; then
    echo "Running data setup directly (not in background)..."
    eval "$CMD"
else
    echo "Starting training in the background on GPU $gpu_id..."
    # Explicitly export the GPU ID inside the subshell so all chained scripts inherit it
    eval "nohup bash -c 'export CUDA_VISIBLE_DEVICES=$gpu_id; $CMD' >> training_output.log 2>&1 &"
    echo "Process started successfully! You can now close this terminal."

    echo "To view live training logs, type: tail -f training_output.log"
fi
