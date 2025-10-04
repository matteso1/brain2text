#!/bin/bash
#
# train.sh - Run brain-to-text training on CHTC GPU node
# This script runs inside the PyTorch Docker container
#

set -e  # Exit on any error
set -x  # Echo all commands (for debugging)

# Force unbuffered output
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "=========================================="
echo "CHTC Brain-to-Text Training Job"
echo "=========================================="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Job Process ID: $1"
echo ""

# Flush output immediately
exec 1> >(stdbuf -o0 cat >&1)
exec 2> >(stdbuf -o0 cat >&2)

# Check GPU availability
echo "Checking for GPU..."
nvidia-smi
echo ""
echo "CUDA devices available:"
python3 -u -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); import sys; sys.stdout.flush()"
echo ""

# Untar code and data
echo "Extracting code tarball..."
tar -xzf brain2text_code.tar.gz
echo "Code extracted successfully."
echo ""

echo "Extracting data tarball (24GB - this may take 2-3 minutes)..."
time tar -xzf brain2text_data.tar.gz
echo "Data extracted successfully."
echo ""

# Verify data structure
echo "Verifying data structure..."
if [ ! -d "data/hdf5_data_final" ]; then
    echo "ERROR: data/hdf5_data_final not found after extraction!"
    echo "Contents of current directory:"
    ls -la
    exit 1
fi

SESSION_COUNT=$(ls -d data/hdf5_data_final/t15.* 2>/dev/null | wc -l)
echo "Found $SESSION_COUNT session directories"
if [ "$SESSION_COUNT" -lt 40 ]; then
    echo "WARNING: Expected 45 sessions, only found $SESSION_COUNT"
fi
echo ""

# Install Python dependencies
echo "Installing Python dependencies (this will show progress)..."
pip install --no-cache-dir h5py scipy pyyaml tqdm jiwer sentencepiece
echo "Dependencies installed successfully."
echo ""

# Set up Python environment
export PYTHONPATH=$PWD:$PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"
echo ""

# Show disk usage before training
echo "Disk usage before training:"
df -h .
echo ""

# Run training with unbuffered output
echo "=========================================="
echo "Starting training (120k batches, ~8 hours)"
echo "=========================================="
echo "Using unbuffered Python for real-time output..."
python3 -u -m src.train_baseline --config configs/rnn_args.yaml

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="

# Show what was created
echo "Checking trained_models directory..."
ls -lh trained_models/baseline_rnn/ || echo "No models found!"
echo ""

# Show disk usage after training
echo "Disk usage after training:"
df -h .
echo ""

# Package output files
echo "Packaging results (checkpoints + logs)..."
if [ -d "trained_models" ]; then
    tar -czf results.tar.gz trained_models/
    RESULT_SIZE=$(ls -lh results.tar.gz | awk '{print $5}')
    echo "Results packaged: results.tar.gz ($RESULT_SIZE)"
else
    echo "WARNING: No trained_models directory found!"
    # Package logs anyway if they exist
    tar -czf results.tar.gz *.log *.out *.err || echo "No files to package"
fi
echo ""

echo "Done! Results will be transferred back to /home/nomatteson/"
echo "=========================================="
