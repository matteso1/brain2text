"""
Check GPU usage for training
"""
import subprocess
import time
import os

print("GPU Monitor")
print("=" * 80)

# Check if CUDA is available in PyTorch
try:
    import torch
    print(f"\n1. PyTorch CUDA Status:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Current GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"   Max GPU Memory: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
except Exception as e:
    print(f"   Error: {e}")

# Check nvidia-smi
print(f"\n2. NVIDIA-SMI Output:")
print("-" * 80)
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    print(result.stdout)
except Exception as e:
    print(f"   Could not run nvidia-smi: {e}")
    print("   Try running manually: nvidia-smi")

# Check for Python processes using GPU
print("\n3. Python Processes:")
print("-" * 80)
try:
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'],
                          capture_output=True, text=True, timeout=5)
    print(result.stdout)
except Exception as e:
    print(f"   Could not query GPU processes: {e}")

print("\n" + "=" * 80)
print("How to verify GPU is being used during training:")
print("1. Run: nvidia-smi -l 1")
print("   (Updates every 1 second, shows GPU utilization %)")
print("2. Look for:")
print("   - GPU-Util should be 80-100%")
print("   - Memory-Usage should be ~10-14 GB")
print("   - Python process listed at bottom")
print("=" * 80)
