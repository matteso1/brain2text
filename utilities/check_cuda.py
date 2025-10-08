import torch

print("="*60)
print("PyTorch CUDA Check")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Test GPU memory
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU Memory: {gpu_mem:.1f} GB")

    # Quick test
    print("\nRunning quick GPU test...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = x @ y
    print("✓ GPU computation successful!")
else:
    print("\nCUDA NOT available - will use CPU only")
    print("Install CUDA PyTorch with:")
    print("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124")

print("="*60)
