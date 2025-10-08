"""
Quick GPU speed test - verify training will use GPU
"""
import torch
import time

print("GPU Speed Test")
print("=" * 60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("\nRunning matrix multiplication test...")
    print("Watch nvidia-smi in another terminal to see GPU usage spike!")

    # Large matrix multiplication on GPU
    size = 10000
    x = torch.randn(size, size, device='cuda')
    y = torch.randn(size, size, device='cuda')

    start = time.time()
    for _ in range(10):
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
    gpu_time = time.time() - start

    print(f"\nGPU Time: {gpu_time:.2f} seconds")
    print(f"GPU Memory Used: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")

    # CPU comparison
    print("\nComparing to CPU...")
    x_cpu = torch.randn(size, size, device='cpu')
    y_cpu = torch.randn(size, size, device='cpu')

    start = time.time()
    for _ in range(10):
        z_cpu = torch.matmul(x_cpu, y_cpu)
    cpu_time = time.time() - start

    print(f"CPU Time: {cpu_time:.2f} seconds")
    print(f"\nSpeedup: {cpu_time / gpu_time:.1f}x faster on GPU")
    print("\n✓ GPU is working correctly!")
else:
    print("ERROR: CUDA not available!")
    print("Training will be VERY slow on CPU.")

print("=" * 60)
