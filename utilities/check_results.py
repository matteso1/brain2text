"""
Check training results after it completes
"""
import torch
import os

print("Checking Training Results")
print("=" * 60)

ckpt_path = '../runs/baseline_checkpoints/gru_baseline.pt'

if not os.path.exists(ckpt_path):
    print(f"❌ Checkpoint not found at {ckpt_path}")
    print("Training may still be running or failed.")
    print("\nRun: python monitor_training.py")
    exit()

print(f"✓ Checkpoint found: {ckpt_path}")

# Load checkpoint
print("\nLoading checkpoint...")
try:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    print("\n✓ Checkpoint loaded successfully!")
    print(f"  - Config: {list(ckpt.keys())}")
    print(f"  - Model state dict keys: {len(ckpt['model'].keys())} layers")
    print(f"  - Normalization: mean shape {ckpt['mean'].shape}, std shape {ckpt['std'].shape}")
    print(f"  - Sessions: {len(ckpt['session_to_day'])} total")

    print("\n" + "=" * 60)
    print("SUCCESS! Model trained and saved.")
    print("\nNext steps:")
    print("1. Setup language model (5-gram) on HPC/Colab")
    print("2. Run inference to generate predictions")
    print("3. Submit to Kaggle leaderboard")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ Error loading checkpoint: {e}")
    print("Training may have failed or is still in progress.")
