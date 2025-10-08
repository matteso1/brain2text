"""
Quick training test with minimal settings to verify everything works
"""
from src.train import train_conformer_ctc

if __name__ == "__main__":
    print("Running quick training test...")
    print("This will train a tiny model for 1 epoch to verify everything works.")
    print()

    train_conformer_ctc(
        data_root='data',
        d_model=64,         # Tiny model
        num_blocks=2,       # Only 2 blocks
        nhead=2,            # Fewer heads
        vocab_size=200,     # Smaller vocab
        batch_size=4,       # Small batch
        epochs=1,           # Just 1 epoch
        lr=1e-3,
        num_workers=0,      # No multiprocessing for simplicity
        device='cuda',       # Force CPU for testing (changed to cuda because im not a loser)
        seed=42
    )

    print()
    print("Training test completed successfully!")
    print("You can now train with full settings using: python -m src.train")
