"""
Quick sanity check to verify basic functionality
"""
import torch
from src.tokenization import SimpleSubwordTokenizer
from src.utils import compute_wer, normalize_text_for_wer


def test_tokenizer():
    print("Testing tokenizer...")

    # Train on sample texts
    texts = [
        "Hello world",
        "Hello there",
        "The quick brown fox",
        "The quick red fox"
    ]

    tokenizer = SimpleSubwordTokenizer()
    tokenizer.train(texts, vocab_size=100, min_freq=1)

    print(f"Vocabulary size: {len(tokenizer.vocab)}")

    # Test encode/decode
    for text in texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"Original: '{text}' -> Encoded: {encoded[:10]}... -> Decoded: '{decoded}'")
        assert decoded == text, f"Decode mismatch: {text} != {decoded}"

    print("[PASS] Tokenizer test passed!")


def test_wer():
    print("\nTesting WER calculation...")

    test_cases = [
        ("hello world", "hello world", 0.0),
        ("hello world", "hello", 0.5),
        ("the quick brown fox", "the quick red fox", 0.25),
        ("Hello, World!", "hello world", 0.0),  # Should normalize
    ]

    for ref, hyp, expected_wer in test_cases:
        wer = compute_wer(ref, hyp)
        print(f"REF: '{ref}' | HYP: '{hyp}' | WER: {wer:.2f} (expected {expected_wer:.2f})")
        assert abs(wer - expected_wer) < 0.01, f"WER mismatch: {wer} != {expected_wer}"

    print("[PASS] WER test passed!")


def test_data_loading():
    print("\nTesting data loading...")

    try:
        from src.data import NeuralTextDataset

        # Create a dummy tokenizer
        tokenizer = SimpleSubwordTokenizer()
        tokenizer.train(["hello world"], vocab_size=100, min_freq=1)

        # Try loading train data
        ds = NeuralTextDataset('data', 'train', tokenizer, warn_empty=True)
        print(f"Found {len(ds)} training samples")

        if len(ds) > 0:
            # Test loading one sample
            x, y, text, filename, trial = ds[0]
            print(f"Sample 0: x.shape={x.shape}, y.shape={y.shape}, text='{text[:50]}...'")
            print("[PASS] Data loading test passed!")
        else:
            print("[WARN] No training data found. Check your data directory.")

    except Exception as e:
        print(f"[FAIL] Data loading test failed: {e}")


def test_model():
    print("\nTesting model...")

    try:
        from src.model import ConformerRNNT

        model = ConformerRNNT(
            in_dim=512,
            d_model=64,  # Small for testing
            num_blocks=2,
            nhead=2,
            vocab_size=100
        )

        # Test forward pass
        B, T, D = 2, 100, 512
        x = torch.randn(B, T, D)
        x_lens = torch.tensor([100, 80])

        # Test encoder
        enc, enc_lens = model.forward_encoder(x, x_lens)
        print(f"Encoder output: {enc.shape}, lengths: {enc_lens}")

        # Test CTC head
        logits = model.ctc_head(enc)
        print(f"CTC logits: {logits.shape}")

        print("[PASS] Model test passed!")

    except Exception as e:
        print(f"[FAIL] Model test failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running basic sanity checks...")
    print("=" * 60)

    test_tokenizer()
    test_wer()
    test_data_loading()
    test_model()

    print("\n" + "=" * 60)
    print("All basic tests completed!")
    print("=" * 60)
