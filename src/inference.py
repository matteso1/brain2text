import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src.data import NeuralTextDataset, collate_batch
from src.tokenization import SimpleSubwordTokenizer
from src.model import ConformerRNNT
from src.utils import normalize_text_for_wer


def greedy_ctc_decode(logits_BTV, tokenizer):
    """
    Greedy CTC decoding using tokenizer.
    logits: (B, T, V)
    Returns: list of decoded strings
    """
    pred = logits_BTV.argmax(-1).cpu().numpy()  # (B,T)
    texts = []
    blank_id = 0

    for b in range(pred.shape[0]):
        # CTC collapse: remove blanks and repeated tokens
        prev = None
        ids = []
        for t in range(pred.shape[1]):
            p = int(pred[b, t])
            if p == blank_id:
                prev = p
                continue
            if prev != p:
                ids.append(p)
            prev = p

        # Use tokenizer to decode
        text = tokenizer.decode(ids)
        texts.append(text)

    return texts


def generate_submission(
    ckpt_path='runs/checkpoints/conformer_ctc_ema.pt',
    data_root='data',
    output_csv='submission.csv',
    batch_size=16,
    device=None
):
    """
    Generate submission CSV for test set.

    Args:
        ckpt_path: Path to model checkpoint
        data_root: Root directory containing hdf5_data_final
        output_csv: Output CSV path
        batch_size: Batch size for inference
        device: Device to run on (cuda/cpu)
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {ckpt_path}...")
    saved = torch.load(ckpt_path, map_location='cpu')
    vocab = saved['tokenizer']
    tokenizer = SimpleSubwordTokenizer(vocab=vocab)
    mean, std = saved.get('mean'), saved.get('std')

    # Load model
    print("Loading model...")
    model = ConformerRNNT(
        in_dim=512,
        d_model=256,
        num_blocks=12,
        nhead=4,
        p=0.1,
        vocab_size=len(vocab)
    ).to(device)
    model.load_state_dict(saved['model'])
    model.eval()

    # Load test dataset
    print("Loading test dataset...")
    test_ds = NeuralTextDataset(data_root, 'test', tokenizer, mean=mean, std=std, warn_empty=True)

    if len(test_ds) == 0:
        print("ERROR: No test data found! Check your data directory.")
        return

    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 for deterministic ordering
        collate_fn=collate_batch
    )

    # Generate predictions
    print(f"Generating predictions for {len(test_ds)} test samples...")
    predictions = []

    with torch.no_grad():
        for x, y_pad, x_lens, y_lens, texts, file_names, trial_keys in test_dl:
            x, x_lens = x.to(device), x_lens.to(device)

            # Forward pass
            enc, enc_lens = model.forward_encoder(x, x_lens)
            logits = model.ctc_head(enc)  # (B,T,V)

            # Decode
            hyps = greedy_ctc_decode(logits, tokenizer)

            # Normalize predictions for submission
            for i, hyp in enumerate(hyps):
                normalized = normalize_text_for_wer(hyp)
                predictions.append({
                    'file': file_names[i],
                    'trial': trial_keys[i],
                    'prediction': normalized
                })

    print(f"Generated {len(predictions)} predictions")

    # Create submission DataFrame
    # Competition format: id,text where id is 0..1449 in chronological order
    df = pd.DataFrame(predictions)

    # Sort chronologically (by file/session, then trial)
    df = df.sort_values(['file', 'trial']).reset_index(drop=True)

    # Create submission with sequential IDs
    submission = pd.DataFrame({
        'id': range(len(df)),
        'text': df['prediction']
    })

    # Save
    submission.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}")
    print(f"Preview:")
    print(submission.head(10))

    return submission


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate submission for Brain to Text competition')
    parser.add_argument('--ckpt', type=str, default='runs/checkpoints/conformer_ctc_ema.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, default='data',
                        help='Root directory containing hdf5_data_final')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output CSV path')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    generate_submission(
        ckpt_path=args.ckpt,
        data_root=args.data_root,
        output_csv=args.output,
        batch_size=args.batch_size,
        device=args.device
    )
