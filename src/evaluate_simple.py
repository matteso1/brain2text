"""
Simple evaluation script that computes Phoneme Error Rate (PER) without language model.
This is much faster and doesn't require Redis/LM server setup.
"""
import os
import torch
import numpy as np
import yaml
import argparse
from tqdm import tqdm
import jiwer

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phoneme_dataset import PhonemeDataset, collate_phoneme_batch
from src.rnn_model import GRUDecoder
from torch.utils.data import DataLoader

# Phoneme mapping (official order)
PHONEME_LIST = [
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    'SIL',  # Silence token at index 40
]

def greedy_ctc_decode(logits_BTV):
    """
    Greedy CTC decoding - collapse repeated tokens and remove blanks.
    
    Args:
        logits_BTV: (B, T, V) logits from model
    
    Returns:
        List of decoded phoneme sequences (as strings)
    """
    pred = logits_BTV.argmax(-1).cpu().numpy()  # (B, T)
    decoded_sequences = []
    blank_id = 0
    
    for b in range(pred.shape[0]):
        prev = None
        phonemes = []
        
        for t in range(pred.shape[1]):
            token_id = int(pred[b, t])
            
            # Skip blanks
            if token_id == blank_id:
                prev = token_id
                continue
            
            # Skip repeated tokens (CTC collapse)
            if prev == token_id:
                continue
            
            # Add phoneme
            phonemes.append(PHONEME_LIST[token_id])
            prev = token_id
        
        # Join phonemes with spaces
        decoded_sequences.append(' '.join(phonemes))
    
    return decoded_sequences


def evaluate(checkpoint_path, split, config_path):
    """
    Evaluate model on a given split (val or test).
    
    Args:
        checkpoint_path: Path to trained model .pt file
        split: 'val' or 'test'
        config_path: Path to config YAML for session mapping
    
    Returns:
        Dictionary with metrics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING: {checkpoint_path}")
    print(f"Split: {split}")
    print(f"{'='*60}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 1. Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model_cfg = config['model']
    dataset_cfg = config['dataset']
    print("[OK] Checkpoint loaded\n")
    
    # 2. Load session mapping from config
    with open(config_path, 'r') as f:
        map_config = yaml.safe_load(f)
    sessions = map_config['dataset']['sessions']
    session_to_day = {sess: idx for idx, sess in enumerate(sessions)}
    print(f"[OK] Loaded session mapping ({len(sessions)} sessions)\n")
    
    # 3. Build model
    print("Building model...")
    model = GRUDecoder(
        neural_dim=model_cfg['n_input_features'],
        n_units=model_cfg['n_units'],
        n_layers=model_cfg['n_layers'],
        n_days=len(sessions),
        n_classes=dataset_cfg['n_classes'],
        rnn_dropout=model_cfg['rnn_dropout'],
        input_dropout=model_cfg['input_network']['input_layer_dropout'],
        patch_size=model_cfg['patch_size'],
        patch_stride=model_cfg['patch_stride'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("[OK] Model loaded and ready\n")
    
    # 4. Prepare dataset
    print("Loading dataset...")
    dataset = PhonemeDataset(
        dataset_cfg['dataset_dir'].replace('../', ''),
        split,
        session_to_day,
        augment=False  # No augmentation for eval
    )
    
    if len(dataset) == 0:
        print(f"ERROR: No data found for split '{split}'!")
        print(f"Check that data exists at: {dataset_cfg['dataset_dir']}")
        return None
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one sentence at a time
        shuffle=False,
        num_workers=0,
        collate_fn=collate_phoneme_batch
    )
    print(f"[OK] Loaded {len(dataset)} samples\n")
    
    # 5. Run evaluation
    print(f"Running evaluation on {len(dataset)} samples...\n")
    
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for x, y, x_lens, y_lens, day_idxs, sess_names, trial_keys, sentences in tqdm(dataloader, desc="Evaluating"):
            # Forward pass
            x = x.to(device)
            day_idxs = day_idxs.to(device)
            
            logits = model(x, day_idxs)  # (B, T, 41)
            
            # Decode phoneme sequence
            decoded_phonemes = greedy_ctc_decode(logits)
            
            # Ground truth phonemes
            # y contains token IDs, convert to phoneme strings
            y_np = y.cpu().numpy()
            for b in range(y.shape[0]):
                gt_phonemes = []
                for token_id in y_np[b]:
                    if token_id == 0:  # Pad token
                        break
                    if 0 <= token_id < len(PHONEME_LIST):
                        gt_phonemes.append(PHONEME_LIST[token_id])
                
                all_ground_truths.append(' '.join(gt_phonemes))
                all_predictions.append(decoded_phonemes[b])
    
    # 6. Calculate Phoneme Error Rate (PER)
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60 + "\n")
    
    # Clean up phoneme sequences (remove SIL, normalize spacing)
    def clean_phonemes(phoneme_str):
        # Remove SIL tokens and extra spaces
        cleaned = phoneme_str.replace('SIL', '').strip()
        cleaned = ' '.join(cleaned.split())  # Normalize whitespace
        return cleaned if cleaned else 'EMPTY'
    
    clean_predictions = [clean_phonemes(p) for p in all_predictions]
    clean_ground_truths = [clean_phonemes(gt) for gt in all_ground_truths]
    
    # Compute PER using jiwer (treats phonemes like words)
    per = jiwer.wer(clean_ground_truths, clean_predictions)
    
    print(f"Phoneme Error Rate (PER): {per * 100:.2f}%")
    print(f"Total samples: {len(all_predictions)}")
    print()
    
    # Show examples
    print("="*60)
    print("EXAMPLE PREDICTIONS (first 10)")
    print("="*60 + "\n")
    
    for i in range(min(10, len(all_predictions))):
        print(f"Sample {i+1}:")
        print(f"  Ground Truth: {clean_ground_truths[i][:100]}")
        print(f"  Prediction:   {clean_predictions[i][:100]}")
        print()
    
    results = {
        'per': per,
        'num_samples': len(all_predictions),
        'predictions': all_predictions,
        'ground_truths': all_ground_truths,
    }
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple evaluation script - computes PER without language model"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/rnn_official_exact.yaml',
        help='Path to config file for session mapping'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['val', 'test'],
        help='Data split to evaluate on'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"ERROR: Config not found at {args.config}")
        sys.exit(1)
    
    results = evaluate(args.checkpoint, args.split, args.config)
    
    if results:
        print("\n" + "="*60)
        print("[SUCCESS] Evaluation complete!")
        print("="*60)

