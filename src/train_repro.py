"""
This script is a clean, ground-up reproduction of the official NEJM baseline
training procedure. It is designed to be a definitive test to isolate any
potential bugs in the original `train_baseline.py` script.

Key principles of this reproduction:
1.  **Direct Re-implementation:** The core training loop, optimizer, scheduler,
    and loss calculation are directly based on the official `rnn_trainer.py`.
2.  **Clean Dependencies:** It only uses the verified `GRUDecoder` from
    `src/rnn_model.py` and the existing `PhonemeDataset`.
3.  **GPU-side Augmentations:** Data augmentations are performed on the GPU
    inside the training loop, exactly as in the official implementation.
4.  **Minimalism:** No complex logging or monitoring classes, just pure,
    essential training logic to ensure stability.
"""

import sys
import os
import random
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phoneme_dataset import PhonemeDataset, collate_phoneme_batch, get_session_mapping
from src.rnn_model import GRUDecoder
from src.data_augmentations import gauss_smooth


def set_seed(s=42):
    """Set all random seeds for reproducibility."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def greedy_ctc_decode(logits, blank_id=0):
    """Simple greedy CTC decode."""
    preds = torch.argmax(logits, dim=-1)
    decoded_seqs = []
    for pred in preds:
        unique_seq = torch.unique_consecutive(pred)
        decoded_seq = unique_seq[unique_seq != blank_id]
        decoded_seqs.append(decoded_seq.cpu().numpy())
    return decoded_seqs


def compute_per(decoded, targets, target_lens):
    """Compute Phoneme Error Rate."""
    total_errors = 0
    total_phonemes = 0
    for d, t, tl in zip(decoded, targets, target_lens):
        ref = t[:tl].cpu().numpy()
        # Simple Levenshtein distance
        n, m = len(ref), len(d)
        if n == 0: continue
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1): dp[i][0] = i
        for j in range(m + 1): dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref[i - 1] == d[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        total_errors += dp[n][m]
        total_phonemes += n
    return total_errors / max(total_phonemes, 1)


def apply_augmentations_on_gpu(features, x_lens, config, device):
    """
    Apply augmentations on the GPU to match official implementation.
    This is a critical step for performance and reproducibility.
    """
    if config['white_noise_std'] > 0:
        features += torch.randn_like(features, device=device) * config['white_noise_std']
    
    if config['constant_offset_std'] > 0:
        offset = torch.randn(features.size(0), 1, features.size(2), device=device) * config['constant_offset_std']
        features += offset

    if config['random_cut'] > 0:
        # A single cut amount is applied to all samples in the batch, matching official impl.
        cut = np.random.randint(0, config['random_cut'] + 1)
        if cut > 0:
            features = features[:, cut:, :]
            x_lens = x_lens - cut
            
    if config['smooth_data']:
        features = gauss_smooth(
            inputs=features,
            device=device,
            smooth_kernel_std=config['smooth_kernel_std'],
            smooth_kernel_size=config['smooth_kernel_size']
        )
    return features, x_lens


def create_cosine_lr_scheduler(optimizer, config, num_batches=None):
    """Creates the official cosine learning rate scheduler."""
    lr_max = config['lr_max']
    lr_min = config['lr_min']
    lr_decay_steps = config['num_training_batches'] if num_batches is None else num_batches
    warmup_steps = config['lr_warmup_steps']
    min_lr_ratio = lr_min / lr_max

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if current_step < lr_decay_steps:
            progress = float(current_step - warmup_steps) / float(max(1, lr_decay_steps - warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
        return min_lr_ratio

    # The official implementation has separate schedules for day params, but we simplify
    # as they are identical in the provided config.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_repro(config_path, num_batches=None, resume_from_checkpoint=None):
    print("--- Starting Clean Reproduction of Official Training ---")

    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config.get('seed', 10))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Ensure output and checkpoint directories exist
    os.makedirs(config['output_dir'], exist_ok=True)
    if 'checkpoint_dir' in config:
        os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # 2. Data Loading
    session_to_day, sessions = get_session_mapping(config_path)
    dataset_cfg = config['dataset']
    # IMPORTANT: Augmentations are now done on the GPU, so set `augment=False`
    train_ds = PhonemeDataset(
        dataset_cfg['dataset_dir'].replace('../', ''), 'train', session_to_day, augment=False
    )
    val_ds = PhonemeDataset(
        dataset_cfg['dataset_dir'].replace('../', ''), 'val', session_to_day, augment=False
    )
    train_dl = DataLoader(
        train_ds, batch_size=dataset_cfg['batch_size'], shuffle=True,
        # TUNED FOR STABILITY: For a long overnight run on Windows, keeping the
        # data loader worker processes alive is critical. This prevents the OS
        # from wasting time creating/destroying them for every batch, which is a
        # major cause of inconsistent training speed ("up and down a lot").
        num_workers=dataset_cfg.get('num_dataloader_workers', 4),
        persistent_workers=True,
        collate_fn=collate_phoneme_batch, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=dataset_cfg['batch_size'], shuffle=False,
        num_workers=0, collate_fn=collate_phoneme_batch
    )
    print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")

    # 3. Model Initialization (using the already verified model class)
    model_cfg = config['model']
    model = GRUDecoder(
        neural_dim=model_cfg['n_input_features'],
        n_units=model_cfg['n_units'],
        n_days=len(sessions),
        n_classes=dataset_cfg['n_classes'],
        rnn_dropout=model_cfg['rnn_dropout'],
        input_dropout=model_cfg['input_network']['input_layer_dropout'],
        n_layers=model_cfg['n_layers'],
        patch_size=model_cfg['patch_size'],
        patch_stride=model_cfg['patch_stride'],
    ).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M params.")

    # 4. Optimizer & Scheduler (exactly as per official implementation)
    bias_params = [p for name, p in model.named_parameters() if 'gru.bias' in name or 'out.bias' in name]
    day_params = [p for name, p in model.named_parameters() if 'day_' in name]
    other_params = [p for name, p in model.named_parameters() if 'day_' not in name and 'gru.bias' not in name and 'out.bias' not in name]

    param_groups = [
        {'params': bias_params, 'weight_decay': 0},
        {'params': day_params, 'lr': config['lr_max_day'], 'weight_decay': config['weight_decay_day']},
        {'params': other_params}
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=config['lr_max'], betas=(config['beta0'], config['beta1']),
        eps=config['epsilon'], weight_decay=config['weight_decay'], fused=True
    )
    scheduler = create_cosine_lr_scheduler(optimizer, config, num_batches)
    
    # 5. Loss Function
    # CRITICAL: zero_infinity=False as in official repo
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)

    # 6. Training Loop
    step = 0

    # --- RESUME FROM CHECKPOINT LOGIC ---
    if resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)

        # CRITICAL FIX: Only load the state dicts and step. The 'config' object from the
        # checkpoint is NOT used, ensuring that the current run's config (from the yaml)
        # is the single source of truth for paths and hyperparameters. This prevents
        # "config pollution" from old runs.
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        step = checkpoint['step']
        print(f"Resumed from step {step}. Last validation PER: {checkpoint.get('val_per', 'N/A')}")
    # ------------------------------------

    total_batches = config['num_training_batches'] if num_batches is None else num_batches
    pbar = tqdm(initial=step, total=total_batches, desc="Training")
    train_iter = iter(train_dl)

    # BUG FIX: Removed faulty pbar.update(step). The 'initial=step' argument to tqdm
    # is the correct and only line needed to start the progress bar correctly.

    while step < total_batches:
        try:
            x, y, x_lens, y_lens, day_idxs, _, _, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            x, y, x_lens, y_lens, day_idxs, _, _, _ = next(train_iter)

        model.train()
        
        # Move data to GPU
        x = x.to(device)
        y = y.to(device)
        x_lens = x_lens.to(device)
        y_lens = y_lens.to(device)
        day_idxs = day_idxs.to(device)
        
        # Flatten targets for CTC
        targets_flat = []
        for i in range(y.size(0)):
            targets_flat.extend(y[i, :y_lens[i]].tolist())
        targets_flat = torch.tensor(targets_flat, dtype=torch.long, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config['use_amp']):
            # Apply augmentations on GPU - this is the key fix area
            x, x_lens_aug = apply_augmentations_on_gpu(x, x_lens, dataset_cfg['data_transforms'], device)
            
            # Forward pass
            logits = model(x, day_idxs)
            
            # Calculate output lengths after patching using AUGMENTED lengths
            out_lens = ((x_lens_aug - model.patch_size) // model.patch_stride + 1).clamp(min=1)

            # Check for invalid CTC lengths
            if torch.any(y_lens > out_lens):
                print(f"Warning: Skipping batch {step} due to invalid target length > input length.")
                step += 1
                pbar.update(1)
                continue
                
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = ctc_loss(log_probs, targets_flat, out_lens, y_lens)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss at step {step}. Skipping batch.")
            optimizer.zero_grad(set_to_none=True)
            step += 1
            pbar.update(1)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm_clip_value'])
        
        if not torch.isfinite(grad_norm):
            print(f"Warning: Non-finite grad norm at step {step}. Skipping optimizer step.")
            optimizer.zero_grad(set_to_none=True)
            step += 1
            pbar.update(1)
            continue
            
        optimizer.step()
        scheduler.step()

        pbar.set_postfix({'loss': f"{loss.item():.3f}", 'grad': f"{grad_norm.item():.2f}", 'lr': f"{scheduler.get_last_lr()[0]:.6f}"})
        pbar.update(1)
        step += 1

        # Validation step
        if step % config['batches_per_val_step'] == 0:
            model.eval()
            val_pers = []
            with torch.no_grad():
                for vx, vy, vx_lens, vy_lens, vday_idxs, _, _, _ in tqdm(val_dl, desc="Validating", leave=False):
                    vx = vx.to(device)
                    v_logits = model(vx, vday_idxs.to(device))
                    decoded = greedy_ctc_decode(v_logits)
                    per = compute_per(decoded, vy, vy_lens)
                    val_pers.append(per)
            avg_per = np.mean(val_pers)
            print(f"\n--- Validation at Step {step}: Avg PER = {avg_per*100:.2f}% ---")
            pbar.set_postfix({'loss': f"{loss.item():.3f}", 'grad': f"{grad_norm.item():.2f}", 'lr': f"{scheduler.get_last_lr()[0]:.6f}", 'val_per': f"{avg_per:.3f}"})

            # Save a checkpoint
            checkpoint_dir = config.get('checkpoint_dir', os.path.join(config['output_dir'], 'checkpoints'))
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_per': avg_per,
                'config': config
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")


    pbar.close()
    print("--- Clean Reproduction Training Finished ---")

    # Save the final model
    final_model_path = os.path.join(config['output_dir'], 'final_model.pt')
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/rnn_official_exact.yaml')
    parser.add_argument('--num-batches', type=int, default=None, help='Override number of training batches from config')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume training from.')
    args = parser.parse_args()
    train_repro(args.config, args.num_batches, args.resume_from_checkpoint)
