"""
Train baseline GRU model with phoneme targets
Following Stanford's RNN baseline approach
"""
import os, random, numpy as np, torch, torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.phoneme_dataset import PhonemeDataset, collate_phoneme_batch, get_session_mapping
from src.rnn_model import GRUDecoder
from src.utils import compute_normalization


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def compute_phoneme_error_rate(logits, targets, target_lens, blank_id=0):
    """
    Compute phoneme error rate from CTC logits.
    Simple greedy CTC decode + edit distance.
    """
    # Greedy decode
    preds = logits.argmax(dim=-1).cpu().numpy()  # (B, T)
    errors = 0
    total_phonemes = 0

    for b in range(preds.shape[0]):
        # CTC collapse
        prev = None
        pred_ids = []
        for t in range(preds.shape[1]):
            p = int(preds[b, t])
            if p == blank_id:
                prev = p
                continue
            if prev != p:
                pred_ids.append(p)
            prev = p

        # Get reference
        target_len = int(target_lens[b])
        ref_ids = targets[b, :target_len].cpu().numpy().tolist()

        # Compute edit distance
        n, m = len(ref_ids), len(pred_ids)
        if n == 0:
            continue

        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_ids[i-1] == pred_ids[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        errors += dp[n][m]
        total_phonemes += n

    return errors / max(total_phonemes, 1)


def train_baseline_gru(config_path='configs/rnn_args.yaml', num_batches=None):
    """
    Train baseline GRU model following Stanford's approach.

    Args:
        config_path: Path to config yaml
        num_batches: Override number of training batches (None = use config)
    """
    # Load config
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    seed = config.get('seed', 10)
    set_seed(seed)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Get session mapping
    session_to_day, sessions = get_session_mapping(config_path)
    n_days = len(sessions)
    print(f"Found {n_days} sessions")

    # Model config
    model_config = config['model']
    dataset_config = config['dataset']

    n_classes = dataset_config['n_classes']
    batch_size = dataset_config['batch_size']
    data_root = dataset_config['dataset_dir']

    # Fix relative path if needed
    if data_root.startswith('../'):
        data_root = data_root[3:]  # Remove '../'

    print(f"Data root: {data_root}")

    # Compute normalization
    print("Computing normalization statistics...")
    mean, std = compute_normalization(data_root)

    # Create datasets
    print("Loading datasets...")
    aug_config = dataset_config['data_transforms']

    train_ds = PhonemeDataset(
        data_root, 'train', session_to_day,
        augment=True, aug_config=aug_config,
        mean=mean, std=std
    )

    val_ds = PhonemeDataset(
        data_root, 'val', session_to_day,
        augment=False, aug_config=None,
        mean=mean, std=std
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataset_config.get('num_dataloader_workers', 4),
        collate_fn=collate_phoneme_batch,
        pin_memory=(device == 'cuda')
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_phoneme_batch
    )

    # Create model
    print("Creating GRU model...")
    model = GRUDecoder(
        neural_dim=model_config['n_input_features'],
        n_units=model_config['n_units'],
        n_days=n_days,
        n_classes=n_classes,
        rnn_dropout=model_config['rnn_dropout'],
        input_dropout=model_config['input_network']['input_layer_dropout'],
        n_layers=model_config['n_layers'],
        patch_size=model_config['patch_size'],
        patch_stride=model_config['patch_stride']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Loss and optimizer
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    # Separate optimizer for day layers vs main model
    day_params = list(model.day_weights.parameters()) + list(model.day_biases.parameters())
    main_params = [p for n, p in model.named_parameters()
                   if not n.startswith('day_weights') and not n.startswith('day_biases')]

    optimizer = torch.optim.AdamW([
        {'params': main_params, 'lr': config['lr_max'], 'weight_decay': config['weight_decay']},
        {'params': day_params, 'lr': config['lr_max_day'], 'weight_decay': config['weight_decay_day']}
    ], betas=(config['beta0'], config['beta1']), eps=config['epsilon'])

    # Learning rate scheduler
    total_batches = num_batches or config['num_training_batches']
    warmup_steps = config['lr_warmup_steps']

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_batches - warmup_steps)
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            return config['lr_min'] / config['lr_max'] + (1 - config['lr_min'] / config['lr_max']) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler
    use_amp = config.get('use_amp', True) and device == 'cuda'
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Setup logging
    log_dir = config.get('output_dir', 'runs/baseline_training')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training_log.txt')

    def log_print(msg):
        """Print and write to log file"""
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
            f.flush()

    log_print(f"\n{'='*80}")
    log_print(f"Training for {total_batches} batches...")
    log_print(f"Logs saved to: {log_file}")
    log_print(f"Progress: Step X/{total_batches} | Loss: X.XXXX | LR: X.XXXXXX")
    log_print(f"{'='*80}\n")

    model.train()
    step = 0
    train_losses = []
    train_iter = iter(train_dl)

    batches_per_log = config.get('batches_per_train_log', 200)
    batches_per_val = config.get('batches_per_val_step', 2000)

    # Progress bar for entire training
    pbar = tqdm(total=total_batches, desc="Training Progress", unit="batch")

    while step < total_batches:
        try:
            x, y, x_lens, y_lens, day_idxs, _, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            x, y, x_lens, y_lens, day_idxs, _, _ = next(train_iter)

        x = x.to(device)
        day_idxs = day_idxs.to(device)
        y_lens = y_lens.to(device)

        # Flatten targets for CTC loss
        targets = []
        for i in range(y.size(0)):
            targets.extend(y[i, :y_lens[i]].tolist())
        targets = torch.tensor(targets, dtype=torch.long, device=device)

        # Forward
        autocast_context = torch.amp.autocast('cuda', enabled=use_amp) if hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast(enabled=use_amp)
        with autocast_context:
            logits = model(x, day_idxs)  # (B, T', n_classes)

            # Compute output lengths after patching
            if model.patch_size > 0:
                out_lens = ((x_lens - model.patch_size) // model.patch_stride + 1).clamp(min=1)
            else:
                out_lens = x_lens

            # CTC loss expects (T, B, C)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = ctc_loss(log_probs, targets, out_lens, y_lens)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm_clip_value'])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_losses.append(loss.item())
        step += 1
        pbar.update(1)
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})

        # Logging
        if step % batches_per_log == 0:
            avg_loss = np.mean(train_losses[-batches_per_log:])
            lr = optimizer.param_groups[0]['lr']
            msg = f"Step {step}/{total_batches} | Loss: {avg_loss:.4f} | LR: {lr:.6f}"
            log_print(msg)

        # Validation
        if step % batches_per_val == 0 or step == total_batches:
            log_print(f"\nValidating at step {step}...")
            model.eval()
            val_pers = []

            with torch.no_grad():
                for x, y, x_lens, y_lens, day_idxs, _, _ in tqdm(val_dl, desc="Val", leave=False):
                    x = x.to(device)
                    day_idxs = day_idxs.to(device)

                    logits = model(x, day_idxs)
                    per = compute_phoneme_error_rate(logits, y, y_lens)
                    val_pers.append(per)

            avg_per = np.mean(val_pers)
            log_print(f"Validation PER: {avg_per*100:.2f}%")
            log_print("="*80)

            # Save intermediate checkpoint
            ckpt_path = os.path.join(log_dir, f'checkpoint_step{step}.pt')
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'config': config,
                'mean': mean,
                'std': std,
                'session_to_day': session_to_day,
                'val_per': avg_per
            }, ckpt_path)
            log_print(f"Saved checkpoint: {ckpt_path}\n")

            model.train()

    pbar.close()

    # Save final model
    log_print("\n" + "="*80)
    log_print("Saving final model...")
    os.makedirs('runs/baseline_checkpoints', exist_ok=True)
    final_path = 'runs/baseline_checkpoints/gru_baseline.pt'
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'mean': mean,
        'std': std,
        'session_to_day': session_to_day
    }, final_path)

    log_print(f"Final model saved to: {final_path}")
    log_print("Training complete!")
    log_print("="*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/rnn_args.yaml')
    parser.add_argument('--num-batches', type=int, default=None, help='Override number of batches')
    args = parser.parse_args()

    train_baseline_gru(args.config, args.num_batches)
