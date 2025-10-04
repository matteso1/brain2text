import os, math, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from phoneme_dataset import PhonemeDataset, collate_phoneme_batch, get_session_mapping
from tokenization import SimpleSubwordTokenizer
from utils import compute_normalization, compute_wer
from model import ConformerRNNT
from tqdm import tqdm

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    # Enable cuDNN optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # Enable TF32 on A100 for faster matmuls
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def build_tokenizer(data_root, split='train', vocab_size=2000, max_texts=50000):
    """Build tokenizer from training texts"""
    # Create a minimal tokenizer just for loading data
    temp_vocab = {'<blank>': 0, '<pad>': 1, '<unk>': 2}
    temp_tok = SimpleSubwordTokenizer(vocab=temp_vocab)

    # Read training texts
    ds = NeuralTextDataset(data_root, split, tokenizer=temp_tok, mean=None, std=None, warn_empty=False)

    if len(ds) == 0:
        raise RuntimeError(f"No {split} data found in {data_root}. Cannot build tokenizer.")

    texts = []
    for i in range(min(len(ds), max_texts)):
        _, _, t, _, _ = ds[i]
        texts.append(t)

    # Train proper tokenizer
    tok = SimpleSubwordTokenizer()
    tok.train(texts, vocab_size=vocab_size, min_freq=2)
    return tok

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

def train_conformer_ctc(data_root='data',
                        d_model=256, num_blocks=12, nhead=4, p=0.1,
                        vocab_size=2000, batch_size=8, epochs=5, lr=1e-3,
                        num_workers=4, device=None, seed=42):
    set_seed(seed)

    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        print("To use GPU, install CUDA-enabled PyTorch from https://pytorch.org")
        device = 'cpu'

    print(f"Using device: {device}")

    # Tokenizer from train transcripts
    print("Building tokenizer from training data...")
    tokenizer = build_tokenizer(data_root, split='train', vocab_size=vocab_size)
    print(f"Tokenizer vocabulary size: {len(tokenizer.vocab)}")

    # Normalization on train
    print("Computing normalization statistics...")
    mean, std = compute_normalization(data_root)

    # Datasets
    print("Loading datasets...")
    train_ds = NeuralTextDataset(data_root, 'train', tokenizer, mean=mean, std=std)
    val_ds   = NeuralTextDataset(data_root, 'val',   tokenizer, mean=mean, std=std)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          collate_fn=collate_batch, pin_memory=True,
                          persistent_workers=True if num_workers > 0 else False,
                          prefetch_factor=4 if num_workers > 0 else None)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                          collate_fn=collate_batch, pin_memory=True)

    model = ConformerRNNT(in_dim=512, d_model=d_model, num_blocks=num_blocks, nhead=nhead, p=p,
                          vocab_size=len(tokenizer.vocab)).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)

    # Use newer GradScaler API if available
    use_amp = (device == 'cuda')
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # EMA - only track floating point parameters
    ema = {}
    for k, v in model.state_dict().items():
        if v.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            ema[k] = v.detach().clone()
    ema_decay = 0.999

    def apply_ema():
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in ema:
                    v.copy_(ema[k])

    def update_ema():
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in ema:
                    ema[k].mul_(ema_decay).add_(v.detach(), alpha=1-ema_decay)

    for epoch in range(1, epochs+1):
        model.train()
        total = 0.0
        for x, y_pad, x_lens, y_lens, *_ in tqdm(train_dl, desc=f"Epoch {epoch}"):
            x, x_lens = x.to(device), x_lens.to(device)
            # Build CTC targets (flattened)
            targets = []
            for i in range(y_pad.size(0)):
                yi = y_pad[i, :y_lens[i]].tolist()
                targets.extend(yi)
            targets = torch.tensor(targets, dtype=torch.long, device=device)

            # Use autocast context manager
            autocast_context = torch.amp.autocast('cuda', enabled=use_amp) if hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast(enabled=use_amp)
            with autocast_context:
                enc, enc_lens = model.forward_encoder(x, x_lens)
                logits = model.ctc_head(enc)                  # (B,T,V)
                logp = logits.log_softmax(-1).transpose(0,1) # (T,B,V)
                loss = ctc_loss(logp, targets, enc_lens, y_lens.to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            update_ema()
            total += loss.item()
        # Eval (EMA)
        model.eval()
        saved = {k:v.clone() for k,v in model.state_dict().items()}
        apply_ema()

        wer_list = []
        with torch.no_grad():
            for x, y_pad, x_lens, y_lens, texts, *_ in val_dl:
                x, x_lens = x.to(device), x_lens.to(device)
                enc, enc_lens = model.forward_encoder(x, x_lens)
                logits = model.ctc_head(enc)  # (B,T,V)
                hyps = greedy_ctc_decode(logits, tokenizer)

                # Compute WER for each sample
                for b in range(len(texts)):
                    ref = texts[b]
                    hyp = hyps[b]
                    wer = compute_wer(ref, hyp)
                    wer_list.append(wer)

        avg_wer = np.mean(wer_list) if wer_list else 1.0
        print(f"Epoch {epoch}: train_loss={total/len(train_dl):.4f} val_WER={avg_wer:.3f}")
        model.load_state_dict(saved)  # restore non-EMA for next epoch

    os.makedirs('runs/checkpoints', exist_ok=True)
    torch.save({'model': model.state_dict(),
                'tokenizer': tokenizer.vocab,
                'mean': mean, 'std': std}, 'runs/checkpoints/conformer_ctc_ema.pt')

if __name__ == "__main__":
    train_conformer_ctc(
        d_model=256,  # Smaller to fit memory
        num_blocks=12,  # Fewer blocks
        batch_size=192,  # Max out GPU - only using 2%!
        epochs=10,  # More epochs
        num_workers=12,  # Match system recommendation
        lr=3e-4,  # Good starting LR
    )