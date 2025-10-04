# Brain to Text - Quick Start Guide

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify data structure:**
Your data should be organized as:
```
data/
  hdf5_data_final/
    t15.2023.08.11/
      data_train.hdf5
    t15.2023.08.13/
      data_train.hdf5
      data_val.hdf5
      data_test.hdf5
    ...
```

## Training

### Quick Test (Small Model)
Train a small model to verify everything works:
```bash
python -m src.train
```

This will use default parameters:
- d_model=256
- num_blocks=12
- vocab_size=2000
- batch_size=8
- epochs=5

### Custom Training
Modify `src/train.py` and call `train_conformer_ctc()` with custom parameters:
```python
from src.train import train_conformer_ctc

train_conformer_ctc(
    data_root='data',
    d_model=384,        # Larger model
    num_blocks=16,      # More layers
    nhead=6,
    vocab_size=2000,
    batch_size=4,       # Smaller batch if OOM
    epochs=20,
    lr=1e-3,
    device='cuda'
)
```

## Evaluation

### Session-level Cross-Validation
Evaluate model performance per session:
```bash
python -m src.eval
```

This will print WER for each session in the validation set.

## Generate Submission

After training, generate predictions for test set:
```bash
python -m src.inference --ckpt runs/checkpoints/conformer_ctc_ema.pt --output submission.csv
```

Arguments:
- `--ckpt`: Path to checkpoint (default: runs/checkpoints/conformer_ctc_ema.pt)
- `--data-root`: Data directory (default: data)
- `--output`: Output CSV path (default: submission.csv)
- `--batch-size`: Inference batch size (default: 16)
- `--device`: cuda or cpu (default: auto-detect)

## Current Status

**Phase 1: Basic CTC Baseline** ✅
- [x] Fixed tokenizer bugs
- [x] Added WER metric
- [x] Training with CTC loss
- [x] Inference pipeline
- [ ] Verify training completes successfully

**Phase 2: RNN-T Training** (Next)
- [ ] Add RNN-T loss
- [ ] Two-stage training (CTC → RNN-T)
- [ ] Beam search decoder

**Phase 3: Advanced Features** (Future)
- [ ] Language model
- [ ] SpecAugment
- [ ] Session embeddings
- [ ] Test-time adaptation
- [ ] Ensembling

## Common Issues

### Out of Memory
- Reduce `batch_size` in training
- Reduce `d_model` or `num_blocks`
- Use gradient accumulation

### No Validation Data
Some sessions only have training data. The code will warn but continue.

### Slow Training
- Ensure you're using CUDA: `device='cuda'`
- Increase `num_workers` in DataLoader
- Use mixed precision (already enabled via AMP)

## Model Architecture

Current implementation:
- **Encoder**: Conformer (ConvSubsample + 12 Conformer blocks)
- **Decoder**: CTC head (auxiliary, not full RNN-T yet)
- **Tokenization**: BPE-style character merges
- **Normalization**: Global mean/std from training set
