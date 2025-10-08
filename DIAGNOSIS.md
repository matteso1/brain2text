# Training Stability Diagnosis

## Problem Summary
GRU baseline training fails with either:
1. **Gradient explosions** (epsilon=1e-8, lr=0.002) → NaN gradients, divergence
2. **Blank collapse** (epsilon=0.1, weak augmentation) → 100% blank predictions, mode collapse

Official NEJM implementation trains stably with epsilon=0.1 to ~10% PER.

## Root Cause Identified

The issue is **NOT** just epsilon, but the **combination** of:
- Epsilon value (optimizer stability)
- Learning rate (step size)
- Data augmentation strength (regularization)

## Critical Differences Found

### 1. Data Augmentation Strength (CRITICAL!)

**Your config** (rnn_full_4090.yaml):
```yaml
white_noise_std: 0.25
constant_offset_std: 0.05
```

**Official config**:
```yaml
white_noise_std: 1.0      # 4x STRONGER
constant_offset_std: 0.2  # 4x STRONGER
```

### 2. Learning Rate

**Your config**:
```yaml
lr_max: 0.002
lr_warmup_steps: 2000
```

**Official config**:
```yaml
lr_max: 0.005             # 2.5x HIGHER
lr_warmup_steps: 1000     # 2x FASTER warmup
```

### 3. Other Differences

**Your config**:
```yaml
batch_size: 48            # Conservative for 4090
use_fused_adamw: false    # Compatibility mode
```

**Official config**:
```yaml
batch_size: 64            # Full batch
fused: true               # Faster optimizer
```

## Why This Matters

### epsilon=0.1 Requires Strong Augmentation

The epsilon parameter in Adam/AdamW stabilizes updates:
```
update = lr * grad / (sqrt(variance) + epsilon)
```

With epsilon=0.1:
- **Denominator is larger** → smaller effective step size
- **Need higher learning rate** to compensate (0.005 vs 0.002)
- **Need strong regularization** to prevent overfitting/mode collapse

Without strong augmentation:
- Model sees cleaner, more consistent data
- Learns to predict blank (safest CTC prediction)
- Gets stuck in mode collapse

With strong augmentation:
- Model sees noisier, more variable data
- Must learn robust phoneme representations
- Regularization prevents blank collapse

### Why Your Attempts Failed

**Attempt 1** (epsilon=1e-8, lr=0.002):
- Small epsilon → unstable denominator
- Even small gradient spikes cause explosions
- Result: Gradient explosions at steps 2550, 3100, 3500...

**Attempt 2 & 3** (epsilon=0.1, weak augmentation):
- Large epsilon → stable but small effective learning rate
- Weak augmentation → insufficient regularization
- Result: Blank collapse (100% blank predictions)

## Solution: Match Official Settings Exactly

Created `configs/rnn_official_exact.yaml` with:
```yaml
epsilon: 0.1              # Official value
lr_max: 0.005             # Official (2.5x higher than yours)
lr_warmup_steps: 1000     # Official (2x faster than yours)
white_noise_std: 1.0      # Official (4x stronger than yours)
constant_offset_std: 0.2  # Official (4x stronger than yours)
batch_size: 64            # Official (vs your 48)
use_fused_adamw: true     # Official optimizer
```

## Testing Plan

### Test 1: Official Exact Settings
```bash
python src/train_baseline.py --config configs/rnn_official_exact.yaml --num-batches 10000
```

**Expected outcome**: Stable training, ~10% PER at 10k batches

**What to watch for**:
- Loss should decrease smoothly from ~2.5 → ~1.5
- Gradient norms should stay below 20-30 (not spike to 100+)
- Unique phonemes should stay above 10 (not collapse to 1-2)
- No NaN/Inf gradients

### Test 2: Verify Augmentation Impact (Optional)
If Test 1 succeeds, try with weaker augmentation to confirm hypothesis:
```yaml
white_noise_std: 0.25     # Weaker (your original)
constant_offset_std: 0.05 # Weaker (your original)
# Keep other settings the same
```

**Expected outcome**: Should show signs of blank collapse (proves augmentation is critical)

### Test 3: Verify Learning Rate Impact (Optional)
If Test 1 succeeds, try with lower learning rate:
```yaml
lr_max: 0.002             # Lower (your original)
# Keep other settings the same
```

**Expected outcome**: Slower convergence or blank collapse (proves higher LR is needed with epsilon=0.1)

## Implementation Notes

All fixes are already in place:
- ✅ Bias weight decay fixed (src/train_baseline.py:198-221)
- ✅ Data augmentation order fixed (src/phoneme_dataset.py:111-142)
- ✅ Validation smoothing fixed (src/phoneme_dataset.py:85-95)
- ✅ CTC loss uses zero_infinity=False (src/train_baseline.py:196)
- ✅ Model architecture matches official (src/rnn_model.py)

Only needed change: **Use the correct hyperparameters** (configs/rnn_official_exact.yaml)

## Prediction

With official exact settings, training should be **completely stable** and reach:
- ~1.5-1.7 loss after 2000 batches
- ~10-12% PER after 10,000 batches
- ~8-10% PER after 120,000 batches (full training)

The combination of:
- epsilon=0.1 (stable optimizer)
- lr=0.005 (higher LR to compensate for epsilon)
- Strong augmentation (prevents overfitting/mode collapse)

...is the "magic formula" that makes training stable.

