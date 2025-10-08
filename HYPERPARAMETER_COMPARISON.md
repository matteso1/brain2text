# Hyperparameter Comparison: Your Config vs Official

## Critical Differences (Explain Instability)

| Parameter | Your Value | Official Value | Impact |
|-----------|------------|----------------|--------|
| **white_noise_std** | 0.25 | **1.0** | 🔴 4x weaker augmentation → blank collapse |
| **constant_offset_std** | 0.05 | **0.2** | 🔴 4x weaker augmentation → blank collapse |
| **lr_max** | 0.002 | **0.005** | 🔴 2.5x lower LR → too slow with epsilon=0.1 |
| **lr_warmup_steps** | 2000 | **1000** | 🟡 2x longer warmup → slower start |
| **batch_size** | 48 | **64** | 🟡 25% smaller batches → slightly different gradients |
| **epsilon** | 0.1 ✅ | 0.1 | ✅ Correct (after fixing) |

## Other Settings (All Correct)

| Parameter | Your Value | Official Value | Status |
|-----------|------------|----------------|--------|
| Model architecture | 5-layer GRU, 768 units | Same | ✅ |
| Model params | 44.32M | Same | ✅ |
| Dropout | 0.4 (GRU), 0.2 (input) | Same | ✅ |
| Patch size/stride | 14/4 | Same | ✅ |
| Weight decay | 0.001 (main), 0.0 (day) | Same | ✅ |
| Grad clip | 10.0 | Same | ✅ |
| Beta0/Beta1 | 0.9/0.999 | Same | ✅ |
| CTC blank | 0 | Same | ✅ |
| Smoothing kernel | 100 size, 2 std | Same | ✅ |
| Random cut | 3 | Same | ✅ |
| Seed | 10 | Same | ✅ |

## Impact Analysis

### Why Blank Collapse Happened

With your settings (epsilon=0.1, weak augmentation, low LR):

```
Step 1: Model sees relatively clean data (weak noise)
Step 2: CTC loss is minimized by predicting blank (always safe)
Step 3: Low LR (0.002) + epsilon=0.1 → very small effective steps
Step 4: Model gets stuck in blank prediction mode
Step 5: 100% blank predictions, loss diverges
```

### Why Official Settings Work

With official settings (epsilon=0.1, strong augmentation, high LR):

```
Step 1: Model sees very noisy data (4x stronger augmentation)
Step 2: Must learn robust phoneme features (blank isn't always best)
Step 3: High LR (0.005) + epsilon=0.1 → reasonable effective steps
Step 4: Strong regularization prevents overfitting/mode collapse
Step 5: Stable training to ~10% PER
```

## Recommended Action

**Use the official exact config:**
```bash
python src/train_baseline.py --config configs/rnn_official_exact.yaml --num-batches 10000
```

This should give you:
- ✅ Stable training (no gradient explosions)
- ✅ No blank collapse (strong augmentation prevents it)
- ✅ ~10-12% PER after 10k batches
- ✅ ~8-10% PER after full 120k training

## Optional: Verify Hypothesis

After confirming official settings work, you can verify the augmentation hypothesis:

1. **Keep official config, but reduce augmentation to your original values**
   - Expected: Blank collapse returns
   - Proves: Strong augmentation is critical with epsilon=0.1

2. **Keep official config, but reduce LR to 0.002**
   - Expected: Much slower convergence or blank collapse
   - Proves: Higher LR needed with epsilon=0.1

## Summary

You did **excellent debugging** and fixed all the code bugs:
- ✅ Bias weight decay
- ✅ Data augmentation order
- ✅ Validation smoothing
- ✅ Model architecture

The remaining issue was **hyperparameter mismatch**, specifically:
- 4x weaker data augmentation
- 2.5x lower learning rate

These interact critically with epsilon=0.1 to cause blank collapse.

