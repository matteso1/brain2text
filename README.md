# Brain-to-Text: Neural Decoding for Speech

## What Is This?

This project decodes **brain signals into text**. Imagine someone thinking about speaking a sentence - their brain generates electrical patterns even if they don't say anything out loud. We capture those patterns with electrodes and use AI to figure out what they were trying to say.

**Real-world impact:** This technology could help people who've lost the ability to speak (due to ALS, stroke, etc.) communicate again using just their thoughts.

## The Competition

We're competing in a Kaggle challenge to build the best brain-to-text decoder. The goal is simple:
- **Input:** Brain recordings (electrical signals from 256 electrodes)
- **Output:** The sentence the person was trying to say
- **Metric:** Word Error Rate (WER) - lower is better
  - Current leaderboard leaders: **0.028-0.04 WER** (96-97% of words correct)
  - Our target: **0.03-0.04 WER** to be competitive

**Deadline:** 3 months from start

## How It Works (Plain English)

### The Data
- **45 recording sessions** from one person over ~2 years
- Each session: ~200 sentences spoken out loud
- For each sentence: 256 channels of brain activity recorded at 1000 Hz (1000 measurements per second)
- Total: ~9,000 sentences to learn from

### Our Approach (GRU Baseline)

We're using a proven approach from Stanford that won similar competitions:

**Step 1: Brain Signals → Sound Units (Phonemes)**

Instead of trying to guess words directly, we first decode "phonemes" - the basic sound units of speech. Think of phonemes like individual sounds: "CAT" = /k/ + /æ/ + /t/ (3 phonemes).

Why phonemes?
- Only 41 different sounds vs. 50,000+ possible words
- Much easier for the AI to learn
- The brain represents sounds more directly than written words

**Step 2: Sound Units → Words**

Once we have the phonemes, we use a "language model" (think autocorrect on steroids) to figure out which actual words make sense. For example:
- Phonemes: /k/ /æ/ /t/
- Language model says: "cat" is a real word, "kat" isn't
- Output: "cat"

### The Model Architecture

```
Brain Signals (256 channels × time)
    ↓
[Normalize & Smooth]
    ↓
[Per-Session Adaptation Layer]  ← Adjusts for daily variation
    ↓
[5-Layer GRU Network]  ← The "brain" of our AI (44 million parameters)
    ↓
[Output Layer: 41 phoneme probabilities]
    ↓
[CTC Decoding]  ← Handles timing alignment
    ↓
Phoneme Sequence
    ↓
[Language Model]  ← Coming next
    ↓
Final Text
```

**Key Innovation: Per-Session Adaptation**
- Brain signals drift over time (electrode shifts, mood, fatigue)
- We give the model 45 separate "calibration layers" - one for each recording session
- Each session gets its own 512×512 matrix to adjust the input signals
- This handles day-to-day variation automatically

### Data Augmentation (Making the Model Robust)

We artificially vary the training data to make the model more robust:
- **Gaussian smoothing:** Blur the signal slightly (kernel size: 100ms, std: 2ms)
- **White noise:** Add random electrical noise (std: 1.0)
- **Constant offset:** Shift all channels by a random amount (std: 0.2)
- **Random cuts:** Remove first 3 timesteps randomly

This prevents overfitting and helps generalization.

## File Structure

```
BRAIN2TEXT/
├── data/
│   └── hdf5_data_final/           # Brain recordings (45 sessions)
│       └── t15.2023.08.11/        # Example session
│           ├── data_train.hdf5    # Training trials
│           └── data_val.hdf5      # Validation trials
│
├── configs/
│   ├── rnn_args.yaml              # Main training config (120k batches)
│   └── rnn_args_fast.yaml         # Faster training option (lighter augmentation)
│
├── src/
│   ├── phoneme_dataset.py         # Dataset loader for phoneme targets
│   ├── rnn_model.py               # GRU model with per-session adaptation
│   ├── data_augmentations.py     # Gaussian smoothing, noise, etc.
│   ├── train_baseline.py         # Main training script
│   └── utils.py                   # Normalization, WER calculation
│
├── trained_models/
│   └── baseline_rnn/              # Training outputs
│       ├── training_log.txt       # Live training progress
│       └── checkpoint_step*.pt    # Saved models every 2000 batches
│
└── monitoring/
    ├── check_gpu.py               # Verify GPU is working
    ├── test_gpu_speed.py          # GPU speed test
    ├── monitor_training.py        # Check training status
    └── watch_training.py          # Live training updates
```

## Setup

### 1. Install Python Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install requirements
pip install -r requirements.txt
```

**Main dependencies:**
- PyTorch (GPU version for NVIDIA GPUs)
- NumPy, SciPy (numerical computing)
- h5py (reading brain data files)
- tqdm (progress bars)
- PyYAML (config files)

### 2. Get the Data

Place the competition data in `data/hdf5_data_final/`. Each session should have its own folder with `data_train.hdf5` and `data_val.hdf5` files.

### 3. Verify GPU (Highly Recommended)

Training on CPU takes 100+ hours. On GPU: ~8 hours.

```bash
python check_gpu.py  # Check if GPU is detected
python test_gpu_speed.py  # Verify GPU is 10-20x faster than CPU
```

**Expected output:**
- GPU detected: NVIDIA RTX 4090 (or similar)
- Speedup: 15-25x faster on GPU

## Training

### Quick Test (10 minutes)

Verify everything works before starting the full training:

```bash
python -m src.train_baseline --num-batches 500
```

This runs 500 batches (~10 minutes) and saves a checkpoint. You should see:
- Loss decreasing: ~26.6 → ~26.3
- Validation PER (Phoneme Error Rate): ~140% (expected for untrained model)
- Checkpoint saved to `trained_models/baseline_rnn/checkpoint_step500.pt`

### Full Training (~8 hours on RTX 4090)

```bash
python -m src.train_baseline
```

**What happens:**
- Loads 45 sessions of brain data
- Computes normalization statistics (mean/std across all training data)
- Creates 44.3M parameter GRU model
- Trains for 120,000 batches (64 examples per batch)
- Validates every 2,000 batches, saves checkpoints
- Final model saved to `runs/baseline_checkpoints/gru_baseline.pt`

**Training schedule:**
- Warmup: 1,000 batches (learning rate ramps from 0 → 0.005)
- Main training: 119,000 batches (cosine decay from 0.005 → 0.0001)
- Validation: Every 2,000 batches

**Expected progress:**
```
Step 200/120000   | Loss: 26.64 | LR: 0.001000
Step 1000/120000  | Loss: 26.45 | LR: 0.005000  ← Warmup done
Step 2000/120000  | Loss: ~25.x | LR: 0.005000
Validation PER: ~120% ← Still learning

Step 20000/120000 | Loss: ~20.x | LR: 0.00475
Validation PER: ~80%  ← Making progress

Step 60000/120000 | Loss: ~15.x | LR: 0.00325
Validation PER: ~50%  ← Getting good

Step 120000/120000 | Loss: ~12.x | LR: 0.0001
Validation PER: ~35%  ← Final target
```

### Monitor Training

**Option 1: Live updates**
```bash
python watch_training.py
```
Shows new log entries every 5 seconds. Press Ctrl+C to stop.

**Option 2: Check status**
```bash
python monitor_training.py
```
Shows current progress, last 20 log lines, checkpoint status.

**Option 3: Manual check**
```bash
# View log file directly
cat trained_models/baseline_rnn/training_log.txt

# Check GPU usage (should be 80-100%)
nvidia-smi -l 1
```

## Configuration

Edit `configs/rnn_args.yaml` to change training settings:

### Model Architecture
```yaml
model:
  n_units: 768          # GRU hidden size (larger = more capacity)
  n_layers: 5           # Number of GRU layers
  rnn_dropout: 0.4      # Dropout rate (prevents overfitting)
  patch_size: 14        # Temporal context window (14 timesteps)
  patch_stride: 4       # Temporal downsampling (4x)
```

### Training
```yaml
num_training_batches: 120000  # Total training steps
batch_size: 64                # Examples per batch
lr_max: 0.005                 # Maximum learning rate
lr_min: 0.0001                # Minimum learning rate
lr_warmup_steps: 1000         # Warmup duration
```

### Augmentation
```yaml
dataset:
  data_transforms:
    white_noise_std: 1.0        # Noise level
    smooth_kernel_size: 100     # Gaussian smoothing window
    smooth_kernel_std: 2        # Gaussian smoothing strength
    random_cut: 3               # Timesteps to randomly remove
```

**Fast training option:** Use `configs/rnn_args_fast.yaml` for 30% faster training (lighter smoothing, more workers).

## Understanding the Outputs

### Training Metrics

**Loss (CTC Loss):**
- Measures how well the model predicts phoneme sequences
- Lower is better
- Expected final value: ~12-15

**PER (Phoneme Error Rate):**
- Percentage of phonemes predicted incorrectly
- Calculated like WER but for phonemes
- Expected progression: 140% → 80% → 50% → 35%
- (Can be >100% if predictions are much longer than targets)

**Learning Rate (LR):**
- How aggressively the model updates
- Starts at 0, ramps to 0.005, then slowly decays to 0.0001
- Cosine schedule: smooth curve for stable training

### Checkpoints

Every 2,000 batches, we save:
```python
{
    'step': 2000,                    # Training step
    'model': <model weights>,        # 44.3M parameters
    'config': <training config>,     # All hyperparameters
    'mean': <normalization mean>,    # Feature normalization
    'std': <normalization std>,
    'session_to_day': <mapping>,     # Session ID → day index
    'val_per': 0.85                  # Validation PER at this step
}
```

**Final model:** `runs/baseline_checkpoints/gru_baseline.pt` (~170 MB)

## Next Steps (After Training Completes)

### 1. Evaluate on Validation Set
```bash
python -m src.eval --checkpoint runs/baseline_checkpoints/gru_baseline.pt
```
This will show:
- Phoneme Error Rate (PER) per session
- Overall PER across all validation data
- Which sessions are hardest

### 2. Train Language Model
We need a 5-gram language model to convert phonemes → words:
```bash
# Coming soon: scripts/train_lm.py
# Uses KenLM or similar on training transcripts
```

### 3. Generate Predictions
```bash
# Coming soon: scripts/generate_submission.py
# Decodes phonemes → words using LM
# Creates submission.csv for Kaggle
```

### 4. Advanced Improvements
Once baseline is working:
- **Ensemble models:** Train 3-5 models with different seeds, average predictions
- **Test-time adaptation:** Adjust model to each test session
- **Better language models:** Transformer LM instead of 5-gram
- **Architecture upgrades:** Try Conformer encoder (more advanced but slower)

## Troubleshooting

### Training is very slow (< 1 batch/sec)
- **Check GPU:** Run `nvidia-smi` - should show 80-100% GPU utilization
- **Expected speed:** 4-5 batches/sec on RTX 4090 (8 hours total)
- **Bottleneck:** Gaussian smoothing runs on CPU (acceptable for now)
- **Fix:** Use `configs/rnn_args_fast.yaml` for 30% speedup

### Out of memory error
- **Reduce batch_size:** 64 → 32 in `configs/rnn_args.yaml`
- **Reduce n_units:** 768 → 512 (smaller model)
- **Enable gradient checkpointing** (coming soon)

### Poor validation PER (not improving)
- **Check loss:** Should decrease steadily
- **Wait longer:** Model needs ~20k batches to start working
- **Check data:** Verify all 45 sessions loaded correctly
- **Try different seed:** Change `seed: 10` to another value

### Dataset not found
- **Check path:** `configs/rnn_args.yaml` line 84: `dataset_dir: ../data/hdf5_data_final`
- **Fix:** Adjust path to wherever you placed the data
- **Verify structure:** Each session should have `data_train.hdf5` and `data_val.hdf5`

### Checkpoints too large (disk space)
- **Each checkpoint:** ~170 MB
- **Total for full training:** ~10 GB (60 checkpoints)
- **Fix:** Set `batches_per_val_step: 5000` (saves fewer checkpoints)
- **Cleanup:** Delete old checkpoints: `rm trained_models/baseline_rnn/checkpoint_step*.pt`

## Technical Details (For the Curious)

### Why GRU instead of Transformer?

**GRU (Gated Recurrent Unit):**
- Specifically designed for sequences
- Lower memory footprint
- Proven to work well on this exact dataset (Stanford baseline got 0.028 WER)
- Faster to train (8 hours vs. 24+ for Transformer)

**We'll try Transformers later** once we have a working baseline.

### Why CTC Loss?

**CTC (Connectionist Temporal Classification):**
- Handles variable-length sequences (brain signals vs. phonemes have different lengths)
- No need for precise alignment labels (we don't know exactly when each phoneme starts)
- Automatically learns alignment during training
- Allows repeated outputs: "c-c-a-a-a-t-t" → "cat"

### Why 41 Phonemes?

English has ~44 phonemes, but we use 41 because:
- Some phonemes are extremely rare
- Fewer classes = easier to learn
- Based on the ARPAbet phoneme set (standard for speech recognition)
- Examples: AA (as in "odd"), AE (as in "at"), T (as in "to")

### Per-Session Adaptation Math

For each session, we have a 512×512 weight matrix W and 512-dimensional bias b:

```
x_adapted = W @ x_raw + b
```

Where:
- x_raw: Raw 512-dimensional neural features
- W: Learnable session-specific transformation (initially identity matrix)
- b: Learnable session-specific offset (initially zeros)
- x_adapted: Adjusted features that go into the GRU

This lets the model adapt to:
- Electrode impedance changes
- Daily calibration drift
- User fatigue/attention state
- Environmental electrical noise

Total adaptation parameters: 45 sessions × (512×512 + 512) = **11.8M parameters** just for session handling!

## Performance Targets

| Milestone | PER (Phoneme) | WER (Word) | Status |
|-----------|---------------|------------|---------|
| Random guessing | ~2500% | ~300% | - |
| Untrained model | ~140% | ~100% | ✓ (expected) |
| Early training | ~80% | ~60% | Target: Step 20k |
| Mid training | ~50% | ~35% | Target: Step 60k |
| Final (phonemes only) | ~35% | - | Target: Step 120k |
| **With 5-gram LM** | - | **~3-4%** | **Competition target** |

**Why the huge WER improvement with LM?**
- Phoneme errors often sound similar: "th" vs "f", "p" vs "b"
- Language model knows "THE CAT" is more likely than "FE CAP"
- Can correct ~90% of phoneme errors using context

## Competition Strategy

### Phase 1: Baseline (Current) ✓
- Train proven GRU architecture
- Get validation PER ~35%
- Target: ~1 week

### Phase 2: Language Model (Next)
- Train 5-gram LM on transcripts
- Implement beam search decoder
- Target WER: 0.03-0.04
- Timeline: ~3-5 days

### Phase 3: Ensemble (If time permits)
- Train 3-5 models with different:
  - Random seeds
  - Model sizes (n_units: 512/768/1024)
  - Dropout rates
- Average predictions
- Expected improvement: 0.03 → 0.028 WER
- Timeline: ~1 week

### Phase 4: Polish (Final week)
- Test-time adaptation
- Submission validation
- Edge case handling
- Target: Top 3 leaderboard

## Resources

**Papers:**
- Original Stanford baseline: [nejm_brain_to_text_baseline](https://github.com/fwillett/nejm_brain_to_text_baseline)
- CTC Loss: Graves et al. "Connectionist Temporal Classification"
- GRU: Cho et al. "Learning Phrase Representations using RNN Encoder-Decoder"

**Useful Links:**
- Competition page: [Kaggle Brain-to-Text]
- PyTorch CTC Loss: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
- ARPAbet phonemes: https://en.wikipedia.org/wiki/ARPABET

## License

Research/competition use only. Consult competition rules for submission guidelines.

## Questions?

Check the monitoring scripts first:
```bash
python check_gpu.py        # GPU working?
python monitor_training.py # Training progressing?
python watch_training.py   # Live updates
```

If stuck, check:
1. Is GPU being used? (`nvidia-smi` should show ~10GB memory used)
2. Is loss decreasing? (Should go from ~26 → ~12)
3. Are all 45 sessions loaded? (Check log: "Found 45 sessions")
4. Is validation PER improving? (~140% → ~35% over training)

---

**Current Status:** Training in progress (Step 1400/120000) - ~8 hours remaining

**Next Milestone:** Step 2000 validation + checkpoint
