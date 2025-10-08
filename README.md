# Brain-to-Text: Neural Navigators Kaggle Competition

**Project Status (October 2025): ✅ STABLE BASELINE ACHIEVED**

This repository contains the code for the Neural Navigators team's entry into the **Brain-to-Text '25 Kaggle Competition**. Our goal is to build a state-of-the-art neural decoder that translates brain signals into text, with the ultimate aim of restoring communication to individuals with paralysis.

---

## 🚀 Breakthrough Achieved: We Have a Stable Model!

After a challenging debugging phase, we have successfully implemented and stabilized the official NEJM baseline model. Our new training script, `src/train_repro.py`, is a clean-room reproduction of the official logic and has proven to be **rock solid**.

**Latest Training Run (20k batches):**
- **PER @ 2k batches**: `54.31%`
- **PER @ 4k batches**: `36.91%`
- **PER @ 6k batches**: `30.84%`

This demonstrates a fast, stable learning curve with **zero gradient explosions or model collapse**. We now have the strong foundation we need to compete.

---

## 🧠 How It Works: The GRU Baseline

Our current approach is a faithful implementation of the successful Stanford baseline.

**Step 1: Brain Signals → Phonemes (Sound Units)**
-   An AI model (a 5-layer GRU) reads the raw 512-channel brain data.
-   Instead of guessing words, it first decodes the data into **phonemes**—the basic sounds of speech (e.g., "cat" -> /k/ /æ/ /t/). There are only 41 phonemes, making this a much more solvable problem than guessing from 50,000+ words.
-   We use **Connectionist Temporal Classification (CTC) Loss**, a specialized algorithm that handles the alignment between the long, messy brain signal and the short, clean phoneme sequence.

**Step 2: Phonemes → Words**
-   Once we have a sequence of phonemes, a traditional **language model** (like a super-powered autocorrect) is used to find the most probable sequence of words.
-   This step is crucial and corrects many errors from the first stage (e.g., it knows "THE CAT" is more likely than "FEE CAP").

---

## Lessons Learned from the Debugging Trenches

Achieving a stable baseline was a major challenge. The initial model suffered from two critical failures:
1.  **Gradient Explosions**: With a standard optimizer configuration (`epsilon=1e-8`), the model would diverge with NaN gradients.
2.  **Blank Collapse**: Using the official configuration (`epsilon=0.1`) but with our own subtle bugs, the model would only predict "blank" tokens, resulting in 100% error rate.

The solution was a **principled reset**. We learned that a series of small, interacting bugs in our original training script were the culprits.

**Key Breakthroughs:**
1.  **A Clean Implementation is King**: Instead of patching the old script (`train_baseline.py`), we wrote `src/train_repro.py` from scratch, faithfully following the official implementation's logic. This eliminated all hidden bugs.
2.  **GPU-Side Augmentations are Critical**: The official code applies data augmentations (noise, etc.) on the GPU right before the forward pass. Our old script did this on the CPU in the data loader. Moving this to the GPU was essential for matching the exact training conditions.
3.  **Hyperparameter "Ecosystems"**: We confirmed that `epsilon=0.1` is not a standalone fix. It *requires* the support of **4x stronger data augmentation** and a **2.5x higher learning rate** to function. One cannot work without the others.

---

## 🏆 Competition Game Plan: Path to the Top 3

Now that we have a stable baseline, we can execute our plan to climb the leaderboard. This plan is informed by the [Brain-to-Text Benchmark '24 paper](https://arxiv.org/html/2412.17227v1), which summarizes the winning strategies.

### Phase 1: Replicate Baseline Performance [Current]
-   **Goal**: Match the official baseline's performance (~10% Phoneme Error Rate).
-   **Action**: Complete a full 120,000 batch training run with `src/train_repro.py`.
-   **Estimated Time**: ~1-2 days of training.

### Phase 2: Implement 4th Place Improvements (Architectural)
-   **Goal**: Improve upon the baseline by enhancing the model itself.
-   **Action**: The 4th place team saw significant gains by adding **Layer Normalization** inside the RNN. We will modify our `GRUDecoder` to include this. This is a direct, proven path to a better architecture.
-   **Estimated Time**: ~2-3 days (1 day implementation, 1-2 days training).

### Phase 3: Implement 1st-3rd Place Strategy (Advanced Decoding)
-   **Goal**: Drastically reduce Word Error Rate using advanced decoding, as done by all top 3 teams.
-   **Action**:
    1.  **Model Ensembling**: Train 3-5 of our best models from Phase 2 with different random seeds.
    2.  **LLM Rescoring**: Average the predictions (logits) from these models and then use a fine-tuned Large Language Model (LLM) to "rescore" the final output, picking the most contextually aware and grammatically correct sentence.
-   **Estimated Time**: ~1 week (training multiple models + implementing the rescoring pipeline).

---

## 💻 How to Use This Repository

### 1. Setup Environment
```powershell
# Make sure you have Python 3.12 installed
# Create and activate the virtual environment
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies (including PyTorch for CUDA 12.1)
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Run Training
This is the single source of truth for training. All other scripts are deprecated.

```powershell
# To run a short, 2000-batch test (~10 minutes)
python src/train_repro.py --config configs/rnn_official_exact.yaml --num-batches 2000

# To run a full 120,000-batch training run (~8-10 hours)
python src/train_repro.py --config configs/rnn_official_exact.yaml
```

### 3. Monitor Training
-   The script will print loss, gradient norm, and learning rate to the console.
-   Validation PER will be calculated every 2,000 steps.

---

## File Structure Overview

-   `src/train_repro.py`: **The official, working training script.** The only one you need to run.
-   `configs/rnn_official_exact.yaml`: **The official, working hyperparameter configuration.**
-   `src/rnn_model.py`: The GRU model architecture.
-   `src/phoneme_dataset.py`: The data loader for HDF5 files.
-   `nejm_repo/`: A clone of the original NEJM implementation for reference.
-   `data/`: (Not in git) This is where the HDF5 data files should be located.

This is our moment. Let's get this done.
