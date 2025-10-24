# Brain-to-Text: Neural Navigators Kaggle Competition

**Project Status (October 2025): ✅ WORKING BASELINE WITH WER METRIC**

This repository contains the code for the Neural Navigators team's entry into the **Brain-to-Text '25 Kaggle Competition**. Our goal is to build a state-of-the-art neural decoder that translates brain signals into text, with the ultimate aim of restoring communication to individuals with paralysis.

---

## 🚀 Breakthrough Achieved: Complete Pipeline Working!

After intensive debugging, we have successfully implemented the complete NEJM baseline pipeline from brain signals to word predictions. Our infrastructure now includes:

**✅ Complete End-to-End Pipeline:**
- **Brain Signals → Phonemes**: 5-layer GRU model with CTC loss
- **Phonemes → Words**: FST-based language model with Redis communication
- **Full Evaluation**: Word Error Rate (WER) calculation on validation set

**📊 Latest Results (Validation Set - 1,426 samples):**
- **Word Error Rate (WER): 40.32%** 🎯 **(This is the actual competition metric!)**
- **Phoneme Error Rate (PER): ~19%** (estimated from training)
- **Language Model**: 1-gram FST model (room for improvement with 3-gram)

**🔧 Infrastructure Status:**
- ✅ Stable model training (`src/train_repro.py`)
- ✅ Working language model server with Redis
- ✅ Complete evaluation pipeline
- ✅ Cross-platform compatibility (Windows + WSL)

This gives us a **solid, reproducible baseline** to build upon!

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

## Lessons Learned from the Complete Pipeline Implementation

Building the complete brain-to-text pipeline taught us critical lessons about neural decoding systems:

**🔧 Infrastructure Challenges Solved:**
1.  **Cross-Platform Compatibility**: Windows symlinks don't work in WSL. We recreated all symlinks (`kaldi`, `utils`, `decoder`, etc.) pointing to `../../core/` directories.
2.  **Data Type Consistency**: PyTorch models trained with BFloat16 require consistent data types throughout the pipeline. Fixed gauss smoothing and model input conversion.
3.  **Redis Communication**: The language model server and evaluation script communicate via Redis streams. Proper setup requires daemonized Redis and background LM server.
4.  **Python Environment Isolation**: The LM compilation requires Python 3.9 with specific PyTorch 1.13.1, while evaluation works with PyTorch 2.x.

**🧠 Model Integration Insights:**
1.  **Complete Pipeline is Essential**: WER (40.32%) is very different from PER (~19%). The language model dramatically affects final performance.
2.  **1-gram vs 3-gram Impact**: Our 1-gram baseline gives reasonable results (40% WER), but 3-gram models should improve this significantly.
3.  **End-to-End Validation**: We now have a complete, reproducible pipeline that any team member can run to get baseline results.

**Key Breakthroughs:**
1.  **Symlink Management**: Fixed all broken Windows symlinks that prevented compilation
2.  **Data Type Matching**: Ensured consistent BFloat16/Float32 handling throughout the pipeline
3.  **Environment Setup**: Created working conda environments for both training and LM serving
4.  **Cross-Process Communication**: Successfully integrated Redis-based LM server with evaluation client

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

### 1. Environment Setup
```powershell
# Make sure you have Python 3.12 installed
# Create and activate the virtual environment
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies (including PyTorch for CUDA 12.1)
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Language Model Setup (WSL Required)
The language model requires compilation and Redis. Follow these steps:

```bash
# In WSL (Ubuntu):
# 1. Install conda and dependencies
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda

# 2. Setup language model environment
export PATH=~/miniconda/bin:$PATH
cd /mnt/c/Users/nilsm/PycharmProjects/BRAIN2TEXT/nejm_repo/nejm-brain-to-text
./setup_lm.sh

# 3. Start Redis server
redis-server --daemonize yes

# 4. Start language model server (in background)
conda activate b2txt25_lm
python language_model/language-model-standalone.py --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55 --redis_ip localhost --gpu_number 0
```

### 3. Run Evaluation (Windows)
```powershell
# Install evaluation dependencies
pip install torch torchvision torchaudio h5py redis omegaconf tqdm editdistance

# Run evaluation to get WER
python nejm_repo/nejm-brain-to-text/model_training/evaluate_model.py --model_path data/t15_pretrained_rnn_baseline --data_dir data/hdf5_data_final --csv_path data/t15_copyTaskData_description.csv --eval_type val --gpu_number 0
```

### 4. Run Training
This is the single source of truth for training. All other scripts are deprecated.

```powershell
# To run a short, 2000-batch test (~10 minutes)
python src/train_repro.py --config configs/rnn_official_exact.yaml --num-batches 2000

# To run a full 120,000-batch training run (~8-10 hours)
python src/train_repro.py --config configs/rnn_official_exact.yaml
```

### 5. Monitor Training
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

## 📈 Current Status Summary

**✅ ACHIEVED:**
- Complete end-to-end pipeline (brain signals → phonemes → words → WER)
- **Baseline WER: 40.32%** on validation set (1,426 samples)
- Stable model training with 19% PER
- Working cross-platform setup (Windows + WSL)
- Reproducible evaluation process

**🚀 IMMEDIATE NEXT STEPS:**
1. **Test 3-gram language model** (should significantly improve WER)
2. **Implement Layer Normalization** (Phase 2 - proven to help)
3. **Model ensembling + LLM rescoring** (Phase 3 - top teams strategy)

**🏁 REPRODUCTION:**
Any team member can now reproduce these results by following the setup instructions above. The pipeline is stable and documented.

## 📋 Quick Start for Team Members

```bash
# 1. Setup Windows environment
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision torchaudio h5py redis omegaconf tqdm editdistance

# 2. Setup WSL environment (one-time)
wsl bash -c "
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda
export PATH=~/miniconda/bin:\$PATH
cd /mnt/c/Users/nilsm/PycharmProjects/BRAIN2TEXT/nejm_repo/nejm-brain-to-text
./setup_lm.sh
"

# 3. Run evaluation (anytime)
# Start Redis: wsl redis-server --daemonize yes
# Start LM server: wsl conda activate b2txt25_lm && wsl python language_model/language-model-standalone.py --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55 --redis_ip localhost --gpu_number 0 &
# Run evaluation: python nejm_repo/nejm-brain-to-text/model_training/evaluate_model.py --model_path data/t15_pretrained_rnn_baseline --data_dir data/hdf5_data_final --csv_path data/t15_copyTaskData_description.csv --eval_type val --gpu_number 0
```

**Expected Results:** WER of ~40% on validation set

This is our moment. Let's get this done.
