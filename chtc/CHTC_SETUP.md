# Running Brain-to-Text Training on CHTC

## Prerequisites

✅ CHTC account created (nomatteson@wisc.edu)
✅ Access to `ap2001.chtc.wisc.edu` via SSH
✅ On UW campus network or VPN

**IMPORTANT**: This data is 24GB, so we MUST use `/staging/nomatteson/` (not `/home`)

---

## Quick Start (TL;DR)

```bash
# 1. On your laptop - package code and data
cd ~/PycharmProjects/BRAIN2TEXT
tar -czf brain2text_code.tar.gz src/ configs/ --exclude="*.pyc" --exclude="__pycache__"
tar -czf brain2text_data.tar.gz data/hdf5_data_final/

# 2. Transfer to CHTC
scp brain2text_code.tar.gz chtc/train.sh chtc/train.sub nomatteson@ap2001.chtc.wisc.edu:~/
scp brain2text_data.tar.gz nomatteson@transfer.chtc.wisc.edu:/staging/nomatteson/

# 3. SSH into CHTC and submit
ssh nomatteson@ap2001.chtc.wisc.edu
chmod +x train.sh
condor_submit train.sub

# 4. Monitor
condor_q          # Check queue
tail -f train_*.out   # Watch live output
```

---

## Detailed Setup Guide

### Step 1: Package Your Code (Laptop)

**Navigate to project directory:**
```bash
cd ~/PycharmProjects/BRAIN2TEXT
```

**Create code tarball** (small, goes in `/home`):
```bash
tar -czf brain2text_code.tar.gz \
    src/ \
    configs/ \
    --exclude="*.pyc" \
    --exclude="__pycache__" \
    --exclude="trained_models"
```

**Verify code size** (should be < 1 MB):
```bash
ls -lh brain2text_code.tar.gz
```

### Step 2: Package Your Data (Laptop)

**Create data tarball** (24GB, goes in `/staging`):
```bash
tar -czf brain2text_data.tar.gz data/hdf5_data_final/
```

**IMPORTANT**: This will take 5-10 minutes to compress. Go grab coffee ☕

**Verify data size** (should be ~15-24 GB):
```bash
ls -lh brain2text_data.tar.gz
```

### Step 3: Transfer Files to CHTC

**A. Transfer code to /home** (small files OK in home):
```bash
scp brain2text_code.tar.gz nomatteson@ap2001.chtc.wisc.edu:~/
scp chtc/train.sh nomatteson@ap2001.chtc.wisc.edu:~/
scp chtc/train.sub nomatteson@ap2001.chtc.wisc.edu:~/
```

**B. Transfer data to /staging** (REQUIRED for files > 1GB):

⚠️ **CRITICAL**: Data > 1GB MUST use `transfer.chtc.wisc.edu`, NOT `ap2001.chtc.wisc.edu`

```bash
scp brain2text_data.tar.gz nomatteson@transfer.chtc.wisc.edu:/staging/nomatteson/
```

This will take ~10-30 minutes depending on your internet speed.

**Verify data uploaded correctly:**
```bash
# SSH into CHTC
ssh nomatteson@ap2001.chtc.wisc.edu

# Check staging directory
ls -lh /staging/nomatteson/
# Should show: brain2text_data.tar.gz (~15-24 GB)
```

### Step 4: Submit the Job

**SSH into CHTC access point:**
```bash
ssh nomatteson@ap2001.chtc.wisc.edu
```

**Verify files are in place:**
```bash
ls -lh ~/
# Should see: brain2text_code.tar.gz, train.sh, train.sub

ls -lh /staging/nomatteson/
# Should see: brain2text_data.tar.gz (~15-24 GB)
```

**Make executable script runnable:**
```bash
chmod +x train.sh
```

**Submit job to HTCondor:**
```bash
condor_submit train.sub
```

**Expected output:**
```
Submitting job(s).
1 job(s) submitted to cluster 12345678.
```

Your job ID is `12345678` - save this number!

---

### Step 5: Monitor Your Job

**Check if job is running:**
```bash
condor_q
```

**Expected output while running:**
```
-- Schedd: ap2001.chtc.wisc.edu : <128.104.101.92:9618?...
OWNER     BATCH_NAME    SUBMITTED   DONE   RUN    IDLE  TOTAL JOB_IDS
nomatteson train.sh     10/3 14:30    _      1      _      1 12345678.0

Total for query: 1 jobs; 0 completed, 0 removed, 0 idle, 1 running, 0 held, 0 suspended
```

**Check detailed job status:**
```bash
condor_q -better-analyze 12345678
```

**Watch live training output** (updates every few seconds):
```bash
tail -f train_*.out
```

You should see:
```
==========================================
CHTC Brain-to-Text Training Job
==========================================
Hostname: e389.chtc.wisc.edu
Date: Thu Oct  3 14:35:12 CDT 2024
...
Checking for GPU...
...
Starting training (120k batches, ~8 hours)
==========================================
Step 200/120000 | Loss: 26.64 | LR: 0.001000
...
```

**Monitor GPU usage** (if you get interactive access):
```bash
nvidia-smi -l 1  # Updates every second
```

**Check error log** (should be empty if all is well):
```bash
tail -f train_*.err
```

**Check HTCondor event log:**
```bash
tail -f train_*.log
```

---

### Step 6: When Job Completes

**Job will automatically email you at nomatteson@wisc.edu when done!**

**Download results to your laptop:**
```bash
# From your laptop (new terminal)
scp nomatteson@ap2001.chtc.wisc.edu:~/results.tar.gz .
```

**Extract checkpoints:**
```bash
tar -xzf results.tar.gz
# Creates: trained_models/baseline_rnn/
#   - checkpoint_step2000.pt
#   - checkpoint_step4000.pt
#   - ...
#   - checkpoint_step120000.pt
#   - training_log.txt
```

**Clean up CHTC (optional but recommended):**
```bash
ssh nomatteson@ap2001.chtc.wisc.edu
rm train_*.log train_*.out train_*.err results.tar.gz
rm /staging/nomatteson/brain2text_data.tar.gz  # Free up 24GB!
```

---

## Expected Timeline

| Stage | Duration | What's Happening |
|-------|----------|------------------|
| Job submission | < 1 min | HTCondor receives job |
| Queued (IDLE) | 5 min - 2 hrs | Waiting for GPU node |
| Starting (RUN) | 2-3 min | Transferring 24GB data to worker node |
| GPU check | 10 sec | Verifying CUDA/GPU |
| Data extraction | 2-3 min | Extracting 24GB tarball |
| Dependency install | 1-2 min | Installing h5py, scipy, etc. |
| **Training** | **~8 hours** | **Main computation** |
| Packaging results | 30 sec | Creating results.tar.gz |
| Transfer back | 1 min | Sending results to /home |
| **TOTAL** | **~8-10 hours** | - |

---

## Resource Details

**What the job requests:**
- **GPU:** 1x GPU with ≥15GB memory
  - Will match: A100 (40GB/80GB), V100 (16GB), L40 (48GB)
  - Won't match: RTX 2080 Ti (11GB - too small)
- **CPU:** 4 cores for parallel data loading
- **RAM:** 32 GB (model uses ~8GB, rest for data batching)
- **Disk:** 60 GB breakdown:
  - Data tarball: 24GB
  - Extracted data: 24GB
  - Checkpoints: ~10GB (60 checkpoints × 170MB each)
  - Overhead: 2GB

**Available GPU types at CHTC:**
- 🔥 **NVIDIA A100 80GB** (9 servers, 36 GPUs) - BEST
- ⚡ **NVIDIA A100 40GB** (2 servers, 8 GPUs) - GREAT
- ✅ **NVIDIA L40 48GB** (3 servers, 30 GPUs) - GOOD
- ✅ **NVIDIA V100 16GB** (2 servers, 4 GPUs) - GOOD
- ❌ **RTX 2080 Ti 11GB** - Too small (won't match)

---

## Troubleshooting

### ❌ Job stays IDLE for > 2 hours

**Check what's wrong:**
```bash
condor_q -better-analyze 12345678
```

**Common causes:**
1. **No GPUs available** - All GPUs busy
   - **Fix:** Wait, or reduce requirements in `train.sub`
2. **Staging access denied**
   - **Fix:** Verify `/staging/nomatteson/` exists and has your data
3. **Resource request too high**
   - **Fix:** Reduce `request_memory` from 32GB → 24GB

### ❌ Job goes on HOLD

**Check hold reason:**
```bash
condor_q -hold
```

**Common reasons:**
1. **"Disk quota exceeded"**
   - Data extraction needs 48GB (24GB compressed + 24GB extracted)
   - **Fix:** Increase `request_disk = 60GB` → `request_disk = 70GB`
2. **"Cannot access file"**
   - Data tarball not found in /staging
   - **Fix:** Verify `ls -lh /staging/nomatteson/brain2text_data.tar.gz`

**Release job after fixing:**
```bash
condor_release 12345678
```

### ❌ Job runs but fails immediately

**Check error log:**
```bash
cat train_*.err
```

**Check output log:**
```bash
cat train_*.out
```

**Common errors:**

**"No module named 'h5py'"**
- Container's pip install failed
- **Fix:** Add `--user` flag in train.sh: `pip install --user h5py scipy ...`

**"CUDA out of memory"**
- Batch size too large for GPU
- **Fix:** Edit `configs/rnn_args.yaml` before packaging:
  ```yaml
  batch_size: 32  # Reduce from 64
  ```

**"No such file: data/hdf5_data_final"**
- Data tarball didn't extract correctly
- **Fix:** Check tarball structure: `tar -tzf brain2text_data.tar.gz | head`

### ❌ Training starts but crashes mid-way

**Check for OOM (Out of Memory):**
```bash
grep -i "memory\|killed\|oom" train_*.err train_*.out
```

**Fix:** Reduce batch size or model size in config before re-submitting

### ⚠️ GPU not detected

**Verify GPU in output:**
```bash
grep -i "gpu\|cuda" train_*.out
```

Should show:
```
CUDA available: True
CUDA devices: 1
```

If shows `CUDA available: False`:
- Job matched to CPU-only node (shouldn't happen with `request_gpus = 1`)
- **Fix:** Check submit file has `request_gpus = 1`

---

## Pro Tips

### 🧪 Test Job First (HIGHLY RECOMMENDED)

Before running full 8-hour training, test with 500 batches (~5 minutes):

**Edit train.sh:**
```bash
# Change this line:
python3 -m src.train_baseline --config configs/rnn_args.yaml

# To this:
python3 -m src.train_baseline --config configs/rnn_args.yaml --num-batches 500
```

**Submit test job:**
```bash
condor_submit train.sub
```

**Verify test completes successfully, then run full training.**

### 🚀 Speed Optimizations

**Use faster config** (30% speedup):
- Edit `train.sh` to use `rnn_args_fast.yaml` instead of `rnn_args.yaml`
- Reduces smoothing kernel size and dataloader workers

**Request specific fast GPU:**
```bash
# In train.sub, change Requirements line to prefer A100:
Requirements = (Target.HasCHTCStaging == true) && (CUDADriverVersion >= 11.0) && (CUDAGlobalMemoryMb >= 15000) && (GPUDeviceName == "A100-SXM4-80GB")
```

### 📊 Monitor Training Progress Remotely

**Set up automatic log sync to your laptop:**
```bash
# On your laptop, run this every 5 minutes:
while true; do
    scp nomatteson@ap2001.chtc.wisc.edu:~/train_*.out .
    tail -20 train_*.out
    sleep 300
done
```

### 🔄 Resume from Checkpoint

If job fails at step 50k, you can resume:

**Edit train.sh to load checkpoint:**
```python
# Add to train_baseline.py call:
python3 -m src.train_baseline \
    --config configs/rnn_args.yaml \
    --init_checkpoint_path trained_models/baseline_rnn/checkpoint_step50000.pt
```

---

## GitHub Workflow (Alternative to Tarballs)

You mentioned wanting to use GitHub - here's how:

### Setup GitHub Repository

**1. Create .gitignore to exclude data:**
```bash
echo "data/" >> .gitignore
echo "*.tar.gz" >> .gitignore
echo "trained_models/" >> .gitignore
git add .gitignore
git commit -m "Exclude data and outputs"
```

**2. Push code to GitHub:**
```bash
git add src/ configs/ requirements.txt chtc/
git commit -m "Add CHTC training setup"
git push origin master
```

**3. On CHTC, clone repo instead of using tarball:**

**Edit train.sh:**
```bash
#!/bin/bash
set -e

# Clone code from GitHub
git clone https://github.com/YOUR_USERNAME/BRAIN2TEXT.git
cd BRAIN2TEXT

# Extract data (still from staging)
tar -xzf ../brain2text_data.tar.gz

# Install deps
pip install -r requirements.txt

# Run training
python3 -m src.train_baseline --config configs/rnn_args.yaml

# Package results
tar -czf ../results.tar.gz trained_models/
```

**Edit train.sub:**
```bash
# Remove brain2text_code.tar.gz from transfer_input_files
transfer_input_files = osdf:///chtc/staging/nomatteson/brain2text_data.tar.gz
```

**Benefits:**
- ✅ No need to create/upload code tarball each time
- ✅ Just push to GitHub and re-submit
- ✅ Always running latest code

**Downside:**
- ❌ Requires internet on worker node (usually fine)

---

## Additional Resources

- **CHTC GPU Jobs Guide:** https://chtc.cs.wisc.edu/uw-research-computing/gpu-jobs
- **Large Data Guide:** https://chtc.cs.wisc.edu/uw-research-computing/file-avail-largedata
- **Office Hours:** Tuesdays 10:30am-12pm, Thursdays 3-4:30pm
  - Zoom: go.wisc.edu/chtc-officehours
- **Email Support:** chtc@cs.wisc.edu
- **System Status:** https://status.chtc.wisc.edu

---

## CHTC is FREE! 🎉

No charge for compute time. Use as much as you need for research!
