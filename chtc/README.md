# CHTC Training Setup

This folder contains everything needed to run training on UW-Madison's CHTC GPU cluster.

## Files

- **`train.sub`** - HTCondor submit file (GPU job configuration)
- **`train.sh`** - Bash script that runs inside Docker container
- **`CHTC_SETUP.md`** - Complete setup guide
- **`CHECKLIST.md`** - Step-by-step checklist

## Quick Start (Using GitHub)

### First Time Setup

**1. On your laptop:**
```bash
# Push code to GitHub (data stays local!)
git add src/ configs/ chtc/ requirements.txt .gitignore
git commit -m "Add CHTC training setup"
git push origin master
```

**2. Upload data to CHTC /staging (one-time):**
```bash
# Create data tarball
tar -czf brain2text_data.tar.gz data/hdf5_data_final/

# Upload to CHTC staging (takes 10-30 min)
scp brain2text_data.tar.gz nomatteson@transfer.chtc.wisc.edu:/staging/nomatteson/
```

**3. SSH into CHTC and clone repo:**
```bash
ssh nomatteson@ap2001.chtc.wisc.edu
cd ~
git clone https://github.com/YOUR_USERNAME/BRAIN2TEXT.git b2txt25
cd b2txt25
```

### Running Training

**Update code (after making changes):**
```bash
# On CHTC
cd ~/b2txt25
git pull
```

**Submit job:**
```bash
cd ~/b2txt25/chtc
chmod +x train.sh
condor_submit train.sub
```

**Monitor:**
```bash
condor_q
tail -f train_*_0.out
```

## Benefits of GitHub Workflow

✅ No SCP password/2FA every time
✅ Version control - can rollback if needed
✅ Easy to update code: just `git pull`
✅ Data stays in /staging (uploaded once)

## File Sizes

- **Code (in GitHub):** ~100 KB
- **Data (in /staging):** 8 GB (not in GitHub!)
- **Trained models:** 10 GB (created on CHTC, not in GitHub)

## Notes

- Data tarball must be in `/staging/nomatteson/` before first job
- Code gets pulled from GitHub each time you run `git pull`
- Results download via SCP after job completes
