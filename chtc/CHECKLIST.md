# CHTC Submission Checklist

Use this checklist to make sure you don't miss any steps before submitting your job.

---

## ☑️ Pre-Submission Checklist

### On Your Laptop

- [ ] **Package code tarball**
  ```bash
  cd ~/PycharmProjects/BRAIN2TEXT
  tar -czf brain2text_code.tar.gz src/ configs/ --exclude="*.pyc" --exclude="__pycache__"
  ```

- [ ] **Verify code size < 1MB**
  ```bash
  ls -lh brain2text_code.tar.gz
  ```

- [ ] **Package data tarball** (takes 5-10 min)
  ```bash
  tar -czf brain2text_data.tar.gz data/hdf5_data_final/
  ```

- [ ] **Verify data size ~15-24 GB**
  ```bash
  ls -lh brain2text_data.tar.gz
  ```

### Transfer to CHTC

- [ ] **Upload code to /home**
  ```bash
  scp brain2text_code.tar.gz chtc/train.sh chtc/train.sub nomatteson@ap2001.chtc.wisc.edu:~/
  ```

- [ ] **Upload data to /staging** (takes 10-30 min)
  ```bash
  scp brain2text_data.tar.gz nomatteson@transfer.chtc.wisc.edu:/staging/nomatteson/
  ```

- [ ] **Verify uploads**
  ```bash
  ssh nomatteson@ap2001.chtc.wisc.edu
  ls -lh ~/brain2text_code.tar.gz ~/train.sh ~/train.sub
  ls -lh /staging/nomatteson/brain2text_data.tar.gz
  ```

### On CHTC (via SSH)

- [ ] **Make script executable**
  ```bash
  chmod +x train.sh
  ```

- [ ] **Verify submit file uses correct paths**
  ```bash
  grep transfer_input_files train.sub
  # Should show: osdf:///chtc/staging/nomatteson/brain2text_data.tar.gz
  ```

- [ ] **(OPTIONAL) Test with 500 batches first**
  ```bash
  # Edit train.sh to add --num-batches 500
  nano train.sh
  ```

- [ ] **Submit job**
  ```bash
  condor_submit train.sub
  ```

- [ ] **Save job ID**
  ```
  Job ID: ____________ (write it down!)
  ```

---

## 📊 Post-Submission Checklist

### Immediate (first 5 minutes)

- [ ] **Check job status**
  ```bash
  condor_q
  ```
  Expected: `1 idle` or `1 running`

- [ ] **If HELD, check why**
  ```bash
  condor_q -hold
  ```

### After 10 minutes

- [ ] **Verify job started running**
  ```bash
  condor_q
  ```
  Should show: `1 running`

- [ ] **Check output file exists**
  ```bash
  ls -lh train_*.out
  tail -20 train_*.out
  ```
  Should see: "Checking for GPU...", "CUDA available: True"

### After 30 minutes

- [ ] **Verify training has started**
  ```bash
  tail -50 train_*.out | grep "Step"
  ```
  Should see: "Step 200/120000 | Loss: ..."

- [ ] **Check no errors**
  ```bash
  cat train_*.err
  ```
  Should be empty or minimal warnings

### Periodic Checks (every 2-4 hours)

- [ ] **Monitor training progress**
  ```bash
  tail train_*.out
  ```

- [ ] **Check job still running**
  ```bash
  condor_q
  ```

---

## ✅ Completion Checklist

### When Job Finishes (~8 hours later)

- [ ] **Check email for completion notification**
  Look for email from HTCondor at nomatteson@wisc.edu

- [ ] **Verify results file exists**
  ```bash
  ssh nomatteson@ap2001.chtc.wisc.edu
  ls -lh results.tar.gz
  ```

- [ ] **Download results to laptop**
  ```bash
  scp nomatteson@ap2001.chtc.wisc.edu:~/results.tar.gz .
  ```

- [ ] **Extract and verify checkpoints**
  ```bash
  tar -xzf results.tar.gz
  ls -lh trained_models/baseline_rnn/
  ```
  Should see: 60 checkpoint files + training_log.txt

- [ ] **Clean up CHTC files** (optional but nice)
  ```bash
  ssh nomatteson@ap2001.chtc.wisc.edu
  rm train_*.log train_*.out train_*.err results.tar.gz brain2text_code.tar.gz
  rm /staging/nomatteson/brain2text_data.tar.gz  # Frees 24GB
  ```

---

## 🚨 Troubleshooting Checklist

If something goes wrong:

### Job stays IDLE > 2 hours

- [ ] Run `condor_q -better-analyze <job_id>`
- [ ] Check GPU availability on CHTC status page
- [ ] Consider reducing resource requirements

### Job goes on HOLD

- [ ] Run `condor_q -hold` to see reason
- [ ] Fix issue (usually disk space or file access)
- [ ] Run `condor_release <job_id>`

### Job runs but fails

- [ ] Check `cat train_*.err` for error messages
- [ ] Check `cat train_*.out` for where it failed
- [ ] Verify data tarball structure: `tar -tzf brain2text_data.tar.gz | head`

### Can't find output files

- [ ] Check if job is still running: `condor_q`
- [ ] Check if job was removed: `condor_history <job_id>`
- [ ] Look in current directory: `ls -lrt | tail`

---

## 📝 Quick Reference

| Command | Purpose |
|---------|---------|
| `condor_submit train.sub` | Submit job |
| `condor_q` | Check job status |
| `condor_q -better-analyze <id>` | Debug why job is idle |
| `condor_q -hold` | See why job is on hold |
| `condor_release <id>` | Release held job |
| `condor_rm <id>` | Cancel/remove job |
| `tail -f train_*.out` | Watch live output |
| `cat train_*.err` | Check errors |

---

## 🎯 Expected Timeline

| Time | What should be happening |
|------|--------------------------|
| 0 min | Job submitted (IDLE) |
| 5-120 min | Queued, waiting for GPU (IDLE) |
| 120 min | Job starts (RUN) |
| 123 min | Data extracted, training starts |
| 125 min | First log output: "Step 200/120000" |
| 8 hours | Training completes |
| 8h 5min | Results transferred back |
| 8h 5min | Email notification sent ✅ |

---

**IMPORTANT REMINDERS:**

1. ⚠️ Data MUST go to `/staging/nomatteson/` (not `/home`) - it's 24GB!
2. ⚠️ Use `transfer.chtc.wisc.edu` for uploading to staging, NOT `ap2001.chtc.wisc.edu`
3. ⚠️ Test with 500 batches first before running full 8-hour job
4. ✅ Job will email you when done - check nomatteson@wisc.edu
5. ✅ CHTC is free - no charge for GPU time!

---

**Questions? Contact:**
- CHTC Email: chtc@cs.wisc.edu
- Office Hours: Tuesdays 10:30am-12pm, Thursdays 3-4:30pm
- Status Page: https://status.chtc.wisc.edu
