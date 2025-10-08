"""
Quick script to monitor training progress
Run with: python monitor_training.py [--tail]
"""
import os
import time
import sys

def tail_log(filepath, n=20):
    """Show last N lines of log file"""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        lines = f.readlines()
        return lines[-n:]

def watch_log(filepath, interval=5):
    """Watch log file for updates"""
    print(f"Watching {filepath} (Ctrl+C to stop)")
    print("=" * 60)

    if not os.path.exists(filepath):
        print(f"Waiting for log file to be created...")

    last_size = 0
    while True:
        try:
            if os.path.exists(filepath):
                current_size = os.path.getsize(filepath)
                if current_size > last_size:
                    with open(filepath, 'r') as f:
                        f.seek(last_size)
                        new_content = f.read()
                        print(new_content, end='')
                    last_size = current_size
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\nStopped watching.")
            break

print("Training Monitor")
print("=" * 60)

# Check if process is running
print("\n1. Checking if training is running...")
os.system('tasklist | findstr python')

# Check log file
log_file = 'runs/baseline_training/training_log.txt'
if os.path.exists(log_file):
    size_kb = os.path.getsize(log_file) / 1024
    mod_time = time.ctime(os.path.getmtime(log_file))
    print(f"\n2. Training log found: {log_file}")
    print(f"   Size: {size_kb:.1f} KB")
    print(f"   Last modified: {mod_time}")

    print(f"\n3. Last 20 lines of log:")
    print("-" * 60)
    lines = tail_log(log_file, 20)
    if lines:
        for line in lines:
            print(line.rstrip())
    print("-" * 60)
else:
    print(f"\n2. No log file yet at {log_file}")
    print("   Training may not have started yet.")

# Check for checkpoints
ckpt_dir = 'runs/baseline_training'
if os.path.exists(ckpt_dir):
    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith('checkpoint_')]
    if ckpts:
        print(f"\n4. Found {len(ckpts)} intermediate checkpoints:")
        for ckpt in sorted(ckpts):
            ckpt_path = os.path.join(ckpt_dir, ckpt)
            size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
            print(f"   - {ckpt} ({size_mb:.1f} MB)")

# Final checkpoint
final_ckpt = 'runs/baseline_checkpoints/gru_baseline.pt'
if os.path.exists(final_ckpt):
    size_mb = os.path.getsize(final_ckpt) / (1024 * 1024)
    mod_time = time.ctime(os.path.getmtime(final_ckpt))
    print(f"\n5. ✓ FINAL checkpoint found: {final_ckpt}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   Last modified: {mod_time}")
    print("\n   Training is COMPLETE! Run: python check_results.py")
else:
    print(f"\n5. Final checkpoint not created yet")
    print("   Training still in progress...")

print("\n" + "=" * 60)
print("Options:")
print("  python monitor_training.py          - Show current status (this)")
print("  python monitor_training.py --tail   - Live tail log file")
print("  python check_results.py             - Validate final results")
print("=" * 60)

# Handle --tail option
if '--tail' in sys.argv:
    print()
    watch_log(log_file)
