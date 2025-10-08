"""
Watch actual training progress
"""
import os
import time

log_file = '../trained_models/baseline_rnn/training_log.txt'

print("Watching training progress (Ctrl+C to stop)...")
print("=" * 60)

last_size = 0
while True:
    try:
        if os.path.exists(log_file):
            current_size = os.path.getsize(log_file)
            if current_size > last_size:
                with open(log_file, 'r') as f:
                    f.seek(last_size)
                    new_content = f.read()
                    print(new_content, end='')
                last_size = current_size
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopped watching.")
        break
