import h5py, numpy as np, glob, os
import re

def compute_normalization(train_root):
    """Compute global mean and std for normalization from training data"""
    sums, sqs, count = None, None, 0

    # Handle both 'data' and 'data/hdf5_data_final' as input
    if os.path.basename(train_root) == 'hdf5_data_final':
        # Already pointing to hdf5_data_final
        files = glob.glob(os.path.join(train_root, 't15.*', 'data_train.hdf5'))
    else:
        # Pointing to parent directory
        files = glob.glob(os.path.join(train_root, 'hdf5_data_final', 't15.*', 'data_train.hdf5'))

    if not files:
        raise RuntimeError(f"No training files found in {train_root}")

    print(f"Found {len(files)} training files for normalization")

    for fp in files:
        with h5py.File(fp, 'r') as f:
            for k in f.keys():
                if not k.startswith('trial_'): continue
                x = np.array(f[k]['input_features'], dtype=np.float32)  # (T, 512)
                if sums is None:
                    sums = np.zeros(x.shape[1], np.float64)
                    sqs = np.zeros_like(sums)
                sums += x.sum(axis=0)
                sqs += (x**2).sum(axis=0)
                count += x.shape[0]
    mean = (sums / count).astype(np.float32)
    var = (sqs / count) - mean.astype(np.float64)**2
    std = np.sqrt(np.clip(var, 1e-6, None)).astype(np.float32)
    return mean, std

def normalize_text_for_wer(text):
    """
    Normalize text according to competition rules:
    - Remove punctuation
    - Lowercase
    - Words separated by single spaces
    """
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase
    text = text.lower()
    # Normalize whitespace to single spaces
    text = ' '.join(text.split())
    return text

def compute_wer(reference, hypothesis):
    """
    Compute Word Error Rate (WER) between reference and hypothesis.
    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=words in reference
    """
    # Normalize both strings
    ref = normalize_text_for_wer(reference)
    hyp = normalize_text_for_wer(hypothesis)

    # Split into words
    ref_words = ref.split()
    hyp_words = hyp.split()

    # Compute edit distance (Levenshtein) at word level
    n, m = len(ref_words), len(hyp_words)

    if n == 0:
        return 1.0 if m > 0 else 0.0

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialize
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # deletion
                    dp[i][j-1] + 1,    # insertion
                    dp[i-1][j-1] + 1   # substitution
                )

    return dp[n][m] / n