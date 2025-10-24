import os
import torch
import numpy as np
import pandas as pd
import redis
import yaml
import time
from tqdm import tqdm
import jiwer
import argparse
import re

# Add project root to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phoneme_dataset import PhonemeDataset, collate_phoneme_batch
from src.rnn_model import GRUDecoder
from torch.utils.data import DataLoader

# --- Official Helper Functions (adapted from evaluate_model_helpers.py) ---

def rearrange_speech_logits_pt(logits):
    # original order is [BLANK, phonemes..., SIL]
    # rearrange so the order is [BLANK, SIL, phonemes...]
    logits = np.concatenate((logits[:, :, 0:1], logits[:, :, -1:], logits[:, :, 1:-1]), axis=-1)
    return logits

def get_current_redis_time_ms(redis_conn):
    t = redis_conn.time()
    return int(t[0]*1000 + t[1]/1000)

def reset_remote_language_model(r, last_entry_seen):
    r.xadd('remote_lm_reset', {'done': 0})
    time.sleep(0.001)
    response = []
    while len(response) == 0:
        response = r.xread({'remote_lm_done_resetting': last_entry_seen}, count=1, block=10000)
    return response[0][1][0][0] # Return the new entry ID

def send_logits_to_remote_lm(r, last_entry_seen, logits):
    r.xadd('remote_lm_input', {'logits': np.float32(logits).tobytes()})
    response = []
    while len(response) == 0:
        response = r.xread({'remote_lm_output_partial': last_entry_seen}, count=1, block=10000)
    return response[0][1][0][0], response[0][1][0][1][b'lm_response_partial'].decode()

def finalize_remote_lm(r, last_entry_seen):
    r.xadd('remote_lm_finalize', {'done': 0})
    time.sleep(0.005)
    response = []
    while len(response) == 0:
        response = r.xread({'remote_lm_output_final': last_entry_seen}, count=1, block=10000)
    
    entry_id, entry_data = response[0][1][0]
    
    candidate_sentences = [str(c) for c in entry_data[b'scoring'].decode().split(';')[::5]]
    if len(candidate_sentences) > 0:
        best_sentence = candidate_sentences[0]
    else:
        best_sentence = ""
        
    return entry_id, best_sentence

def remove_punctuation(sentence):
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence).lower()
    sentence = sentence.replace('- ', ' ').replace('--', '').replace(" '", "'").strip()
    return ' '.join([word for word in sentence.split() if word != ''])

# --- Main Evaluation Function ---

def evaluate(checkpoint_path, split, config_path_for_mapping, redis_host, redis_port):
    print(f"--- Starting evaluation for split: {split} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model and Config
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model_cfg = config['model']
    dataset_cfg = config['dataset']
    print("Model loaded successfully.")

    with open(config_path_for_mapping, 'r') as f:
        map_config = yaml.safe_load(f)
    sessions = map_config['dataset']['sessions']
    session_to_day = {sess: idx for idx, sess in enumerate(sessions)}

    model = GRUDecoder(
        neural_dim=model_cfg['n_input_features'],
        n_units=model_cfg['n_units'],
        n_layers=model_cfg['n_layers'],
        n_days=len(sessions),
        n_classes=dataset_cfg['n_classes'],
        rnn_dropout=model_cfg['rnn_dropout'],
        input_dropout=model_cfg['input_network']['input_layer_dropout'],
        patch_size=model_cfg['patch_size'],
        patch_stride=model_cfg['patch_stride'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 2. Prepare Dataset
    dataset = PhonemeDataset(
        dataset_cfg['dataset_dir'].replace('../', ''), split, session_to_day, augment=False
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_phoneme_batch)

    # 3. Connect to Redis
    print(f"Connecting to Redis server at {redis_host}:{redis_port}...")
    r = redis.Redis(host=redis_host, port=redis_port, db=0)
    r.ping()
    r.flushall()
    print("Connected to Redis.")

    # 4. Run Evaluation Loop
    all_predictions = []
    all_ground_truths = []

    # Set initial timestamps for Redis streams
    reset_last_id = get_current_redis_time_ms(r)
    partial_last_id = get_current_redis_time_ms(r)
    final_last_id = get_current_redis_time_ms(r)

    with torch.no_grad():
        for x, y, x_lens, y_lens, day_idxs, sess_names, trial_keys, sentences in tqdm(dataloader, desc=f"Evaluating on {split} split"):
            
            # Reset the language model state for the new sentence
            reset_last_id = reset_remote_language_model(r, reset_last_id)
            
            # Get logits from our trained model
            logits = model(x.to(device), day_idxs.to(device)).cpu().numpy()
            
            # The official LM expects a different phoneme order.
            logits = rearrange_speech_logits_pt(logits)
            
            # Send logits to the LM. We only send one chunk for offline evaluation.
            partial_last_id, _ = send_logits_to_remote_lm(r, partial_last_id, logits)
            
            # Tell the LM we are done and get the final best sentence
            final_last_id, best_sentence = finalize_remote_lm(r, final_last_id)
            
            all_predictions.append(best_sentence)
            all_ground_truths.extend(sentences)

    # 5. Calculate and Print Word Error Rate (WER)
    clean_ground_truths = [remove_punctuation(s) for s in all_ground_truths]
    clean_predictions = [remove_punctuation(s) for s in all_predictions]

    wer = jiwer.wer(clean_ground_truths, clean_predictions)
    print(f"\n--- AGGREGATE WORD ERROR RATE (WER) FOR '{split}' SPLIT: {wer * 100:.2f}% ---")

    # 6. Print some examples
    print("\n--- Example Decoded Sentences ---")
    for i in range(min(10, len(all_predictions))):
        print(f"  Prediction:   '{clean_predictions[i]}'")
        print(f"  Ground Truth: '{clean_ground_truths[i]}'")
        print("-" * 20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model using the official language model decoder via Redis.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pt file).')
    parser.add_argument('--config_path_for_mapping', type=str, default='configs/rnn_official_exact.yaml', help='Path to the original config file to get session mapping.')
    parser.add_argument('--split', type=str, default='val', help='Data split to evaluate (e.g., "val" or "test").')
    parser.add_argument('--redis_host', type=str, default='localhost', help='Redis server host.')
    parser.add_argument('--redis_port', type=int, default=6379, help='Redis server port.')
    args = parser.parse_args()
    
    evaluate(args.checkpoint, args.split, args.config_path_for_mapping, args.redis_host, args.redis_port)
