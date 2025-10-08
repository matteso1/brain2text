"""
Debug script to see what the model is actually predicting
"""
import torch
from src.data import NeuralTextDataset, collate_batch
from src.tokenization import SimpleSubwordTokenizer
from src.model import ConformerRNNT
from torch.utils.data import DataLoader

print("Loading checkpoint...")
saved = torch.load('../runs/checkpoints/conformer_ctc_ema.pt', map_location='cpu')
vocab = saved['tokenizer']
tokenizer = SimpleSubwordTokenizer(vocab=vocab)
mean, std = saved['mean'], saved['std']

print(f"Tokenizer vocab size: {len(vocab)}")
print(f"First 20 vocab items: {list(vocab.items())[:20]}")

# Load model
model = ConformerRNNT(in_dim=512, d_model=384, num_blocks=16, vocab_size=len(vocab))
model.load_state_dict(saved['model'])
model.eval()

# Load a few validation samples
val_ds = NeuralTextDataset('data', 'val', tokenizer, mean=mean, std=std)
val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=lambda x: collate_batch(x))

print(f"\nLoading {len(val_ds)} validation samples...")
print("="*80)

def greedy_ctc_decode(logits_BTV, tokenizer):
    pred = logits_BTV.argmax(-1).cpu().numpy()
    texts = []
    blank_id = 0
    for b in range(pred.shape[0]):
        prev = None
        ids = []
        for t in range(pred.shape[1]):
            p = int(pred[b, t])
            if p == blank_id:
                prev = p
                continue
            if prev != p:
                ids.append(p)
            prev = p
        text = tokenizer.decode(ids)
        texts.append(text)
    return texts

# Test on first batch
with torch.no_grad():
    for x, y_pad, x_lens, y_lens, texts, *_ in val_dl:
        enc, enc_lens = model.forward_encoder(x, x_lens)
        logits = model.ctc_head(enc)
        hyps = greedy_ctc_decode(logits, tokenizer)

        for i in range(len(texts)):
            print(f"Sample {i+1}:")
            print(f"  Reference: '{texts[i]}'")
            print(f"  Prediction: '{hyps[i]}'")
            print(f"  Ref length: {len(texts[i])}, Hyp length: {len(hyps[i])}")

            # Show token IDs
            ref_ids = y_pad[i, :y_lens[i]].tolist()
            print(f"  Ref token IDs (first 20): {ref_ids[:20]}")
            print()

        break  # Only first batch

print("="*80)
print("\nDiagnostics:")
print(f"Vocab coverage test:")
test_text = "Hello world this is a test"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
print(f"  Input: '{test_text}'")
print(f"  Encoded: {encoded}")
print(f"  Decoded: '{decoded}'")
