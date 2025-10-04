import os, glob, json, numpy as np, torch
from torch.utils.data import DataLoader
from src.data import NeuralTextDataset, collate_batch
from src.tokenization import SimpleSubwordTokenizer
from src.model import ConformerRNNT
from src.utils import compute_wer

def session_cv(data_root='data', ckpt='runs/checkpoints/conformer_ctc_ema.pt', batch_size=8, num_workers=4, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    saved = torch.load(ckpt, map_location='cpu')
    vocab = saved['tokenizer']
    tokenizer = SimpleSubwordTokenizer(vocab=vocab)
    id2tok = {i:t for t,i in vocab.items()}
    mean, std = saved.get('mean', None), saved.get('std', None)

    sessions = sorted([os.path.basename(p) for p in glob.glob(os.path.join(data_root, 'hdf5_data_final', 't15.*'))])
    results = {}
    model = ConformerRNNT(in_dim=512, d_model=256, num_blocks=12, nhead=4, p=0.1, vocab_size=len(vocab)).to(device)
    model.load_state_dict(saved['model']); model.eval()

    def greedy_ctc_decode(logits_BTV):
        """Greedy CTC decoding using tokenizer"""
        pred = logits_BTV.argmax(-1).cpu().numpy()
        texts = []
        blank_id = 0

        for b in range(pred.shape[0]):
            # CTC collapse
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

            # Use tokenizer to decode
            text = tokenizer.decode(ids)
            texts.append(text)

        return texts

    with torch.no_grad():
        for sess in sessions:
            ds = NeuralTextDataset(data_root, 'val', tokenizer, mean=mean, std=std, session_list=[sess], warn_empty=False)
            if len(ds) == 0:
                continue

            dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
            wer_list = []

            for x, y_pad, x_lens, y_lens, texts, *_ in dl:
                x, x_lens = x.to(device), x_lens.to(device)
                enc, enc_lens = model.forward_encoder(x, x_lens)
                logits = model.ctc_head(enc)
                hyps = greedy_ctc_decode(logits)

                # Compute WER for each sample
                for b in range(len(texts)):
                    ref = texts[b]
                    hyp = hyps[b]
                    wer = compute_wer(ref, hyp)
                    wer_list.append(wer)

            results[sess] = float(np.mean(wer_list)) if wer_list else None

    print("Session-level WER:", json.dumps(results, indent=2))