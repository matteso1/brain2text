import os, glob, h5py, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

def list_trials(split_dir):
    index = []
    for sess in sorted(glob.glob(os.path.join(split_dir, 't15.*'))):
        for fp in sorted(glob.glob(os.path.join(sess, 'data_*.hdf5'))):
            with h5py.File(fp, 'r') as f:
                for k in f.keys():
                    if k.startswith('trial_'):
                        index.append((fp, k))
    return index

def _decode_sentence_label(label_attr):
    # Handles str, bytes, and numpy bytes scalars/arrays robustly
    if isinstance(label_attr, str):
        return label_attr
    if isinstance(label_attr, (bytes, np.bytes_)):
        try:
            return label_attr.decode('utf-8')
        except Exception:
            return label_attr.decode('latin-1', errors='ignore')
    try:
        as_bytes = np.asarray(label_attr).tobytes()
        return as_bytes.decode('utf-8').rstrip('\x00')
    except Exception:
        return str(label_attr)

class NeuralTextDataset(Dataset):
    def __init__(self, root, split, tokenizer, mean=None, std=None, session_list=None, warn_empty=True):
        """
        root: data root containing hdf5_data_final
        split: 'train'|'val'|'test'
        tokenizer: object with encode(text)->List[int]
        mean/std: optional normalization vectors (512,)
        session_list: optional subset of sessions (list of folder names) to include
        warn_empty: whether to warn if no data found
        """
        self.root = root
        self.split = split
        sess_glob = os.path.join(root, 'hdf5_data_final', 't15.*')
        sessions = sorted(glob.glob(sess_glob))
        if session_list:
            sessions = [s for s in sessions if os.path.basename(s) in session_list]
        self.files = []
        for s in sessions:
            fp = os.path.join(s, f'data_{split}.hdf5')
            if os.path.exists(fp):
                self.files.append(fp)

        if not self.files and warn_empty:
            print(f"Warning: No {split} data files found in {root}")

        self.index = []
        for fp in self.files:
            with h5py.File(fp, 'r') as f:
                for k in f.keys():
                    if k.startswith('trial_'):
                        self.index.append((fp, k))

        if not self.index and warn_empty:
            print(f"Warning: No trials found in {split} split (found {len(self.files)} files)")

        self.tokenizer = tokenizer
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fp, tk = self.index[i]
        with h5py.File(fp, 'r') as f:
            g = f[tk]
            x = np.array(g['input_features'], dtype=np.float32)  # (T, 512)
            text = _decode_sentence_label(g.attrs['sentence_label'])
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-6)
        y = np.array(self.tokenizer.encode(text), dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y), text, os.path.basename(fp), tk

def collate_batch(batch):
    xs, ys, texts, file_names, trial_keys = zip(*batch)
    T = max(x.shape[0] for x in xs)
    B = len(batch)
    D = xs[0].shape[1]
    x_pad = torch.zeros(B, T, D, dtype=torch.float32)
    x_lens = torch.tensor([x.shape[0] for x in xs], dtype=torch.int32)
    y_lens = torch.tensor([y.shape[0] for y in ys], dtype=torch.int32)
    U = max(int(l) for l in y_lens)
    y_pad = torch.full((B, U), fill_value=-1, dtype=torch.long)
    for i,(x,y) in enumerate(zip(xs, ys)):
        x_pad[i, :x.shape[0]] = x
        y_pad[i, :y.shape[0]] = y
    return x_pad, y_pad, x_lens, y_lens, texts, file_names, trial_keys