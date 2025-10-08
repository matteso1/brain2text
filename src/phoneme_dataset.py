"""
Phoneme dataset for baseline GRU training
Loads neural data + phoneme targets (seq_class_ids) with augmentation
"""
import os, glob, h5py, numpy as np, torch
from torch.utils.data import Dataset
import yaml
from src.data_augmentations import gauss_smooth


# Load session mapping from config
def get_session_mapping(config_path='configs/rnn_args.yaml'):
    """Get mapping from session name to day index"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    sessions = config['dataset']['sessions']
    session_to_day = {sess: idx for idx, sess in enumerate(sessions)}
    return session_to_day, sessions


class PhonemeDataset(Dataset):
    """
    Dataset that loads neural features and phoneme targets.
    Compatible with baseline GRU training.
    """
    def __init__(self, root, split, session_to_day,
                 augment=False, aug_config=None,
                 mean=None, std=None):
        """
        root: data root containing hdf5_data_final
        split: 'train'|'val'|'test'
        session_to_day: dict mapping session name to day index
        augment: whether to apply augmentations
        aug_config: augmentation configuration dict
        mean/std: optional normalization vectors (512,)
        """
        self.root = root
        self.split = split
        self.session_to_day = session_to_day
        self.augment = augment
        self.aug_config = aug_config or {}
        self.mean = mean
        self.std = std

        # Find all sessions - handle both 'data' and 'data/hdf5_data_final' as root
        if os.path.basename(root) == 'hdf5_data_final':
            sess_glob = os.path.join(root, 't15.*')
        else:
            sess_glob = os.path.join(root, 'hdf5_data_final', 't15.*')
        sessions = sorted(glob.glob(sess_glob))

        # Build index of (file, trial_key, session_name, day_idx)
        self.index = []
        for sess_path in sessions:
            sess_name = os.path.basename(sess_path)
            if sess_name not in session_to_day:
                continue

            day_idx = session_to_day[sess_name]
            fp = os.path.join(sess_path, f'data_{split}.hdf5')

            if not os.path.exists(fp):
                continue

            with h5py.File(fp, 'r') as f:
                for k in f.keys():
                    if k.startswith('trial_'):
                        self.index.append((fp, k, sess_name, day_idx))

        print(f"{split.upper()}: {len(self.index)} trials from {len(set(x[2] for x in self.index))} sessions")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fp, tk, sess_name, day_idx = self.index[i]

        with h5py.File(fp, 'r') as f:
            g = f[tk]
            x = np.array(g['input_features'], dtype=np.float32)  # (T, 512)
            y = np.array(g['seq_class_ids'], dtype=np.int64)     # (max_seq_len,) phoneme IDs

        # Apply augmentations if training
        if self.augment and self.split == 'train':
            x = self._apply_augmentations(x)
        else:
            # Still apply smoothing to validation (match official implementation!)
            smooth_data = self.aug_config.get('smooth_data', False) if self.aug_config else False
            if smooth_data:
                smooth_kernel_std = self.aug_config.get('smooth_kernel_std', 2)
                smooth_kernel_size = self.aug_config.get('smooth_kernel_size', 100)
                x_tensor = torch.from_numpy(x).unsqueeze(0)
                x_tensor = gauss_smooth(x_tensor, 'cpu', smooth_kernel_std, smooth_kernel_size)
                x = x_tensor.squeeze(0).numpy()

        # Normalize
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-6)

        # Remove padding from phoneme targets
        y_len = (y != 0).sum()  # Assuming 0 is padding
        if y_len == 0:  # Handle edge case
            y_len = 1
            y = np.array([0], dtype=np.int64)
        else:
            y = y[:y_len]

        return torch.from_numpy(x), torch.from_numpy(y), day_idx, sess_name, tk

    def _apply_augmentations(self, x):
        """Apply data augmentations to neural data - MATCH OFFICIAL ORDER"""
        # Random cut - remove random frames from beginning
        random_cut = self.aug_config.get('random_cut', 0)
        if random_cut > 0:
            cut_amt = np.random.randint(0, random_cut + 1)
            x = x[cut_amt:]

        # White noise BEFORE smoothing (official order!)
        white_noise_std = self.aug_config.get('white_noise_std', 0.0)
        if white_noise_std > 0:
            noise = np.random.randn(*x.shape).astype(np.float32) * white_noise_std
            x = x + noise

        # Constant offset BEFORE smoothing (official order!)
        constant_offset_std = self.aug_config.get('constant_offset_std', 0.0)
        if constant_offset_std > 0:
            offset = np.random.randn(1, x.shape[1]).astype(np.float32) * constant_offset_std
            x = x + offset

        # Gaussian smoothing AFTER noise (official order!) - reduces effective noise magnitude
        smooth_data = self.aug_config.get('smooth_data', False)
        if smooth_data:
            smooth_kernel_std = self.aug_config.get('smooth_kernel_std', 2)
            smooth_kernel_size = self.aug_config.get('smooth_kernel_size', 100)

            # Convert to tensor, smooth, convert back
            x_tensor = torch.from_numpy(x).unsqueeze(0)  # (1, T, D)
            x_tensor = gauss_smooth(x_tensor, 'cpu', smooth_kernel_std, smooth_kernel_size)
            x = x_tensor.squeeze(0).numpy()

        return x


def collate_phoneme_batch(batch):
    """
    Collate function for phoneme dataset.
    Returns padded tensors suitable for CTC loss.
    """
    xs, ys, day_idxs, sess_names, trial_keys = zip(*batch)

    # Pad neural features
    T_max = max(x.shape[0] for x in xs)
    B = len(batch)
    D = xs[0].shape[1]

    x_pad = torch.zeros(B, T_max, D, dtype=torch.float32)
    x_lens = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)

    for i, x in enumerate(xs):
        x_pad[i, :x.shape[0]] = x

    # Pad phoneme targets
    y_lens = torch.tensor([y.shape[0] for y in ys], dtype=torch.long)
    y_max = max(int(l) for l in y_lens)
    y_pad = torch.zeros(B, y_max, dtype=torch.long)

    for i, y in enumerate(ys):
        y_pad[i, :y.shape[0]] = y

    day_idxs = torch.tensor(day_idxs, dtype=torch.long)

    return x_pad, y_pad, x_lens, y_lens, day_idxs, sess_names, trial_keys
