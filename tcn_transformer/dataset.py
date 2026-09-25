"""
seq_L{5,10,20}/{train,val,test}.npz 를 PyTorch Dataset으로 포장.
X: (N, L, 9), is_padding: (N, L), y: (N,)
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceFraudDataset(Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.X = d["X"].astype(np.float32)                # (N, L, 9)
        self.is_padding = d["is_padding"].astype(np.float32)  # (N, L) 1=padding
        self.y = d["y"].astype(np.float32)                 # (N,)
        self.feature_cols = list(d["feature_cols"])
        self.L = self.X.shape[1]
        self.n_features = self.X.shape[2]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])                  # (L, 9)
        pad = torch.from_numpy(self.is_padding[idx])        # (L,)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, pad, y
