from __future__ import print_function, division

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader



class LoadDataset_feat(Dataset):
    def __init__(self, data, device = 'cpu'):
        """
            Load batches of feature matrix
        """
        self.x = data
        self.device = device

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        """Make sure idx is of long type"""
        return torch.from_numpy(np.array(idx, dtype=np.int64)).to(self.device), self.x[idx].to(self.device)



class LoadDataset_feat_prob(Dataset):
    def __init__(self, data, p, device = 'cpu'):
        """
            p should always default to be a tensor on the device (cuda)
        """
        self.x = data
        self.p = p
        self.device = device

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        """Make sure idx is of long type"""
        return torch.from_numpy(np.array(idx, dtype=np.int64)).to(self.device), self.x[idx].to(self.device), self.p[idx]
