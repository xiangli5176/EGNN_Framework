from __future__ import print_function, division

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader



class LoadDataset_train(Dataset):
    def __init__(self, data, device = 'cpu'):
        """
        """
        self.x = data
        self.device = device

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        """Make sure idx is of long type"""
        return torch.from_numpy(np.array(idx, dtype=np.int64)).to(self.device), self.x[idx].to(self.device)

class LoadDataset_pretrain(Dataset):
    def __init__(self, data, device = 'cpu'):
        """
            data: is the target training tensor
        """
        self.x = data
        self.device = device

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        """Make sure idx is of long type"""
        return  torch.from_numpy(np.array(idx, dtype=np.int64)).to(self.device), self.x[idx].to(self.device)