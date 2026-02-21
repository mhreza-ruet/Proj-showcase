"""Torch Dataset wrapper (on-the-fly only)."""

import torch
from torch.utils.data import Dataset
import data_utils   as du

class SMILESDataset(Dataset):
    def __init__(self, smiles_list: list[str], seq_length: int, token_to_idx: dict,
                 augmentor=None, augment_train: bool = False):
        if not smiles_list:
            raise ValueError("Empty dataset passed to SMILESDataset.")
        self.smiles = smiles_list
        self.seq_length = seq_length
        self.token_to_idx = token_to_idx
        self.augmentor = augmentor
        self.augment_train = augment_train

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        if self.augment_train and self.augmentor is not None:
            smi = self.augmentor.randomize_smiles(smi)
        idx_seq = du.encode_smiles(smi, self.seq_length, self.token_to_idx)
        seq = torch.tensor(idx_seq, dtype=torch.long)
        return seq, seq.clone()
