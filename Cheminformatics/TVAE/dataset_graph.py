# dataset_graph.py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from rdkit import Chem

import data_utils as du

class SMILESGraphDataset(Dataset):
    def __init__(self, smiles_list, seq_length, token_to_idx, augmentor=None, augment_train=False):
        if not smiles_list:
            raise ValueError("Empty dataset passed to SMILESGraphDataset.")
        self.smiles = smiles_list
        self.seq_length = seq_length
        self.token_to_idx = token_to_idx
        self.augmentor = augmentor
        self.augment_train = augment_train

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        orig = self.smiles[idx]

        smi_graph = orig
        if self.augment_train and self.augmentor is not None:
            smi_graph = self.augmentor.randomize_smiles(orig)
            m1 = Chem.MolFromSmiles(orig)
            m2 = Chem.MolFromSmiles(smi_graph)
            if (m1 is None) or (m2 is None):
                return None
            if Chem.MolToSmiles(m1, canonical=True) != Chem.MolToSmiles(m2, canonical=True):
                return None

        g = du.smiles_to_graph_data(smi_graph)
        if g is None:
            return None

        mol = Chem.MolFromSmiles(orig)
        if mol is None:
            return None
        smi_tgt = Chem.MolToSmiles(mol, canonical=True)

        tgt_ids = du.encode_smiles(smi_tgt, self.seq_length, self.token_to_idx)
        tgt = torch.tensor(tgt_ids, dtype=torch.long)
        return g, tgt

def collate_graph_smiles(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return Batch(), torch.empty((0, 0), dtype=torch.long)
    graphs, tgts = zip(*batch)
    return Batch.from_data_list(list(graphs)), torch.stack(list(tgts), dim=0)