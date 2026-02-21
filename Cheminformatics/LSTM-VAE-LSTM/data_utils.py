"""Data augmentation, tokenisation, vocabulary, encoding and sampling utilities."""

import os
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# ───────────────────────────────────────────────
#  SMILES augmentation
# ───────────────────────────────────────────────
class SmilesEnumerator:
    """Randomises atom order to create non‑canonical SMILES strings."""

    def __init__(self):
        pass

    def randomize_smiles(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        atoms = list(mol.GetAtoms())
        np.random.shuffle(atoms)
        shuffled_mol = Chem.RenumberAtoms(mol, [atom.GetIdx() for atom in atoms])
        randomized_smiles = Chem.MolToSmiles(shuffled_mol, canonical=False)

        if Chem.MolFromSmiles(randomized_smiles) is not None:
            return randomized_smiles
        else:
            return smiles


# ───────────────────────────────────────────────
#  Tokenisation helpers
# ───────────────────────────────────────────────

MULTI_TOKENS = [
    "[N+]", "[S]", "[N-]", "[B-]", "[o+]", "[NH2+]", "[NH+]", "[n+]", "[nH]", "[nH+]", "[Zn]", "[Na+]", "[Ca2+]", "[Cr+3]", "[Cd+2]", "[Co+2]", "[Pb+2]", "[O-]", "[Cl-]", "(C)", "[C+]", "(O)", "(=O)", "(Cl)", "(F)", "(Br)", "(I)",
    "Br", "Cl", "Si", "Na", "Li", "Ca", "Mg",
    "[C@H]", "[C@@H]"]

MULTI_TOKENS.sort(key=len, reverse=True)
pattern = re.compile("|".join(sorted(map(re.escape, MULTI_TOKENS), key=len, reverse=True)))

def tokenize_smiles(smiles: str):
    tokens, i, n = [], 0, len(smiles)
    while i < n:
        match = pattern.match(smiles, i)
        if match:
            tok = match.group(0)
            tokens.append(tok)
            i += len(tok)
        else:
            tokens.append(smiles[i])
            i += 1
    return tokens



import pandas as pd

# ────────────────────────────────────────────────────────────────
def create_vocabulary(file_paths, test_smiles=None, normalizer=None):
    """
    Build token ↔ index dicts from CSVs.
    Accepts either 'smiles' OR ('Analyte_SMILES','Dye_SMILES') per file.
    - file_paths: list[str] of CSV paths
    - test_smiles: optional list[str] to guarantee coverage
    - normalizer: optional callable(smiles:str)->str for pre-tokenization cleanup
    """
    if test_smiles is None:
        test_smiles = []

    special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
    token_to_idx = {tok: idx for idx, tok in enumerate(special_tokens)}
    idx_to_token = {idx: tok for tok, idx in token_to_idx.items()}

    def _yield_smiles_from_df(df: pd.DataFrame):
        cols = []
        if "smiles" in df.columns:
            cols.append("smiles")
        # Accept VOC/Dye pair columns too
        if "Analyte_SMILES" in df.columns:
            cols.append("Analyte_SMILES")
        if "Dye_SMILES" in df.columns:
            cols.append("Dye_SMILES")
        if not cols:
            return  # nothing to yield from this file

        series = pd.concat([df[c] for c in cols], ignore_index=True)
        for s in series.dropna().astype(str):
            s = s.strip()
            if not s:
                continue
            if normalizer is not None:
                s = normalizer(s)
            yield s

    # -------- collect from CSV files --------
    for fp in file_paths:
        df = pd.read_csv(fp)
        seen = set()
        for smi in _yield_smiles_from_df(df):
            if smi in seen:
                continue
            seen.add(smi)
            for tok in tokenize_smiles(smi):
                if tok not in token_to_idx:
                    idx = len(token_to_idx)
                    token_to_idx[tok] = idx
                    idx_to_token[idx] = tok

    # -------- ensure test coverage --------
    for smi in test_smiles:
        if normalizer is not None:
            smi = normalizer(smi)
        for tok in tokenize_smiles(smi):
            if tok not in token_to_idx:
                idx = len(token_to_idx)
                token_to_idx[tok] = idx
                idx_to_token[idx] = tok

    # sanity: special tokens are fixed
    assert token_to_idx["<PAD>"] == 0
    assert token_to_idx["<UNK>"] == 1
    assert token_to_idx["<SOS>"] == 2
    assert token_to_idx["<EOS>"] == 3

    return token_to_idx, idx_to_token


# ────────────────────────────────────────────────────────────────
#  One‑time build / cached load for the vocabulary
# ────────────────────────────────────────────────────────────────

def load_or_create_vocabulary(csv_paths, cache_path="vocab.json", test_smiles=None):

    # ---------- try to load ----------
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            token_to_idx = json.load(f)
        # json stores keys as str → convert values back to int
        token_to_idx = {k: int(v) for k, v in token_to_idx.items()}
        idx_to_token = {v: k for k, v in token_to_idx.items()}
        print(f"[vocab] loaded cached vocabulary from {cache_path} "
              f"({len(token_to_idx)} tokens)")
        return token_to_idx, idx_to_token

    # ---------- build from scratch ----------
    token_to_idx, idx_to_token = create_vocabulary(csv_paths, test_smiles=test_smiles)

    with open(cache_path, "w") as f:
        json.dump(token_to_idx, f)
    print(f"[vocab] built and saved new vocabulary to {cache_path}"
          f"({len(token_to_idx)} tokens)")
    return token_to_idx, idx_to_token

#-----------------------------------------------------------------------

def encode_smiles(smiles: str, seq_length: int, token_to_idx: dict):
    """Add <SOS>/<EOS>, lookup indices, then pad/truncate to *seq_length*."""
    tokens = ["<SOS>"] + tokenize_smiles(smiles) + ["<EOS>"]
    idx_seq = [token_to_idx.get(tok, token_to_idx["<UNK>"]) for tok in tokens]

    if len(idx_seq) > seq_length:
        idx_seq = idx_seq[:seq_length]
    else:
        idx_seq += [token_to_idx["<PAD>"]] * (seq_length - len(idx_seq))

    return idx_seq


# ───────────────────────────────────────────────
#  Sampling helper (loads CSVs, optional augmentation)
# ───────────────────────────────────────────────

def sample_data(file_paths, n_samples: int, seq_length: int, token_to_idx: dict,
                augmentor: SmilesEnumerator | None = None, augment_train: bool = True):
    """Return a list[ list[int] ] of encoded SMILES indices."""
    data_frames = [pd.read_csv(fp) for fp in file_paths]
    total_available = sum(len(df) for df in data_frames)
    if n_samples > total_available:
        print(f"Requested {n_samples} samples, but only {total_available} available; using all.")
        n_samples = total_available

    sampled = []
    for df in data_frames:
        need = n_samples - len(sampled)
        if need <= 0:
            break
        df_sampled = df.sample(n=min(len(df), need), random_state=42)
        for smi in df_sampled["smiles"]:
            try:
                smi_aug = augmentor.randomize_smiles(smi) if augmentor and augment_train else smi
                sampled.append(encode_smiles(smi_aug, seq_length, token_to_idx))
                if len(sampled) >= n_samples:
                    return sampled
            except Exception as exc:
                print(f"Error processing {smi}: {exc}")
    return sampled


#-------------------------------- Data Loader --------------------------------

def load_smiles_list(file_paths: list[str], n_samples: int, shuffle: bool) -> list[str]:
    """Collect up to n_samples SMILES strings from CSV files (column 'smiles')."""
    smiles: list[str] = []
    for fp in file_paths:
        df = pd.read_csv(fp)
        if shuffle:
            df = df.sample(frac=1.0, random_state=42)
        for smi in df["smiles"]:
            smiles.append(smi)
            if len(smiles) >= n_samples:
                return smiles
    return smiles


#--------------------- SMILES Canonicalization ----------------------------

def canonicalize_smiles(smiles_list):
    """
    Returns a list of canonical SMILES.
    Invalid strings are skipped and reported.
    """
    canonical = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"[warning] could not parse: {smi}")
            continue
        can_smi = Chem.MolToSmiles(mol, canonical=True)
        canonical.append(can_smi)
    return canonical


#------------------------ Model parameter counting -------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)