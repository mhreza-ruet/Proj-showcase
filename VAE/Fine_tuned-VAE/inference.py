"""Reconstruction helper + quick validity check."""

import torch
import pandas as pd
from rdkit import RDLogger, Chem

import data_utils as du
import metrics as met

RDLogger.DisableLog("rdApp.*")

try:
    def is_valid_smiles(s: str) -> bool:
        return Chem.MolFromSmiles(s) is not None
except Exception:
    def is_valid_smiles(s: str) -> bool:
        return bool(s)

# ────────────────────────────────────────────────────────────────
@torch.no_grad()
def reconstruct_smiles_table( smiles_list=None, test_csv=None, model=None, token_to_idx=None, idx_to_token=None, seq_length=160, pad_idx=0, sos_idx=2, eos_idx=3, device="cpu", mode="beam", beam_size=5):
    # unwrap DataParallel if needed
    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    core.eval()

    # load SMILES from CSV if not provided directly
    if smiles_list is None:
        df = pd.read_csv(test_csv)
        col = "smiles" if "smiles" in df.columns else next(c for c in df.columns if "smile" in c.lower())
        smiles_list = df[col].dropna().astype(str).tolist()

    def _encode(smi):
        toks = du.tokenize_smiles(smi)
        ids  = [token_to_idx.get(t, token_to_idx["<UNK>"]) for t in toks]
        ids  = [sos_idx] + ids[: seq_length - 2] + [eos_idx]
        return ids + [pad_idx] * (seq_length - len(ids))
    
    batch_idx = torch.tensor([_encode(s) for s in smiles_list], device=device)

    if mode == "beam":
        pred = core.beam_search(batch_idx, beam_size=beam_size, max_len=seq_length)
    else:
        pred, _, _ = core(batch_idx, teacher_forcing=False, max_len=seq_length)

    def decode(tokens):
        if tokens.dim() == 3: tokens = tokens.argmax(-1)
        out = []
        for seq in tokens.cpu().tolist():
            if eos_idx in seq: seq = seq[: seq.index(eos_idx)]
            if pad_idx in seq: seq = seq[: seq.index(pad_idx)]
            if seq and seq[0] == sos_idx: seq = seq[1:]
            out.append("".join(idx_to_token.get(i, "") for i in seq))
        return out

    recon = decode(pred)

    valid = [Chem.MolFromSmiles(s) is not None for s in recon]
    levs  = [met.levenshtein_distance(o, r) for o, r in zip(smiles_list, recon)]

    return pd.DataFrame({ "input": smiles_list, "reconstructed": recon, "valid": ["yes" if v else "no" for v in valid], "lev": levs, })


# ────────────────────────────────────────────────────────────────
#  Batch decoding helpers (no RDKit needed)
# ────────────────────────────────────────────────────────────────
def tensor_to_smiles(tensor, idx_to_token, pad_idx, sos_idx=None, eos_idx=None, strip_sos_if_present=False):
    if tensor.dim() == 3:
        tensor = tensor.argmax(dim=-1)
    seqs = tensor.cpu().tolist()
    out = []
    for seq in seqs:
        if strip_sos_if_present and sos_idx is not None and seq and seq[0] == sos_idx:
            seq = seq[1:]
        if eos_idx is not None and eos_idx in seq:
            seq = seq[:seq.index(eos_idx)]
        if pad_idx in seq:
            seq = seq[:seq.index(pad_idx)]
        out.append("".join(idx_to_token.get(i, "") for i in seq))
    return out