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

@torch.no_grad()
def reconstruct_smiles_table(smiles_list: list[str] | None, model, token_to_idx: dict, idx_to_token: dict, seq_length: int, pad_idx: int,
    sos_idx: int, eos_idx: int, device: torch.device | str = "cpu", test_csv: str | None = None, mode: str = "greedy", beam_size: int = 3) -> pd.DataFrame:
    # --------- collect inputs ----------
    inputs = []
    if smiles_list:
        inputs.extend(smiles_list)
    if test_csv:
        df = pd.read_csv(test_csv)
        if "smiles" not in df.columns:
            raise ValueError(f"{test_csv} must contain a 'smiles' column.")
        inputs.extend(df["smiles"].astype(str).tolist())
    if not inputs:
        return pd.DataFrame(columns=["input", "reconstructed", "valid", "lev"])
    enc = [du.encode_smiles(smi, seq_length, token_to_idx) for smi in inputs]
    batch = torch.tensor(enc, dtype=torch.long, device=device)

    # --------- decode ----------
    model.eval()
    if mode == "beam":
        preds = model.beam_search(batch, beam_size=beam_size, max_len=seq_length)
    elif mode == "greedy":
        preds, _, _ = model(batch, teacher_forcing=False, max_len=seq_length)
    else:
        raise ValueError("mode must be 'greedy' or 'beam'")
    recon = tensor_to_smiles(preds, idx_to_token, pad_idx, sos_idx=sos_idx, eos_idx=eos_idx, strip_sos_if_present=True)
    rows = []
    for smi_in, smi_out in zip(inputs, recon):
        valid = "yes" if is_valid_smiles(smi_out) else "no"
        lev = met.levenshtein_distance(smi_in, smi_out) if hasattr(met, "levenshtein_distance") else met.levenshtein_distance(smi_in, smi_out)
        rows.append((smi_in, smi_out, valid, lev))

    return pd.DataFrame(rows, columns=["input", "reconstructed", "valid", "lev"])


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