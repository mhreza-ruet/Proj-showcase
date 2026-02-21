"""
inference.py
Reconstruction helper + quick validity check for TVAE (graph encoder).

Requires:
  - torch_geometric installed
  - data_utils.smiles_to_graph_data(smiles)-> torch_geometric.data.Data (or None)
"""

from __future__ import annotations

import torch
import pandas as pd
from rdkit import RDLogger, Chem
from torch_geometric.data import Batch

import data_utils as du
import metrics as met

RDLogger.DisableLog("rdApp.*")


# -------------------- validity --------------------
def is_valid_smiles(s: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(s)
        return (mol is not None) and (mol.GetNumAtoms() > 0)
    except Exception:
        return False


# -------------------- decoding helper --------------------
def tensor_to_smiles(tensor, idx_to_token, pad_idx, sos_idx=None, eos_idx=None, strip_sos_if_present=False):
    """
    Converts:
      - logits [B,T,V] or token ids [B,T]
    into list[str] SMILES by concatenating tokens.
    """
    if tensor.dim() == 3:
        tensor = tensor.argmax(dim=-1)

    seqs = tensor.detach().cpu().tolist()
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


def _collect_inputs(smiles_list: list[str] | None, test_csv: str | None) -> list[str]:
    inputs: list[str] = []
    if smiles_list:
        inputs.extend([str(s).strip() for s in smiles_list if str(s).strip()])
    if test_csv:
        df = pd.read_csv(test_csv)
        if "smiles" not in df.columns:
            raise ValueError(f"{test_csv} must contain a 'smiles' column.")
        inputs.extend(df["smiles"].astype(str).map(str.strip).tolist())
    return [s for s in inputs if s]


def _make_graph_batch(inputs: list[str], device: torch.device):
    """
    Build PyG Batch from SMILES using du.smiles_to_graph_data.
    Invalid SMILES/graphs are skipped.
    Returns:
      pyg_batch, kept_inputs
    """
    graphs = []
    kept = []

    if not hasattr(du, "smiles_to_graph_data"):
        raise RuntimeError("data_utils.smiles_to_graph_data() not found. Required for TVAE inference.")

    for smi in inputs:
        g = du.smiles_to_graph_data(smi)
        if g is None:
            continue
        graphs.append(g)
        kept.append(smi)

    if not graphs:
        raise ValueError("No valid graphs built from inputs (all SMILES failed graph conversion).")

    pyg_batch = Batch.from_data_list(graphs).to(device)
    return pyg_batch, kept


@torch.no_grad()
def reconstruct_smiles_table(
    smiles_list: list[str] | None,
    model,
    token_to_idx: dict,
    idx_to_token: dict,
    seq_length: int,
    pad_idx: int,
    sos_idx: int,
    eos_idx: int,
    device: torch.device | str | None = None,
    test_csv: str | None = None,
    mode: str = "beam",     # "beam" or "greedy"
    beam_size: int = 5,
) -> pd.DataFrame:
    """
    TVAE inference:
      - Build graph Batch from SMILES
      - Decode with beam_search or greedy
      - Return table with validity + Levenshtein
    """
    inputs = _collect_inputs(smiles_list, test_csv)
    if not inputs:
        return pd.DataFrame(columns=["input", "reconstructed", "valid", "lev"])

    # device
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()

    # ---- Build graph batch (THIS is the key fix) ----
    src_graph_batch, kept_inputs = _make_graph_batch(inputs, device=device)

    # ---- Decode ----
    if mode == "beam":
        m = model.module if isinstance(model, torch.nn.DataParallel) else model
        preds = m.beam_search(src_graph_batch, beam_size=beam_size, max_len=seq_length)
    elif mode == "greedy":
        preds, _, _ = model(src_graph_batch, teacher_forcing=False, max_len=seq_length)
    else:
        raise ValueError("mode must be 'greedy' or 'beam'")

    recon = tensor_to_smiles( preds, idx_to_token, pad_idx, sos_idx=sos_idx, eos_idx=eos_idx, strip_sos_if_present=True )

    rows = []
    for smi_in, smi_out in zip(kept_inputs, recon):
        valid = "yes" if is_valid_smiles(smi_out) else "no"
        lev = met.levenshtein_distance(smi_in, smi_out)
        rows.append((smi_in, smi_out, valid, lev))

    return pd.DataFrame(rows, columns=["input", "reconstructed", "valid", "lev"])