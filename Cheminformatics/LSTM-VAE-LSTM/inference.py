"""Reconstruction helper + quick validity check (GREEDY ONLY)."""

import torch
import pandas as pd
from rdkit import RDLogger, Chem

import data_utils as du
import metrics as met

RDLogger.DisableLog("rdApp.*")


def is_valid_smiles(s: str) -> bool:
    try:
        return Chem.MolFromSmiles(s) is not None
    except Exception:
        return bool(s)


@torch.no_grad()
def reconstruct_smiles_table(
    smiles_list=None,
    test_csv=None,
    model=None,
    token_to_idx=None,
    idx_to_token=None,
    seq_length=160,
    pad_idx=0,
    sos_idx=2,
    eos_idx=3,
    device="cpu",
    max_len = None
):
    if model is None:
        raise ValueError("model must be provided.")
    if token_to_idx is None or idx_to_token is None:
        raise ValueError("token_to_idx and idx_to_token must be provided.")
    if smiles_list is None and test_csv is None:
        raise ValueError("Provide smiles_list or test_csv.")

    # unwrap DataParallel if needed
    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    core.eval()

    # load SMILES from CSV if not provided directly
    if smiles_list is None:
        df = pd.read_csv(test_csv)
        col = "smiles" if "smiles" in df.columns else next(c for c in df.columns if "smile" in c.lower())
        smiles_list = df[col].dropna().astype(str).tolist()

    unk_idx = token_to_idx.get("<UNK>", None)

    def _encode_one(smi: str):
        toks = du.tokenize_smiles(smi)
        if unk_idx is None:
            ids = [token_to_idx[t] for t in toks]
        else:
            ids = [token_to_idx.get(t, unk_idx) for t in toks]

        ids = [sos_idx] + ids[: seq_length - 2] + [eos_idx]
        if len(ids) < seq_length:
            ids = ids + [pad_idx] * (seq_length - len(ids))
        else:
            ids = ids[:seq_length]
        return ids

    batch_idx = torch.tensor([_encode_one(s) for s in smiles_list], device=device)

    # ---- GREEDY decode (no beam) ----
    decode_len = seq_length if max_len is None else int(max_len)
    pred, _, _ = core(batch_idx, teacher_forcing=False, max_len=decode_len)

    def _decode(tokens: torch.Tensor):
        if tokens.dim() == 3:
            tokens = tokens.argmax(-1)

        out = []
        for seq in tokens.cpu().tolist():
            if seq and seq[0] == sos_idx:
                seq = seq[1:]
            if eos_idx in seq:
                seq = seq[: seq.index(eos_idx)]
            if pad_idx in seq:
                seq = seq[: seq.index(pad_idx)]
            out.append("".join(idx_to_token.get(i, "") for i in seq))
        return out

    recon = _decode(pred)

    valid = [is_valid_smiles(s) for s in recon]
    levs = [met.levenshtein_distance(o, r) for o, r in zip(smiles_list, recon)]

    return pd.DataFrame({
        "input": smiles_list,
        "reconstructed": recon,
        "valid": ["yes" if v else "no" for v in valid],
        "lev": levs,
    })


# ────────────────────────────────────────────────────────────────
#  Batch decoding helpers (no RDKit needed)
# ────────────────────────────────────────────────────────────────
def tensor_to_smiles(
    tensor,
    idx_to_token,
    pad_idx,
    sos_idx=None,
    eos_idx=None,
    strip_sos_if_present=False,
):
    """
    Generic tensor->SMILES helper (greedy/beam/logits-safe).
    """
    if tensor.dim() == 3:
        tensor = tensor.argmax(dim=-1)

    seqs = tensor.cpu().tolist()
    out = []
    for seq in seqs:
        if strip_sos_if_present and sos_idx is not None and seq and seq[0] == sos_idx:
            seq = seq[1:]
        if eos_idx is not None and eos_idx in seq:
            seq = seq[: seq.index(eos_idx)]
        if pad_idx in seq:
            seq = seq[: seq.index(pad_idx)]
        out.append("".join(idx_to_token.get(i, "") for i in seq))
    return out