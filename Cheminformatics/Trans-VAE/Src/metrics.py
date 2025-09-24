"""Reconstruction accuracy + VAE loss."""
import torch
import torch.nn as nn
from rdkit import Chem
import torch.nn.functional as F

def compute_accuracy_counts(pred_logits_or_tokens, target, padding_idx=0, eos_idx=None, stop_at_eos=True):
    if pred_logits_or_tokens.dim() == 3:
        pred = pred_logits_or_tokens.argmax(dim=-1)
    elif pred_logits_or_tokens.dim() == 2:
        pred = pred_logits_or_tokens
    else:
        raise ValueError("pred_logits_or_tokens must be [B,T,V] or [B,T]")
    tgt = target[:, 1:]
    T = min(pred.size(1), tgt.size(1))
    pred, tgt = pred[:, :T], tgt[:, :T]
    mask = (tgt != padding_idx)
    if eos_idx is not None:
        mask &= (tgt != eos_idx)
    if stop_at_eos and eos_idx is not None:
        eos_hits = (tgt == eos_idx)
        has_eos = eos_hits.any(dim=1)
        first_eos = torch.argmax(eos_hits.to(torch.int32), dim=1)
        lengths = torch.where(has_eos, first_eos, torch.full_like(first_eos, T))
        ar = torch.arange(T, device=tgt.device).unsqueeze(0)
        mask &= (ar < lengths.unsqueeze(1))
    total = int(mask.sum().item())
    correct = int((pred[mask] == tgt[mask]).sum().item())
    return correct, total

def compute_accuracy(pred_logits_or_tokens, target, padding_idx=0, eos_idx=None, stop_at_eos=True):
    correct, total = compute_accuracy_counts(pred_logits_or_tokens, target, padding_idx, eos_idx, stop_at_eos)
    return (correct / total) if total > 0 else 0.0


def vae_loss(pred_logits, target, mu, logvar, padding_idx=0, eos_idx=None, kl_weight=1.0, label_smoothing: float = 0.0):
    tgt = target[:, 1:]
    T = min(pred_logits.size(1), tgt.size(1))
    pred = pred_logits[:, :T]
    tgt = tgt[:, :T]
    mask = (tgt != padding_idx)
    if eos_idx is not None:
        mask &= (tgt != eos_idx)

    if mask.any():
        logp = F.log_softmax(pred, dim=-1)                            # [B,T,V]
        gold_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)      # [B,T]
        if label_smoothing > 0.0:
            V = logp.size(-1)
            uni_lp = logp.mean(dim=-1)                                # [B,T]
            nll_tok = -(1.0 - label_smoothing) * gold_lp - label_smoothing * uni_lp
        else:
            nll_tok = -gold_lp
        recon = nll_tok[mask].mean()
    else:
        recon = pred.new_tensor(0.0)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl_weight * kl, recon, kl


# ───────────────────────── chemical validity ─────────────────────────
def chemical_validity_ratio(recon_smiles: list[str]) -> float:
    valid = sum(Chem.MolFromSmiles(smi) is not None for smi in recon_smiles)
    return valid / len(recon_smiles)

# ───────────────────────── Levenshtein distance ──────────────────────
def levenshtein_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = range(len(b)+1)
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert_cost  = current[j-1] + 1
            delete_cost  = previous[j] + 1
            replace_cost = previous[j-1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]

def average_levenshtein(orig_list: list[str], recon_list: list[str]) -> float:
    dists = [levenshtein_distance(o, r) for o, r in zip(orig_list, recon_list)]
    return sum(dists) / len(dists)