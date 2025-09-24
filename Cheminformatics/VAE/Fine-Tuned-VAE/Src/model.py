# /home/md_halim_mondol/LSTM_VAE_Paper/model.py

import torch
from .model_bs import CNNCharVAE   # import your actual model class

def build_model(token_to_idx: dict, latent_dim: int, device: str = "cuda") -> torch.nn.Module:
    """
    Factory used by chemselect/latent.py to instantiate your CNNCharVAE model.
    """
    vocab_size = max(token_to_idx.values()) + 1

    # Special token indices (try common names, fall back to 0 if missing)
    pad_idx = token_to_idx.get("<pad>", 0)
    sos_idx = token_to_idx.get("<bos>", token_to_idx.get("<sos>", None))
    eos_idx = token_to_idx.get("<eos>", None)

    # === IMPORTANT ===
    # Fill in the SAME hyperparameters you used during training.
    # If you had a cfg dict, copy those values here.
    model = CNNCharVAE(
        vocab_size=vocab_size,
        d_model=256,
        latent_dim=latent_dim,
        pad_idx=pad_idx,
        sos_idx=sos_idx,
        eos_idx=eos_idx,
        enc_layers=3,
        dec_layers=7,
        dropout=0.05,
        emb_dropout=0.05,
        max_len=160)

    model.to(device)
    return model