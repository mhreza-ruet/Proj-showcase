# train.py  (DataParallel, streamlined)

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from inference import tensor_to_smiles
import data_utils   as du
import dataset      as ds
import model_bs     as mdl
import metrics      as met

# ────────────────────────── dataloader builder ─────────────────────
def build_dataloaders(train_files, val_files, seq_length, batch_size, n_train, n_val, token_to_idx):
    augmentor = du.SmilesEnumerator()

    train_smiles = du.load_smiles_list(train_files, n_samples=n_train, shuffle=True)
    val_smiles   = du.load_smiles_list(val_files,   n_samples=n_val,   shuffle=False)

    # on-the-fly: train augments, val does not
    train_ds = ds.SMILESDataset(train_smiles, seq_length=seq_length, token_to_idx=token_to_idx, augmentor=augmentor, augment_train=True)
    val_ds   = ds.SMILESDataset(val_smiles, seq_length=seq_length, token_to_idx=token_to_idx, augmentor=None, augment_train=False)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=8, pin_memory=True, prefetch_factor=2)
    val_dl   = DataLoader(val_ds,   batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)
    return train_dl, val_dl

# ────────────────────────── train one epoch (no-TF) ───────────────────────
def train_one_epoch(model, loader, optimizer, scaler, pad_idx, eos_idx, kl_w,
                    device, corruption_p, label_smoothing, clip_grad, use_amp):
    model.train()
    total_loss = 0.0
    total_corr = 0
    total_tok  = 0

    for inp, tgt in loader:
        inp, tgt = inp.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, mu, logvar = model(inp, tgt=tgt, teacher_forcing=True)
            loss, _, _ = met.vae_loss( logits, tgt, mu, logvar, padding_idx=pad_idx, eos_idx=None, kl_weight=kl_w, label_smoothing=label_smoothing )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        corr, tok = met.compute_accuracy_counts( logits, tgt, padding_idx=pad_idx, eos_idx=eos_idx, stop_at_eos=True )
        total_corr += corr
        total_tok  += tok

    avg_loss = total_loss / max(1, len(loader))
    acc = (total_corr / total_tok) if total_tok > 0 else 0.0
    return avg_loss, acc


# ────────────────────────── validation (no-TF) ────────────────────────────
@torch.no_grad()
def validate_tf(model, loader, pad_idx, eos_idx, kl_w, device, use_amp, collect_strings: bool = False, idx_to_token=None, sos_idx=None):

    model.eval()
    total_loss = 0.0
    total_corr = 0
    total_tok  = 0

    want_strings = bool(collect_strings and (idx_to_token is not None))
    recon_smiles = [] if want_strings else None
    orig_smiles  = [] if want_strings else None

    for inp, tgt in loader:
        inp, tgt = inp.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, mu, logvar = model(inp, tgt=tgt, teacher_forcing=True)
            loss, _, _ = met.vae_loss( logits, tgt, mu, logvar, padding_idx=pad_idx, eos_idx=None, kl_weight=kl_w, label_smoothing=0.0 )

        total_loss += loss.item()
        corr, tok = met.compute_accuracy_counts( logits, tgt, padding_idx=pad_idx, eos_idx=eos_idx, stop_at_eos=True )
        total_corr += corr
        total_tok  += tok

        if want_strings:
            pred_tokens = logits.argmax(dim=-1)
            recon_smiles.extend(tensor_to_smiles(pred_tokens, idx_to_token, pad_idx, sos_idx=sos_idx, eos_idx=eos_idx, strip_sos_if_present=True))
            orig_smiles.extend(tensor_to_smiles(tgt, idx_to_token, pad_idx, sos_idx=sos_idx, eos_idx=eos_idx, strip_sos_if_present=True))

    avg_loss = total_loss / max(1, len(loader))
    acc = (total_corr / total_tok) if total_tok > 0 else 0.0

    if want_strings:
        return avg_loss, acc, recon_smiles, orig_smiles
    else:
        return avg_loss, acc

# ----------------------- validation (beam metrics) -------------------------
@torch.no_grad()
def validate_beam_metrics(model, loader, pad_idx, idx_to_token, device, beam_size, max_len, sos_idx, eos_idx):
    model.eval()
    # unwrap DataParallel for custom method calls
    m = model.module if isinstance(model, nn.DataParallel) else model

    orig_smiles, recon_smiles = [], []
    tot_corr = tot_tok = 0
    for inp, tgt in loader:
        inp, tgt = inp.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        beams = m.beam_search(inp, beam_size=beam_size, max_len=max_len, length_penalty=0.6)
        c, t = met.compute_accuracy_counts(beams, tgt, padding_idx=pad_idx, eos_idx=eos_idx, stop_at_eos=True)
        tot_corr += c; tot_tok += t
        recon_smiles.extend(tensor_to_smiles(beams, idx_to_token, pad_idx, sos_idx=sos_idx, eos_idx=eos_idx, strip_sos_if_present=True))
        orig_smiles.extend(tensor_to_smiles(tgt,   idx_to_token, pad_idx, sos_idx=sos_idx, eos_idx=eos_idx, strip_sos_if_present=True))
    valid_ratio = met.chemical_validity_ratio(recon_smiles)
    lev = met.average_levenshtein(orig_smiles, recon_smiles)
    beam_acc = (tot_corr / tot_tok) if tot_tok > 0 else 0.0
    return valid_ratio, lev, beam_acc

# ----------------------------- main training -------------------------------
def run_training(cfg, token_to_idx, idx_to_token):
    required = [
        "train_files","val_files","seq_length","n_train","n_val",
        "d_model","latent_dim","dec_layers","enc_layers","dropout",
        "pad_idx","sos_idx","eos_idx",
        "batch","lr","epochs","early_stop",
        "kl_anneal","kl_max",
        "save_dir","metrics_every","beam_size",
        "label_smoothing","corruption_p","clip_grad"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required cfg keys: {missing}")

    train_dl, val_dl = build_dataloaders( cfg["train_files"], cfg["val_files"], cfg["seq_length"], cfg["batch"], cfg["n_train"], cfg["n_val"], token_to_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print("Visible CUDA devices:", torch.cuda.device_count())

    base_model = mdl.CNNCharVAE(
        vocab_size=len(token_to_idx),
        d_model=cfg["d_model"],
        latent_dim=cfg["latent_dim"],
        pad_idx=cfg["pad_idx"],
        sos_idx=cfg["sos_idx"],
        eos_idx=cfg["eos_idx"],
        enc_layers=cfg.get("enc_layers"),
        dec_layers=cfg.get("dec_layers"),
        dropout=cfg.get("dropout"),
        emb_dropout=cfg.get("emb_dropout"),
        max_len=cfg["seq_length"],
    ).to(device)
    model = nn.DataParallel(base_model) if torch.cuda.device_count() > 1 else base_model

    opt   = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay"))
    use_amp = torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    sched = None
    if "plateau" in cfg:
        p = cfg["plateau"]
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=p["mode"], factor=p["factor"], patience=p["patience"], verbose=p["verbose"],
            min_lr=p["min_lr"], threshold=p["threshold"], threshold_mode=p.get("threshold_mode", "rel"),
            cooldown=p.get("cooldown", 0), eps=p.get("eps", 1e-8))

    def kl_weight(ep):
        warm = min(ep / cfg["kl_anneal"], 1.0) * cfg["kl_max"]
        plateau = cfg.get("kl_plateau_until", 15)
        return min(warm, 0.02) if ep <= plateau else warm

    best_val_loss, no_improve = float("inf"), 0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss_tf": [], "val_acc_tf": [], "val_acc_beam": [],
        "val_valid": [], "val_lev": []}
    os.makedirs(cfg["save_dir"], exist_ok=True)
    start = time.time()
    
    freq = max(1, int(cfg.get("metrics_every", 5)))

    for ep in range(1, cfg["epochs"] + 1):
        kl_w = kl_weight(ep)
        tr_loss, tr_acc = train_one_epoch( model, train_dl, opt, scaler, pad_idx=cfg["pad_idx"], eos_idx=cfg["eos_idx"], kl_w=kl_w, device=device, 
                                          corruption_p=cfg["corruption_p"], label_smoothing=cfg["label_smoothing"], clip_grad=cfg["clip_grad"], use_amp=use_amp )

        va_loss, va_acc = validate_tf( model, val_dl, pad_idx=cfg["pad_idx"], eos_idx=cfg["eos_idx"], kl_w=kl_w,
                                                                        device=device, use_amp=use_amp, collect_strings=False, idx_to_token=idx_to_token, sos_idx=cfg["sos_idx"] )
        if (ep % freq) == 0:
            va_valid, va_lev, va_acc_beam = validate_beam_metrics( model, val_dl, pad_idx=cfg["pad_idx"], idx_to_token=idx_to_token, device=device, beam_size=cfg["beam_size"], max_len=cfg["seq_length"], sos_idx=cfg["sos_idx"], eos_idx=cfg["eos_idx"] )
            print(f"Epoch {ep:2d}: train {tr_loss:.4f}/{tr_acc:.3f}  "
                  f"val(tf) {va_loss:.4f}/{va_acc:.3f}  "
                  f"beam_acc {va_acc_beam:.3f}  valid {va_valid:.3f}  lev {va_lev:.2f}  "
                  f"KL {kl_w:.2f}")
        else:
            va_valid = va_lev = va_acc_beam = np.nan
            print(f"Epoch {ep:2d}: train {tr_loss:.4f}/{tr_acc:.3f}  "
                  f"val(tf) {va_loss:.4f}/{va_acc:.3f}  KL {kl_w:.2f}")

        # ---- log ----
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss_tf"].append(va_loss); history["val_acc_tf"].append(va_acc)
        history["val_valid"].append(va_valid);  history["val_lev"].append(va_lev)
        history["val_acc_beam"].append(va_acc_beam)

        if sched is not None:
            sched.step(va_loss)

        min_delta = cfg.get("early_stop_min_delta", 0.0)
        if best_val_loss - va_loss > min_delta:
            best_val_loss, no_improve = va_loss, 0
            to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(to_save, os.path.join(cfg["save_dir"], "best_model.pth"))
        else:
            no_improve += 1
            if no_improve >= cfg["early_stop"]:
                print("Early stopping."); break

        if "save_every" in cfg and (ep % cfg["save_every"] == 0):
            to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(to_save, os.path.join(cfg["save_dir"], f"model_epoch_{ep}.pth"))

    print(f"Total training time: {(time.time()-start)/60:.2f} min")
    return model, history