import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ───────────────────────── Positional Encoding ──────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ───────────────────────────── Bi-directional LSTM Encoder  ────────────────────────────────
class EncoderBiLSTM(nn.Module):
    def __init__(self, vocab_size, d_model, pad_idx, num_layers=1, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        assert d_model % 2 == 0
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.emb_ln = nn.LayerNorm(d_model)
        self.emb_do = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(d_model, d_model//2, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0, bidirectional=True)
        self.out_do = nn.Dropout(dropout)
        self.seq_ln = nn.LayerNorm(d_model)
        self.pool_ln = nn.LayerNorm(d_model)

    def forward(self, src):
        mask = (src != self.pad_idx).float()
        x = self.emb(src)
        x = self.emb_do(self.emb_ln(x))
        seq_out, _ = self.lstm(x)
        seq_out = self.seq_ln(self.out_do(seq_out))
        lengths = mask.sum(1).clamp(min=1)
        pooled = (seq_out * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)
        pooled = self.pool_ln(pooled)
        src_key_padding_mask = (src == self.pad_idx)
        return seq_out, src_key_padding_mask, pooled


# ---------- Transformer Decoder ----------
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, pad_idx: int, dropout: float = 0.1, max_len: int = 512, dim_feedforward: int | None = None):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.emb_ln = nn.LayerNorm(d_model)
        self.pe = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        dff = 4 * d_model if dim_feedforward is None else dim_feedforward
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dff, dropout=dropout, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _causal_mask(L: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)

    def forward(self, tgt_inp: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask: torch.Tensor) -> torch.Tensor:
        tgt_pad = (tgt_inp == self.pad_idx)
        x = self.emb_ln(self.emb(tgt_inp))
        x = self.pe(x)
        y = self.dec(x, memory, tgt_mask=self._causal_mask(tgt_inp.size(1), tgt_inp.device), tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=memory_key_padding_mask)
        y = self.out_ln(y)
        return self.out(y)


    # ---------- Full Model ----------
class LSTM_VAE_Trans(nn.Module):
    def __init__(self, vocab_size, d_model, latent_dim, pad_idx, sos_idx, eos_idx, enc_layers=1, dec_layers=4, nhead=8, dropout=0.1, max_len=512, dim_feedforward: int | None = None):
        super().__init__()
        self.pad_idx, self.sos_idx, self.eos_idx = pad_idx, sos_idx, eos_idx
        self.d_model     = d_model
        self.latent_dim  = latent_dim
        self.encoder = EncoderBiLSTM(vocab_size, d_model, pad_idx, num_layers=enc_layers, dropout=dropout)
        self.to_mu     = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)
        self.latent_to_token = nn.Sequential(nn.Linear(latent_dim, d_model), nn.LayerNorm(d_model))
        self.decoder = TransformerDecoder(vocab_size, d_model, nhead, dec_layers, pad_idx, dropout, max_len, dim_feedforward=dim_feedforward)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    
    def _encode(self, src):
        mem, mem_pad, pooled_h = self.encoder(src)
        mu     = self.to_mu(pooled_h)
        logvar = self.to_logvar(pooled_h)
        z      = self.reparameterize(mu, logvar)
        z_tok  = self.latent_to_token(z).unsqueeze(1)
        memory = z_tok
        mem_pad = torch.zeros(memory.size(0), 1, dtype=torch.bool, device=src.device)
        return memory, mem_pad, mu, logvar

    def _rand_tokens(self, shape, device):
        V = self.decoder.emb.num_embeddings
        t = torch.randint(0, V, shape, device=device)
        for bad in (self.pad_idx, self.sos_idx, self.eos_idx):
            t = torch.where(t == bad, (t + 1) % V, t)
        return t

    def _corrupt_tgt_inputs(self, tgt_inp, p: float):
        if p <= 0: return tgt_inp
        keep = (tgt_inp != self.pad_idx) & (tgt_inp != self.sos_idx) & (tgt_inp != self.eos_idx)
        drop = (torch.rand_like(tgt_inp, dtype=torch.float) < p) & keep
        noise = self._rand_tokens(tgt_inp.shape, tgt_inp.device)
        return torch.where(drop, noise, tgt_inp)

    def forward(self, src, tgt=None, teacher_forcing=True, max_len=128, corruption_p: float = 0.0):
        memory, mem_pad, mu, logvar = self._encode(src)
        if teacher_forcing and tgt is not None:
            tgt_inp = tgt[:, :-1]
            if corruption_p > 0:
                tgt_inp = self._corrupt_tgt_inputs(tgt_inp, corruption_p)
            logits = self.decoder(tgt_inp, memory, mem_pad)           # [B, T-1, V]
            return logits, mu, logvar

        # Greedy decoding (returns tokens without <SOS>)
        B = src.size(0)
        ys = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=src.device)
        out = []
        for _ in range(max_len):
            logits = self.decoder(ys, memory, mem_pad)                # [B, t, V]
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)          # [B, 1]
            ys = torch.cat([ys, nxt], dim=1)
            out.append(nxt)
            if (nxt == self.eos_idx).all():
                break
        gen = torch.cat(out, dim=1) if out else ys[:, 1:]
        return gen, mu, logvar

    @torch.no_grad()
    def beam_search(self, src, beam_size=4, max_len=128, length_penalty=0.8):
        memory, mem_pad, _, _ = self._encode(src)
        B, device = src.size(0), src.device
        memory = memory.unsqueeze(1).repeat(1, beam_size, 1, 1).view(B*beam_size, memory.size(1), memory.size(2))
        mem_pad = mem_pad.unsqueeze(1).repeat(1, beam_size, 1).view(B*beam_size, mem_pad.size(1))
        ys = torch.full((B*beam_size, 1), self.sos_idx, dtype=torch.long, device=device)
        beam_scores = torch.full((B, beam_size), -1e9, device=device); beam_scores[:, 0] = 0.0
        beam_scores = beam_scores.view(-1)
        finished = torch.zeros(B*beam_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits = self.decoder(ys, memory, mem_pad)
            logp = F.log_softmax(logits[:, -1, :], dim=-1)
            if finished.any():
                frozen = torch.full_like(logp, -float('inf'))
                frozen[:, self.eos_idx] = 0.0
                logp = torch.where(finished.unsqueeze(1), frozen, logp)
            cand = (beam_scores.unsqueeze(1) + logp).view(B, beam_size, -1)
            topk_scores, topk_idx = torch.topk(cand.view(B, -1), k=beam_size, dim=-1)
            next_beam = torch.div(topk_idx, logp.size(-1), rounding_mode='floor')
            next_tok  = topk_idx % logp.size(-1)
            base = (torch.arange(B, device=device) * beam_size).unsqueeze(1)
            sel  = (base + next_beam).view(-1)
            ys = torch.cat([ys[sel], next_tok.view(-1, 1)], dim=1)
            beam_scores = topk_scores.view(-1)
            finished = finished[sel] | (next_tok.view(-1) == self.eos_idx)
            if finished.view(B, beam_size).all():
                break
        seqs   = ys.view(B, beam_size, -1)
        scores = beam_scores.view(B, beam_size)
        eos_hits = (seqs == self.eos_idx)
        has_eos  = eos_hits.any(dim=-1)
        first_eos = torch.argmax(eos_hits.to(torch.int32), dim=-1)
        eff_len = torch.where(has_eos, first_eos + 1, seqs.size(2))
        lp = ((5.0 + eff_len.float()) ** length_penalty) / ((5.0 + 1.0) ** length_penalty)
        norm = scores / lp
        pref = torch.where(has_eos, norm, norm - 1e6)
        best_idx = pref.argmax(dim=1)
        best = seqs[torch.arange(B, device=device), best_idx]
        out = best[:, 1:]
        assert (out[:, 0] != self.sos_idx).all(), "beam_search should not return <SOS> as first token"
        return out