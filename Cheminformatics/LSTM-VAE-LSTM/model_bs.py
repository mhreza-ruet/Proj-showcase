import torch
import torch.nn as nn
import torch.nn.functional as F

# ───────────────────────────── Bi-directional LSTM Encoder ────────────────────────────────
class EncoderBiLSTM(nn.Module):
    def __init__(self, vocab_size, d_model, pad_idx, num_layers=1, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        assert d_model % 2 == 0
        self.pad_idx = pad_idx

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.emb_ln = nn.LayerNorm(d_model)
        self.emb_do = nn.Dropout(emb_dropout)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        self.out_do = nn.Dropout(dropout)
        self.seq_ln = nn.LayerNorm(d_model)
        self.pool_ln = nn.LayerNorm(d_model)

    def forward(self, src):
        # src: [B, S]
        mask = (src != self.pad_idx).float()          # [B, S]
        x = self.emb(src)                             # [B, S, d_model]
        x = self.emb_do(self.emb_ln(x))
        seq_out, _ = self.lstm(x)                     # [B, S, d_model]
        seq_out = self.seq_ln(self.out_do(seq_out))

        lengths = mask.sum(1).clamp(min=1)            # [B]
        pooled = (seq_out * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)  # [B, d_model]
        pooled = self.pool_ln(pooled)

        return seq_out, pooled # seq_out not used in latent-only decoder ablation


# ───────────────────────────── LSTM Decoder (teacher forcing + greedy) ─────────────────────────────
class DecoderLSTM(nn.Module):
    """
    Auto-regressive LSTM decoder conditioned on VAE latent z via (h0, c0) initialization.

    Notes:
    - Uses causal generation naturally via recurrence (no positional encoding needed).
    - Token corruption uses decoder vocab size via self.emb.num_embeddings.
    """
    def __init__(self, vocab_size, d_model, pad_idx, num_layers=4, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.num_layers = num_layers
        self.d_model = d_model

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.emb_ln = nn.LayerNorm(d_model)
        self.emb_do = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM( input_size=d_model, hidden_size=d_model, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0 )
        self.out_ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_inp, h0, c0):
        # tgt_inp: [B, T]
        x = self.emb_ln(self.emb(tgt_inp))            # [B, T, d_model]
        x = self.emb_do(x)
        y, _ = self.lstm(x, (h0, c0))                 # [B, T, d_model]
        y = self.out_ln(y)
        return self.out(y)                            # [B, T, V]

    @torch.no_grad()
    def step(self, last_token, h, c):
        """
        last_token: [B, 1]
        h,c: [L, B, d_model]
        returns logits_last: [B, V], new h,c
        """
        x = self.emb_ln(self.emb(last_token))         # [B, 1, d_model]
        x = self.emb_do(x)
        y, (h, c) = self.lstm(x, (h, c))              # y: [B, 1, d_model]
        y = self.out_ln(y)
        logits_last = self.out(y[:, -1, :])           # [B, V]
        return logits_last, h, c


# ───────────────────────────── BiLSTM–VAE–BiLSTM (Decoder Ablation) ─────────────────────────────
class LSTM_VAE_LSTM(nn.Module):
    """
    BiLSTM encoder + VAE bottleneck + LSTM decoder.

    - Training/validation: teacher forcing always on (per your setup)
    - Token corruption (decoder input) supported via corruption_p
    - Greedy decoding for inference when teacher_forcing=False
    """
    def __init__( self, vocab_size, d_model, latent_dim, pad_idx, sos_idx, eos_idx, enc_layers=7, dec_layers=7, dropout=0.05, emb_dropout=0.05, max_len=160 ):
        super().__init__()
        self.pad_idx, self.sos_idx, self.eos_idx = pad_idx, sos_idx, eos_idx
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.dec_layers = dec_layers

        self.encoder = EncoderBiLSTM(vocab_size, d_model, pad_idx, num_layers=enc_layers, dropout=dropout, emb_dropout=emb_dropout)

        # VAE parameters from pooled encoder state
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

        # z -> initial decoder states (h0, c0)
        self.z_to_h0 = nn.Linear(latent_dim, dec_layers * d_model)
        self.z_to_c0 = nn.Linear(latent_dim, dec_layers * d_model)
        self.h0_ln = nn.LayerNorm(d_model)
        self.c0_ln = nn.LayerNorm(d_model)

        self.decoder = DecoderLSTM(vocab_size, d_model, pad_idx, num_layers=dec_layers, dropout=dropout, emb_dropout=emb_dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def _z_to_init_states(self, z):
        # z: [B, latent_dim] -> h0,c0: [L, B, d_model]
        B = z.size(0)
        L = self.dec_layers
        d = self.d_model

        h0 = self.z_to_h0(z).view(B, L, d).transpose(0, 1).contiguous()
        c0 = self.z_to_c0(z).view(B, L, d).transpose(0, 1).contiguous()

        h0 = self.h0_ln(h0)
        c0 = self.c0_ln(c0)
        return h0, c0

    def _encode(self, src):
        _, pooled = self.encoder(src)                 # pooled: [B, d_model]
        mu = self.to_mu(pooled)                       # [B, latent_dim]
        logvar = self.to_logvar(pooled)               # [B, latent_dim]
        z = self.reparameterize(mu, logvar)           # [B, latent_dim]
        return z, mu, logvar

    def _rand_tokens(self, shape, device):
        V = self.decoder.emb.num_embeddings
        t = torch.randint(0, V, shape, device=device)
        for bad in (self.pad_idx, self.sos_idx, self.eos_idx):
            t = torch.where(t == bad, (t + 1) % V, t)
        return t

    def _corrupt_tgt_inputs(self, tgt_inp, p: float):
        if p <= 0.0:
            return tgt_inp
        keep = (tgt_inp != self.pad_idx) & (tgt_inp != self.sos_idx) & (tgt_inp != self.eos_idx)
        drop = (torch.rand_like(tgt_inp, dtype=torch.float) < p) & keep
        noise = self._rand_tokens(tgt_inp.shape, tgt_inp.device)
        return torch.where(drop, noise, tgt_inp)

    def forward(self, src, tgt=None, teacher_forcing=True, max_len=128, corruption_p: float = 0.0):
        """
        If teacher_forcing=True and tgt is provided:
            returns logits [B, T-1, V], mu, logvar
        Else:
            greedy-decodes and returns tokens [B, <=max_len] (without SOS), mu, logvar
        """
        z, mu, logvar = self._encode(src)
        h0, c0 = self._z_to_init_states(z)

        # ── Teacher forcing path ──
        if teacher_forcing and tgt is not None:
            tgt_inp = tgt[:, :-1]                     # [B, T-1]
            if corruption_p > 0.0:
                tgt_inp = self._corrupt_tgt_inputs(tgt_inp, corruption_p)
            logits = self.decoder(tgt_inp, h0, c0)    # [B, T-1, V]
            return logits, mu, logvar

        # ── Greedy decoding path ──
        B = src.size(0)
        device = src.device
        ys = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=device)
        out = []

        h, c = h0, c0
        for _ in range(max_len):
            logits_last, h, c = self.decoder.step(ys[:, -1:], h, c)       # [B, V]
            nxt = logits_last.argmax(dim=-1, keepdim=True)                # [B, 1]
            ys = torch.cat([ys, nxt], dim=1)
            out.append(nxt)
            if (nxt == self.eos_idx).all():
                break

        gen = torch.cat(out, dim=1) if out else ys[:, 1:]
        return gen, mu, logvar