# model_cnn_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- Encoder (CNN) -----------------------------
class EncoderCNN(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 latent_dim: int,
                 pad_idx: int,
                 enc_layers: int | None = None,
                 conv_channels: tuple[int, ...] | None = None,
                 conv_kernels:  tuple[int, ...] | None = None,
                 conv_dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 proj_dim: int = 196):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.emb_ln = nn.LayerNorm(d_model)
        self.emb_do = nn.Dropout(emb_dropout)

        # --- build conv spec ---
        if conv_channels is None or conv_kernels is None:
            if enc_layers is None:
                enc_layers = 3
            ks = [9, 9, 10] if enc_layers == 3 else ([9] * max(1, enc_layers - 1) + [10])
            ch = [256] * enc_layers
        else:
            assert len(conv_channels) == len(conv_kernels)
            ch = list(conv_channels)
            ks = list(conv_kernels)

        # --- conv stack with SAME-length padding for any k ---
        layers = []
        in_ch = d_model
        for out_ch, k in zip(ch, ks):
            left = (k - 1) // 2
            right = k - 1 - left
            layers += [ nn.ConstantPad1d((left, right), 0.0), nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=0, bias=True), nn.GELU(), nn.Dropout(conv_dropout)]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.out_ch = in_ch

        self.proj = nn.Sequential(nn.Linear(self.out_ch, proj_dim), nn.ReLU())
        self.to_mu     = nn.Linear(proj_dim, latent_dim)
        self.to_logvar = nn.Linear(proj_dim, latent_dim)

    def forward(self, src: torch.Tensor):
        # src: [B, T]
        pad_mask   = (src == self.pad_idx)         # [B,T]
        token_mask = (~pad_mask).float()           # [B,T]

        x = self.emb(src)                          # [B,T,D]
        x = self.emb_do(self.emb_ln(x))
        x = x.transpose(1, 2)                      # [B,D,T] for Conv1d

        h = self.conv(x)                           # [B,C,T]  (T preserved)
        m = token_mask.unsqueeze(1)                # [B,1,T]
        h_masked = h * m
        lengths = token_mask.sum(1).clamp(min=1).unsqueeze(1)  # [B,1]
        pooled = h_masked.sum(2) / lengths                     # [B,C]

        rep = self.proj(pooled)                    # [B,proj_dim]
        mu = self.to_mu(rep)                       # [B,latent_dim]
        logvar = self.to_logvar(rep)               # [B,latent_dim]
        return mu, logvar

# ----------------------------- Decoder (GRU) -----------------------------
class DecoderGRU(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, latent_dim: int, pad_idx: int, sos_idx: int, eos_idx: int, 
                 num_layers: int = 3, dropout: float = 0.1, emb_dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx, self.sos_idx, self.eos_idx = pad_idx, sos_idx, eos_idx
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.emb_ln = nn.LayerNorm(d_model)
        self.emb_do = nn.Dropout(emb_dropout)
        self.z_to_h0 = nn.Linear(latent_dim, d_model)
        self.gru = nn.GRU( input_size=d_model, hidden_size=d_model, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.out_do = nn.Dropout(dropout)
        self.proj   = nn.Linear(d_model, vocab_size)

    # ---- Teacher-forcing path (kept for completeness) ----
    def forward_tf(self, z, tgt_tokens):
        h = self.z_to_h0(z).tanh().unsqueeze(0).repeat(self.gru.num_layers, 1, 1)  # [L,B,D]
        x = self.emb(tgt_tokens[:, :-1])                         # [B,T-1,D]
        x = self.emb_do(self.emb_ln(x))
        out, _ = self.gru(x, h)                                  # [B,T-1,D]
        out = self.out_do(out)
        logits = self.proj(out)                                  # [B,T-1,V]
        return logits

    # ---- Greedy decode (tokens only) ----
    @torch.no_grad()
    def greedy_tokens(self, z, max_len=None):
        B, device = z.size(0), z.device
        max_len = max_len or self.max_len
        h = self.z_to_h0(z).tanh().unsqueeze(0).repeat(self.gru.num_layers, 1, 1)
        ys = torch.full((B,1), self.sos_idx, dtype=torch.long, device=device)
        out_tok = []
        for _ in range(max_len):
            x = self.emb(ys[:, -1:])
            x = self.emb_do(self.emb_ln(x))
            o, h = self.gru(x, h)                                # [B,1,D]
            logit = self.proj(self.out_do(o))[:, -1, :]          # [B,V]
            nxt = logit.argmax(-1, keepdim=True)                 # [B,1]
            ys = torch.cat([ys, nxt], 1)
            out_tok.append(nxt)
            if (nxt == self.eos_idx).all():
                break
        return torch.cat(out_tok, 1) if out_tok else ys          # excludes initial <SOS> position

    # ---- Free-running greedy with logits (for no-TF training/validation) ----
    def greedy_with_logits(self, z, steps: int, start_token: torch.Tensor = None):

        B, device = z.size(0), z.device
        h = self.z_to_h0(z).tanh().unsqueeze(0).repeat(self.gru.num_layers, 1, 1)
        if start_token is None:
            ys = torch.full((B,1), self.sos_idx, dtype=torch.long, device=device)
        else:
            ys = start_token

        logits_list = []
        tok_list = []
        for _ in range(steps):
            x = self.emb(ys[:, -1:])
            x = self.emb_do(self.emb_ln(x))
            o, h = self.gru(x, h)                                 # [B,1,D]
            logit = self.proj(self.out_do(o))[:, -1, :]           # [B,V]
            nxt = logit.argmax(-1, keepdim=True)                  # [B,1]
            ys = torch.cat([ys, nxt], 1)
            logits_list.append(logit.unsqueeze(1))                # [B,1,V]
            tok_list.append(nxt)                                  # [B,1]

        logits = torch.cat(logits_list, dim=1)                    # [B,steps,V]
        tokens = torch.cat(tok_list, dim=1)                       # [B,steps]
        return logits, tokens

# ----------------------------- Full VAE -----------------------------
class CNNCharVAE(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, latent_dim: int, pad_idx: int, sos_idx: int, eos_idx: int, enc_layers: int, 
                 dec_layers: int, dropout: float = 0.1, emb_dropout: float = 0.1, max_len: int = 512, proj_dim: int = 196):
        super().__init__()
        self.pad_idx, self.sos_idx, self.eos_idx = pad_idx, sos_idx, eos_idx
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.encoder = EncoderCNN( vocab_size=vocab_size, d_model=d_model, latent_dim=latent_dim, pad_idx=pad_idx, enc_layers=enc_layers, conv_channels=None,
                                  conv_kernels=None, conv_dropout=dropout, emb_dropout=emb_dropout, proj_dim=proj_dim)
        self.decoder = DecoderGRU( vocab_size=vocab_size, d_model=d_model, latent_dim=latent_dim, pad_idx=pad_idx, sos_idx=sos_idx, eos_idx=eos_idx,
                                  num_layers=dec_layers, dropout=dropout, emb_dropout=emb_dropout, max_len=max_len)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor | None = None, teacher_forcing: bool = True, collect_logits: bool = False, max_len: int | None = None):

        mu, logvar = self.encoder(src)
        z = self.reparameterize(mu, logvar)

        if teacher_forcing:
            assert tgt is not None, "tgt is required for teacher forcing"
            logits = self.decoder.forward_tf(z, tgt_tokens=tgt)       # [B,T-1,V]
            return logits, mu, logvar
        
        if collect_logits and (tgt is not None):
            steps = tgt.size(1) - 1
            logits, _tokens = self.decoder.greedy_with_logits(z, steps=steps)
            return logits, mu, logvar
        else:
            tokens = self.decoder.greedy_tokens(z, max_len=max_len or self.max_len)
            return tokens, mu, logvar

    # ----------------------- Beam Search (testing only) -----------------------
    @torch.no_grad()
    def beam_search(self, src, beam_size=4, max_len=128, length_penalty=1.0):
        mu, logvar = self.encoder(src)
        z = self.reparameterize(mu, logvar)                      # [B,L]
        B, device = z.size(0), z.device

        # Expand z for beams
        z = z.unsqueeze(1).repeat(1, beam_size, 1).view(B * beam_size, -1)

        # Init decoder state
        h = self.decoder.z_to_h0(z).tanh().unsqueeze(0).repeat(self.decoder.gru.num_layers, 1, 1)
        ys = torch.full((B * beam_size, 1), self.sos_idx, dtype=torch.long, device=device)

        beam_scores = torch.full((B, beam_size), -1e9, device=device); beam_scores[:, 0] = 0.0
        beam_scores = beam_scores.view(-1)
        finished = torch.zeros(B * beam_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            x = self.decoder.emb(ys[:, -1:])
            x = self.decoder.emb_do(self.decoder.emb_ln(x))
            o, h = self.decoder.gru(x, h)                        # [B*beam,1,D]
            logp = F.log_softmax(self.decoder.proj(self.decoder.out_do(o))[:, -1, :], dim=-1)  # [B*beam,V]

            cand = (beam_scores.unsqueeze(1) + logp).view(B, beam_size, -1)                    # [B,beam,V]
            topk_scores, topk_idx = torch.topk(cand.view(B, -1), k=beam_size, dim=-1)
            next_beam = torch.div(topk_idx, logp.size(-1), rounding_mode='floor')
            next_tok  = topk_idx % logp.size(-1)

            base = (torch.arange(B, device=device) * beam_size).unsqueeze(1)
            sel  = (base + next_beam).view(-1)

            ys = torch.cat([ys[sel], next_tok.view(-1, 1)], dim=1)
            h  = h[:, sel, :]
            beam_scores = topk_scores.view(-1)
            finished = finished[sel] | (next_tok.view(-1) == self.eos_idx)

        seqs   = ys.view(B, beam_size, -1)
        scores = beam_scores.view(B, beam_size)
        lengths = (seqs != self.pad_idx).sum(-1).clamp(min=1).float()
        norm_scores = scores / (lengths ** length_penalty)

        is_finished = (seqs == self.eos_idx).any(-1)
        best_idx = torch.where(is_finished, norm_scores, norm_scores - 1e6).argmax(dim=1)
        best = seqs[torch.arange(B, device=device), best_idx]
        out  = best[:, 1:]  # drop <SOS>
        assert (out[:, 0] != self.sos_idx).all(), "beam_search should not return <SOS> first"
        return out