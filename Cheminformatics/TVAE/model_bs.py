import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, global_mean_pool

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


# ───────────────────────────── Graph Attention Based Encoder  ────────────────────────────────

class GraphEncoderGAT(nn.Module):
    """
    Graph attention encoder (GATv2Conv stack) + global add pool -> pooled graph embedding.

    Returns:
      seq_out: node embeddings (ragged, concatenated)  [num_nodes_total, d_model]
      src_key_padding_mask: None (not applicable for graphs)
      pooled: graph embedding per molecule            [B, d_model]
    """
    def __init__( self, node_feat_dim: int, edge_feat_dim: int, d_model: int, enc_layers: int, nhead: int, dropout: float, use_layernorm: bool = True, ):
        super().__init__()
        self.node_in = nn.Linear(node_feat_dim, d_model)
        self.edge_dim = edge_feat_dim

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layernorm else None
        self.dropout = dropout
        assert d_model % nhead == 0, "d_model must be divisible by nhead for concat=True"
        out_per_head = d_model // nhead

        for _ in range(enc_layers):
            self.convs.append(GATv2Conv( in_channels=d_model, out_channels=out_per_head, heads=nhead, concat=True, dropout=dropout, edge_dim=edge_feat_dim if edge_feat_dim > 0 else None, add_self_loops=False ))
            if use_layernorm:
                self.norms.append(nn.LayerNorm(d_model))

        self.out_ln  = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        self.pool_ln = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

    def forward(self, src_graph_batch):
        """
        src_graph_batch: torch_geometric.data.Batch with fields:
          - x         [num_nodes_total, node_feat_dim]
          - edge_index[2, num_edges_total]
          - edge_attr [num_edges_total, edge_feat_dim]  (optional if edge_feat_dim=0)
          - batch     [num_nodes_total] graph id per node
        """
        x = self.node_in(src_graph_batch.x)

        edge_attr = getattr(src_graph_batch, "edge_attr", None)
        if self.edge_dim == 0:
            edge_attr = None

        for i, conv in enumerate(self.convs):
            h = conv(x, src_graph_batch.edge_index, edge_attr=edge_attr)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = h
            if self.norms is not None:
                x = self.norms[i](x)

        seq_out = self.out_ln(x)  # node embeddings

        pooled = global_mean_pool(seq_out, src_graph_batch.batch)  # [B, d_model]
        pooled = self.pool_ln(pooled)

        # not meaningful for graphs, but keep signature aligned
        src_key_padding_mask = None
        return seq_out, src_key_padding_mask, pooled


# ---------- Transformer Decoder ----------
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, pad_idx: int, dropout: float, max_len: int = 160, dim_feedforward: int | None = None):
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
class TVAE(nn.Module):
    def __init__( self, vocab_size: int, d_model: int, latent_dim: int, pad_idx: int, sos_idx: int, eos_idx: int, enc_layers: int, dec_layers: int, enc_heads: int, dec_heads: int, dropout: float, max_len: int, node_feat_dim: int = 10, edge_feat_dim: int = 6, dim_feedforward: int | None = None, ):
        super().__init__()
        self.pad_idx, self.sos_idx, self.eos_idx = pad_idx, sos_idx, eos_idx
        self.d_model = d_model
        self.latent_dim = latent_dim
        # ---- Graph encoder (paper-style GATv2) ----
        self.encoder = GraphEncoderGAT( node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim, d_model=d_model, enc_layers=enc_layers, nhead=enc_heads, dropout=dropout)
        # ---- VAE heads ----
        self.to_mu     = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)
        # ---- latent -> 1 token memory (keep SAME as your previous setup) ----
        self.latent_to_token = nn.Sequential(nn.Linear(latent_dim, d_model), nn.LayerNorm(d_model))

        # ---- decoder unchanged ----
        self.decoder = TransformerDecoder(vocab_size=vocab_size, d_model=d_model, nhead=dec_heads, num_layers=dec_layers, pad_idx=pad_idx, dropout=dropout, max_len=max_len, dim_feedforward=dim_feedforward)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def _get_batch_size(self, src):
        """
        src is a PyG Batch: src.num_graphs is batch size.
        """
        # torch_geometric.data.Batch has num_graphs
        if hasattr(src, "num_graphs"):
            return int(src.num_graphs)
        # fallback (shouldn't happen)
        if hasattr(src, "batch"):
            return int(src.batch.max().item() + 1) if src.batch.numel() else 0
        raise ValueError("Cannot infer batch size from src; expected PyG Batch with .num_graphs or .batch")

    def _get_device(self, src):
        """
        PyG Batch doesn't have .device, but src.x does.
        """
        if hasattr(src, "x") and torch.is_tensor(src.x):
            return src.x.device
        # fallback: search any tensor field
        for v in src.__dict__.values():
            if torch.is_tensor(v):
                return v.device
        raise ValueError("Cannot infer device from src; expected PyG Batch with tensor fields like .x")

    def _encode(self, src):
        # src: PyG Batch
        mem, mem_pad, pooled_h = self.encoder(src)   # pooled_h: [B, d_model]
        mu     = self.to_mu(pooled_h)                # [B, latent_dim]
        logvar = self.to_logvar(pooled_h)            # [B, latent_dim]
        z      = self.reparameterize(mu, logvar)     # [B, latent_dim]
        # keep your original arrangement: latent -> single token memory
        z_tok  = self.latent_to_token(z).unsqueeze(1)  # [B, 1, d_model]
        memory = z_tok
        device = memory.device
        mem_pad = torch.zeros(memory.size(0), 1, dtype=torch.bool, device=device)  # [B, 1]
        return memory, mem_pad, mu, logvar

    def _rand_tokens(self, shape, device):
        V = self.decoder.emb.num_embeddings
        t = torch.randint(0, V, shape, device=device)
        for bad in (self.pad_idx, self.sos_idx, self.eos_idx):
            t = torch.where(t == bad, (t + 1) % V, t)
        return t

    def _corrupt_tgt_inputs(self, tgt_inp, p: float):
        if p <= 0:
            return tgt_inp
        keep = (tgt_inp != self.pad_idx) & (tgt_inp != self.sos_idx) & (tgt_inp != self.eos_idx)
        drop = (torch.rand_like(tgt_inp, dtype=torch.float) < p) & keep
        noise = self._rand_tokens(tgt_inp.shape, tgt_inp.device)
        return torch.where(drop, noise, tgt_inp)

    def forward(self, src, tgt=None, teacher_forcing=True, max_len=128, corruption_p: float = 0.0):
        memory, mem_pad, mu, logvar = self._encode(src)

        # ---------- teacher forcing ----------
        if teacher_forcing and tgt is not None:
            tgt_inp = tgt[:, :-1]
            if corruption_p > 0:
                tgt_inp = self._corrupt_tgt_inputs(tgt_inp, corruption_p)
            logits = self.decoder(tgt_inp, memory, mem_pad)
            return logits, mu, logvar

        # ---------- greedy decoding ----------
        B = self._get_batch_size(src)
        device = self._get_device(src)

        ys = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=device)
        out = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits = self.decoder(ys, memory, mem_pad)
            logits[:, -1, self.pad_idx] = -1e9
            logits[:, -1, self.sos_idx] = -1e9
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)

            # freeze sequences that already finished
            nxt = torch.where(finished.unsqueeze(1), torch.full_like(nxt, self.eos_idx), nxt)

            ys = torch.cat([ys, nxt], dim=1)
            out.append(nxt)

            finished = finished | (nxt.squeeze(1) == self.eos_idx)
            if finished.all():
                break

        gen = torch.cat(out, dim=1) if out else ys[:, 1:]
        return gen, mu, logvar

    @torch.no_grad()
    def beam_search(self, src, beam_size=4, max_len=128, length_penalty=0.6):
        memory, mem_pad, _, _ = self._encode(src)

        B = self._get_batch_size(src)
        device = self._get_device(src)

        memory = memory.unsqueeze(1).repeat(1, beam_size, 1, 1).view(B * beam_size, memory.size(1), memory.size(2))
        mem_pad = mem_pad.unsqueeze(1).repeat(1, beam_size, 1).view(B * beam_size, mem_pad.size(1))

        ys = torch.full((B * beam_size, 1), self.sos_idx, dtype=torch.long, device=device)

        beam_scores = torch.full((B, beam_size), -1e9, device=device)
        beam_scores[:, 0] = 0.0
        beam_scores = beam_scores.view(-1)

        finished = torch.zeros(B * beam_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits = self.decoder(ys, memory, mem_pad)
            logp = F.log_softmax(logits[:, -1, :], dim=-1)
            rep_penalty = 1.2  # best starting value
            last_tok = ys[:, -1]  # [B*beam]
            logp[torch.arange(logp.size(0), device=logp.device), last_tok] /= rep_penalty

            # --- prevent empty outputs: disallow EOS too early ---
            min_len = 5  # best practical default for SMILES
            cur_len = ys.size(1) - 1  # generated tokens so far (excluding SOS)
            if cur_len < min_len:
                logp[:, self.eos_idx] = -float("inf")

            if finished.any():
                frozen = torch.full_like(logp, -float("inf"))
                frozen[:, self.eos_idx] = 0.0
                logp = torch.where(finished.unsqueeze(1), frozen, logp)

            cand = (beam_scores.unsqueeze(1) + logp).view(B, beam_size, -1)
            topk_scores, topk_idx = torch.topk(cand.view(B, -1), k=beam_size, dim=-1)

            next_beam = torch.div(topk_idx, logp.size(-1), rounding_mode="floor")
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

        eos_hits  = (seqs == self.eos_idx)
        has_eos   = eos_hits.any(dim=-1)
        first_eos = torch.argmax(eos_hits.to(torch.int32), dim=-1)
        eff_len   = torch.where(has_eos, first_eos + 1, torch.full_like(first_eos, seqs.size(2))).clamp(min=1)

        lp = ((5.0 + eff_len.float()) ** length_penalty) / ((5.0 + 1.0) ** length_penalty)
        norm = scores / lp

        pref = torch.where(has_eos, norm, norm - 1e6)
        best_idx = pref.argmax(dim=1)

        best = seqs[torch.arange(B, device=device), best_idx]
        out = best[:, 1:]
        return out