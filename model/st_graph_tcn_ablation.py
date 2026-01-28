# -*- coding: utf-8 -*-
# model/st_graph_tcn_pred.py
# Only added: disable_temporal / disable_spatial switches in forward, with minimal supporting changes

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Temporal branch: Partial Causal Conv + causal Transformer ----------

class PartialCausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        self.ks = kernel_size
        self.dl = dilation
        pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation, bias=True)
        self.register_buffer("ones_kernel", torch.ones(1, 1, kernel_size))

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        x_m = x * m
        y = super().forward(x_m)
        denom = F.conv1d(m, self.ones_kernel.to(m),
                         padding=self.padding[0], dilation=self.dilation[0])
        cut = (self.ks - 1) * self.dl
        if cut > 0:
            y = y[..., :-cut]
            denom = denom[..., :-cut]
        y = y / denom.clamp_min(1e-6)
        return y, (denom > 0).float()

class TCNBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "pconv": PartialCausalConv1d(d_model, d_model, kernel_size=kernel_size, dilation=2**i),
                "act": nn.ReLU(),
                "drop": nn.Dropout(dropout),
                "pw": nn.Conv1d(d_model, d_model, kernel_size=1),
            }) for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, m_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x.transpose(1, 2)       # [B,D,T]
        m = m_t.unsqueeze(1)        # [B,1,T]
        for blk in self.layers:
            res = h
            y, m = blk["pconv"](h, m)
            y = blk["drop"](blk["act"](y))
            y = blk["pw"](y)
            h = F.relu(y) + res
        return self.norm(h.transpose(1, 2)), m.squeeze(1)

class TemporalEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, tcn_layers: int = 3,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.tcn = TCNBlock(d_model, kernel_size=kernel_size, n_layers=tcn_layers, dropout=dropout)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.trans = nn.TransformerEncoder(enc, num_layers=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Ht_in: torch.Tensor, m_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = Ht_in.shape
        device = Ht_in.device
        h, m_out = self.tcn(Ht_in, m_t)
        causal = torch.full((T, T), float('-inf'), device=device).triu(1)
        key_pad = (m_out < 0.5)  # True=mask
        h = self.trans(h, mask=causal, src_key_padding_mask=key_pad)
        return self.norm(h), (~key_pad).float()

# ---------- Spatial branch: attention-induced dynamic graph + static fusion ----------

class DynamicGraphAttention(nn.Module):
    def __init__(self, d_model: int, eta: float = 1.0):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.eta = eta

    def forward(self, Hn: torch.Tensor, A_stat: torch.Tensor, M_mask: torch.Tensor) -> torch.Tensor:
        B, N, D = Hn.shape
        Q = self.Wq(Hn)
        K = self.Wk(Hn)
        logits = (Q @ K.transpose(1, 2)) / math.sqrt(D)
        bias = torch.logit(A_stat.clamp(1e-3, 1 - 1e-3))
        logits = logits + self.eta * bias.unsqueeze(0)
        mask = (M_mask == 0).unsqueeze(0)
        logits = logits.masked_fill(mask, float("-inf"))
        return torch.softmax(logits, dim=-1)

class GraphPropagate(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.lin  = nn.Linear(d_in, d_out, bias=False)
        self.norm = nn.LayerNorm(d_out)
        self.do   = nn.Dropout(dropout)

    def forward(self, Hn: torch.Tensor, A: torch.Tensor):
        H = torch.matmul(A, Hn)
        H = self.lin(H)
        return self.norm(self.do(F.relu(H)))

class SpatialEncoder(nn.Module):
    def __init__(self, in_ch: int, d_model: int, eta: float = 1.0, beta: float = 0.6, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_ch, d_model, bias=False)
        self.dyn = DynamicGraphAttention(d_model, eta=eta)
        self.prop = GraphPropagate(d_model, d_model, dropout=dropout)
        self.beta = beta
        self.norm = nn.LayerNorm(d_model)

    def forward(self, X_short: torch.Tensor, mask_short: torch.Tensor,
                A_stat: torch.Tensor, M_mask: torch.Tensor,
                node_reliab: torch.Tensor):
        B, N, S, C = X_short.shape
        denom = mask_short.sum(dim=2, keepdim=True).clamp_min(1.0)
        x = (X_short * mask_short.unsqueeze(-1)).sum(dim=2) / denom
        Hn = self.proj(x)                                  # [B,N,D]

        Adyn = self.dyn(Hn, A_stat, M_mask)                # [B,N,N]

        A_fuse = self.beta * Adyn + (1 - self.beta) * A_stat.unsqueeze(0)
        r = node_reliab.clamp(0.0, 1.0).unsqueeze(1)
        A_fuse = A_fuse * r.transpose(1, 2) * r
        A_fuse = A_fuse * M_mask.unsqueeze(0)
        A_fuse = A_fuse / (A_fuse.sum(dim=-1, keepdim=True) + 1e-8)

        Hn = self.prop(Hn, A_fuse)
        return self.norm(Hn), A_fuse

# ---------- Cross-branch sync ----------

class CrossAttnSync(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.t2s = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.s2t = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ln_t = nn.LayerNorm(d_model)
        self.ln_s = nn.LayerNorm(d_model)
        self.do = nn.Dropout(dropout)

    @staticmethod
    def _causal_mask(T: int, device):
        m = torch.full((T, T), float('-inf'), device=device)
        return torch.triu(m, diagonal=1)

    @staticmethod
    def _build_s2t_prefix_mask(T: int, N: int, device) -> torch.Tensor:
        base = CrossAttnSync._causal_mask(T, device)      # [T,T]
        base = base.unsqueeze(0).repeat(N, 1, 1)          # [N,T,T]
        return base.reshape(N * T, T)                     # [T*N,T]

    def forward(self, Ht: torch.Tensor, Hs: torch.Tensor, m_t: torch.Tensor):
        B, T, D = Ht.shape
        N = Hs.size(1)
        device = Ht.device

        Ht_new, _ = self.t2s(query=Ht, key=Hs, value=Hs)
        Ht_sync = self.ln_t(Ht + self.do(Ht_new))

        Hs_rep = Hs.unsqueeze(1).expand(B, T, N, D).reshape(B, T * N, D)
        attn_mask = self._build_s2t_prefix_mask(T, N, device)
        key_pad = (m_t < 0.5)
        Hs_s2t, _ = self.s2t(query=Hs_rep, key=Ht, value=Ht,
                             attn_mask=attn_mask, key_padding_mask=key_pad)
        Hs_s2t = Hs_s2t.view(B, T, N, D)
        Hs_base = Hs.unsqueeze(1).expand_as(Hs_s2t)
        Hs_sync_t = self.ln_s(Hs_base + self.do(Hs_s2t))
        return Ht_sync, Hs_sync_t

# ---------- Top-level model (prediction head) ----------

class STGraphTCN(nn.Module):
    def __init__(
        self,
        num_nodes_hint: Optional[int],
        in_feat: int,
        d_model: int = 128,
        short_kernel: int = 5,
        nhead: int = 4,
        tcn_layers: int = 3,
        dropout: float = 0.1,
        eta: float = 1.0,
        beta: float = 0.6,
        out_feat: Optional[int] = None,
        horizon: int = 1
    ):
        super().__init__()
        self.in_feat = in_feat
        self.d_model = d_model
        self.horizon = int(horizon)
        self.out_feat = int(out_feat) if out_feat is not None else in_feat

        self.feat_proj_t = None  # Lazy init: Linear(N_mic*F -> D)
        self.feat_proj_s = nn.Linear(in_feat, d_model)

        self.temporal = TemporalEncoder(d_model, nhead=nhead, tcn_layers=tcn_layers, dropout=dropout)
        self.spatial  = SpatialEncoder(in_ch=d_model, d_model=d_model, eta=eta, beta=beta, dropout=dropout)
        self.cross    = CrossAttnSync(d_model, nhead=nhead, dropout=dropout)

        # When spatial branch is disabled, map temporal features to node embeddings
        self.time_to_node = nn.Linear(d_model, d_model)

        self.pred_node = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.horizon * self.out_feat)
        )

        self._num_nodes_hint = num_nodes_hint
        self._short_kernel = int(short_kernel)

    def _ensure_time_proj(self, N_mic: int, device):
        if (self.feat_proj_t is None) or (self.feat_proj_t.in_features != N_mic * self.in_feat):
            self.feat_proj_t = nn.Linear(N_mic * self.in_feat, self.d_model).to(device)

    @staticmethod
    def _gather_nodes(X: torch.Tensor, node_index: torch.Tensor) -> torch.Tensor:
        return X.index_select(dim=2, index=node_index)

    def _spatial_all_steps(self, X: torch.Tensor, Zmask: torch.Tensor,
                           A_stat: torch.Tensor, M_mask: torch.Tensor, short_patch: int):
        B, T, N_all, F = X.shape
        device = X.device
        N = A_stat.size(0)

        node_index = torch.arange(N, device=device, dtype=torch.long)
        X_sel = self._gather_nodes(X, node_index)                      # [B,T,N,F]
        Zm_sel = Zmask.index_select(2, node_index).float()             # [B,T,N]

        S = int(short_patch) if short_patch else min(5, int(T))
        pads_X = torch.zeros(B, S - 1, N, F, device=device)
        pads_M = torch.ones(B, S - 1, N, device=device)
        X_pad = torch.cat([pads_X, X_sel], dim=1)
        M_pad = torch.cat([pads_M, Zm_sel], dim=1)

        short_seq, mask_seq = [], []
        for t in range(T):
            xs = X_pad[:, t:t + S]                                     # [B,S,N,F]
            ms = M_pad[:, t:t + S]                                     # [B,S,N]
            xs = self.feat_proj_s(xs)                                  # -> [B,S,N,D]
            short_seq.append(xs.permute(0, 2, 1, 3))                   # -> [B,N,S,D]
            mask_seq.append(ms.permute(0, 2, 1))                       # -> [B,N,S]

        X_short_t = torch.stack(short_seq, dim=1)                       # [B,T,N,S,D]
        M_short_t = torch.stack(mask_seq, dim=1)                        # [B,T,N,S]

        Hn_list, Afuse_list = [], []
        for t in range(T):
            node_rel_t = M_short_t[:, t].float().mean(dim=2)           # [B,N]
            Hn_t, Af_t = self.spatial(X_short_t[:, t], M_short_t[:, t].float(),
                                      A_stat, M_mask, node_rel_t)
            Hn_list.append(Hn_t)
            Afuse_list.append(Af_t)
        Hs_t = torch.stack(Hn_list, dim=1)                               # [B,T,N,D]
        A_fuse_t = torch.stack(Afuse_list, dim=1)                        # [B,T,N,N]
        return Hs_t, A_fuse_t, node_index

    def forward(self, X: torch.Tensor, A_stat: torch.Tensor, M_mask: torch.Tensor,
                node_index: Optional[torch.Tensor] = None, short_patch: int = 5,
                Zmask: Optional[torch.Tensor] = None,
                disable_temporal: bool = False, disable_spatial: bool = False):
        """
        disable_temporal=True: drop temporal branch (spatial only)
        disable_spatial=True : drop spatial branch (temporal only)
        """
        B, T, N_all, F = X.shape
        device = X.device
        N = A_stat.size(0)
        if node_index is None:
            node_index = torch.arange(N, device=device, dtype=torch.long)
        if Zmask is None:
            Zmask = torch.ones(B, T, N_all, device=device)

        # ---------- Temporal branch ----------
        X_sel = self._gather_nodes(X, node_index)                       # [B,T,N,F]
        Xt = X_sel.reshape(B, T, -1)                                    # [B,T,N*F]
        self._ensure_time_proj(N, device)
        Ht_in = self.feat_proj_t(Xt)                                    # [B,T,D]
        m_t = Zmask.index_select(2, node_index).float().mean(dim=2)     # [B,T]

        if not disable_temporal:
            Ht, m_t_out = self.temporal(Ht_in, m_t)                     # regular temporal encoding
        else:
            # Temporal disabled: no sequence modeling, provide zero-impact placeholder
            Ht = torch.zeros_like(Ht_in)
            m_t_out = torch.ones_like(m_t)

        # ---------- Spatial branch ----------
        if not disable_spatial:
            Hs_t, A_fuse_t, node_index_ = self._spatial_all_steps(
                X, Zmask, A_stat, M_mask, short_patch or self._short_kernel
            )                                                            # [B,T,N,D], [B,T,N,N]
        else:
            # Spatial disabled: skip graph encoding, provide zero placeholder
            Hs_t = torch.zeros(B, T, N, self.d_model, device=device)
            A_fuse_t = torch.zeros(B, T, N, N, device=device)

        # ---------- Assemble prediction inputs ----------
        if (not disable_temporal) and (not disable_spatial):
            # Full model: cross-branch sync + node prediction
            Ht_sync, Hs_sync_t = self.cross(Ht, Hs_t[:, -1], m_t_out)
            Hn_last = Hs_sync_t[:, -1]                                  # [B,N,D]
        elif disable_temporal and (not disable_spatial):
            # Spatial only: use last-step spatial node representations
            Hn_last = Hs_t[:, -1]                                       # [B,N,D]
        elif (not disable_temporal) and disable_spatial:
            # Temporal only: adapt last-step Ht and replicate to each node
            h_t_last = Ht[:, -1, :]                                     # [B,D]
            h_node = self.time_to_node(h_t_last)                         # [B,D]
            Hn_last = h_node.unsqueeze(1).expand(B, N, self.d_model)     # [B,N,D]
        else:
            # Both disabled (should not happen): fall back to zeros
            Hn_last = torch.zeros(B, N, self.d_model, device=device)

        # ---------- Prediction head ----------
        y_flat = self.pred_node(Hn_last)                                # [B,N,H*out_feat]
        y_hat = y_flat.view(B, N, self.horizon, self.out_feat).permute(0, 2, 1, 3)  # [B,H,N,out_feat]

        aux = {
            "A_fuse": A_fuse_t[:, -1] if A_fuse_t.numel()>0 else None,
            "A_fuse_t": A_fuse_t if A_fuse_t.numel()>0 else None,
            "Ht": Ht,                          # temporal features before sync (debug/visualization)
            "Hs_sync_last": Hn_last,           # node features entering the decoder
            "node_index": node_index
        }
        return y_hat, aux
