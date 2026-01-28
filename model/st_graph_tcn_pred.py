# -*- coding: utf-8 -*-
# model/st_graph_tcn_pred.py
# Notes (original design kept, with performance and stability fixes):
# - Temporal branch: Partial Causal Conv + Transformer (bool masks to avoid dtype warnings)
# - Spatial branch: short-window masked mean + dynamic graph attention + static graph fusion (vectorized)
# - Cross-branch: t2s uses MHA, s2t uses MHA with bool prefix masks, same interface
# - Prediction head: per-node multi-step forecasting
# - Minor tweaks: precompute logit(A_stat) as buffer, in-place relu_, contiguous after transpose

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Utilities ----------

def causal_bool_mask(T: int, device) -> torch.Tensor:
    """
    Return a [T, T] upper-triangular boolean mask (True = mask future positions).
    """
    return torch.ones((T, T), dtype=torch.bool, device=device).triu(1)


# ---------- Temporal branch: Partial Causal Conv + causal Transformer ----------

class PartialCausalConv1d(nn.Conv1d):
    """
    Partial causal convolution:
    - Accumulate only mask=1 positions and normalize by the valid count;
    - Right trim to keep causality (no peeking into the future).
    Input:
      x:[B,C,T], m:[B,1,T] (1=valid, 0=anomalous/missing)
    """
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        self.ks = int(kernel_size)
        self.dl = int(dilation)
        pad = (self.ks - 1) * self.dl
        super().__init__(in_ch, out_ch, self.ks, padding=pad, dilation=self.dl, bias=True)
        self.register_buffer("ones_kernel", torch.ones(1, 1, self.ks))

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        # x:[B,C,T], m:[B,1,T]
        x_m = x * m
        y = super().forward(x_m)
        # denom counts valid samples
        denom = F.conv1d(m, self.ones_kernel, padding=self.padding[0], dilation=self.dilation[0])
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
                "drop": nn.Dropout(dropout),
                "pw": nn.Conv1d(d_model, d_model, kernel_size=1),
            }) for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, m_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:[B,T,D], m_t:[B,T] (1=valid)
        Returns: h:[B,T,D], m_out:[B,T]
        """
        h = x.transpose(1, 2).contiguous()     # [B,D,T]
        m = m_t.unsqueeze(1)                   # [B,1,T]
        for blk in self.layers:
            res = h
            y, m = blk["pconv"](h, m)          # [B,D,T], [B,1,T]
            y = F.relu_(y)
            y = blk["pw"](blk["drop"](y))
            h = F.relu_(y + res)
        return self.norm(h.transpose(1, 2).contiguous()), m.squeeze(1)


class TemporalEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, tcn_layers: int = 3,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.tcn = TCNBlock(d_model, kernel_size=kernel_size, n_layers=tcn_layers, dropout=dropout)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.trans = nn.TransformerEncoder(enc, num_layers=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Ht_in: torch.Tensor, m_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ht_in:[B,T,D], m_t:[B,T] (1=valid)
        Returns: h:[B,T,D], m_out:[B,T] (passed to downstream modules)
        """
        B, T, _ = Ht_in.shape
        device = Ht_in.device

        # TCN (with mask)
        h, m_out = self.tcn(Ht_in, m_t)

        # Causal + key padding (both bool masks)
        causal = causal_bool_mask(T, device)        # True=mask future
        key_pad = (m_out < 0.5)                     # True=mask this time step
        h = self.trans(h, mask=causal, src_key_padding_mask=key_pad)
        return self.norm(h), (~key_pad).float()     # Return valid-step mask (float)


# ---------- Spatial branch: attention-induced dynamic graph + static fusion (with reliability down-weighting) ----------

class DynamicGraphAttention(nn.Module):
    def __init__(self, d_model: int, eta: float = 1.0):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.eta = float(eta)
        self.register_buffer("_bias_logit", None, persistent=False)  # [1,1,N,N]

    @torch.no_grad()
    def set_static_bias(self, A_stat: torch.Tensor):
        # Precompute and scale logit prior bias
        b = torch.logit(A_stat.clamp(1e-3, 1 - 1e-3)) * self.eta
        self.register_buffer("_bias_logit", b[None, None, ...])  # [1,1,N,N]

    def forward(self, Hn_t: torch.Tensor, A_stat: torch.Tensor, M_mask: torch.Tensor) -> torch.Tensor:
        """
        Hn_t: [B,T,N,D]
        A_stat, M_mask: [N,N] (static graph and available-edge mask)
        Returns: Adyn_t: [B,T,N,N] (row-normalized)
        """
        B, T, N, D = Hn_t.shape
        if self._bias_logit is None:
            self.set_static_bias(A_stat)

        Q = self.Wq(Hn_t)             # [B,T,N,D]
        K = self.Wk(Hn_t)             # [B,T,N,D]
        # logits: [B,T,N,N]
        logits = torch.einsum("btnd,btmd->btnm", Q, K) / math.sqrt(D)
        logits = logits + self._bias_logit  # broadcast: [1,1,N,N]

        # Disable edges (bool -> -inf)
        mask = (M_mask == 0)
        logits = logits.masked_fill(mask, float("-inf"))
        Adyn = torch.softmax(logits, dim=-1)  # row-normalized
        return Adyn


class GraphPropagate(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.lin  = nn.Linear(d_in, d_out, bias=False)
        self.norm = nn.LayerNorm(d_out)
        self.do   = nn.Dropout(dropout)

    def forward(self, Hn: torch.Tensor, A: torch.Tensor):
        # Hn:[B,T,N,D], A:[B,T,N,N] or [B,N,N] or [N,N]
        H = torch.einsum("btnm,btmd->btnd", A, Hn) if A.dim()==4 else torch.einsum("bnm,btmd->btnd", A, Hn)
        H = self.lin(H)
        H = F.relu_(H)
        H = self.do(H)
        return self.norm(H)


class SpatialEncoder(nn.Module):
    """
    Short-window → per-node representation (mask-weighted), then dynamic + static graph fusion
    with node reliability down-weighting.
    * Fully vectorized; no per-time-step Python loops.
    """
    def __init__(self, in_ch: int, d_model: int, eta: float = 1.0, beta: float = 0.6, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_ch, d_model, bias=False)
        self.dyn = DynamicGraphAttention(d_model, eta=eta)
        self.prop = GraphPropagate(d_model, d_model, dropout=dropout)
        self.beta = float(beta)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, X_short_t: torch.Tensor, mask_short_t: torch.Tensor,
                A_stat: torch.Tensor, M_mask: torch.Tensor,
                node_reliab_t: torch.Tensor):
        """
        X_short_t:   [B,T,N,S,F_proj]   per-node short window (pre-projection channels)
        mask_short_t:[B,T,N,S]          valid positions within the short window
        node_reliab_t:[B,T,N]           node reliability at current step r_i∈[0,1]
        """
        B, T, N, S, C = X_short_t.shape

        # Mask-weighted mean (short-window summary) -> [B,T,N,C]
        denom = mask_short_t.sum(dim=3, keepdim=True).clamp_min(1.0)   # [B,T,N,1]
        x = (X_short_t * mask_short_t.unsqueeze(-1)).sum(dim=3) / denom
        Hn = self.proj(x)                                              # [B,T,N,D]

        # Dynamic graph (vectorized, with time dimension)
        Adyn_t = self.dyn(Hn, A_stat, M_mask)                          # [B,T,N,N]

        # Dynamic-static fusion + reliability down-weighting + normalization (vectorized)
        Astat = A_stat.unsqueeze(0).unsqueeze(0)                       # [1,1,N,N]
        A_fuse_t = self.beta * Adyn_t + (1.0 - self.beta) * Astat      # [B,T,N,N]
        r = node_reliab_t.clamp(0.0, 1.0).unsqueeze(-1)                # [B,T,N,1]
        A_fuse_t = A_fuse_t * r * r.transpose(-2, -1)                  # [B,T,N,N]
        A_fuse_t = A_fuse_t * M_mask                                   # [N,N] broadcast
        A_fuse_t = A_fuse_t / (A_fuse_t.sum(dim=-1, keepdim=True) + 1e-8)

        # One-hop propagation
        Hn = self.prop(Hn, A_fuse_t)                                   # [B,T,N,D]
        return self.norm(Hn), A_fuse_t


# ---------- Cross-branch alignment: t2s + s2t (s2t uses key_padding_mask + prefix mask on time keys) ----------

class CrossAttnSync(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.t2s = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.s2t = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ln_t = nn.LayerNorm(d_model)
        self.ln_s = nn.LayerNorm(d_model)
        self.do = nn.Dropout(dropout)

    @staticmethod
    def _build_s2t_prefix_mask(T: int, N: int, device) -> torch.Tensor:
        """
        s2t query is [B,T*N,D] (flattened); each (t,n) must not attend to future t' > t.
        Returns a [T*N, T] upper-triangular boolean mask (True = masked).
        """
        base = causal_bool_mask(T, device)       # [T,T] True=mask future
        base = base.unsqueeze(0).repeat(N, 1, 1) # [N,T,T]
        return base.reshape(N * T, T)            # [T*N, T]

    def forward(self, Ht: torch.Tensor, Hs_t: torch.Tensor, m_t: torch.Tensor):
        """
        Ht:[B,T,D]; Hs_t:[B,T,N,D]; m_t:[B,T] (1=valid)
        Returns:
          Ht_sync:[B,T,D], Hs_sync_t:[B,T,N,D]
        """
        B, T, D = Ht.shape
        N = Hs_t.size(2)
        device = Ht.device

        # t2s: inject temporal context into the "last-step" node representations (same as original)
        Hs_last = Hs_t[:, -1]                                        # [B,N,D]
        Ht_new, _ = self.t2s(query=Ht, key=Hs_last, value=Hs_last)   # [B,T,D]
        Ht_sync = self.ln_t(Ht + self.do(Ht_new))                    # [B,T,D]

        # s2t: align nodes with temporal context at each time step (causal + key_padding_mask)
        Hs_rep = Hs_last.unsqueeze(1).expand(B, T, N, D).reshape(B, T * N, D)  # [B,T*N,D]
        attn_mask = self._build_s2t_prefix_mask(T, N, device)                  # [T*N,T] (bool)
        key_pad = (m_t < 0.5)                                                  # [B,T] True=mask
        Hs_s2t, _ = self.s2t(query=Hs_rep, key=Ht, value=Ht,
                             attn_mask=attn_mask, key_padding_mask=key_pad)
        Hs_s2t = Hs_s2t.view(B, T, N, D)
        Hs_base = Hs_last.unsqueeze(1).expand_as(Hs_s2t)
        Hs_sync_t = self.ln_s(Hs_base + self.do(Hs_s2t))                       # [B,T,N,D]
        return Ht_sync, Hs_sync_t


# ---------- Top-level model (prediction head) ----------

class STGraphTCN(nn.Module):
    """
    Inputs:
      X: [B, T, N_all, F]
      A_stat, M_mask: [N, N]  (static MIC graph and its available-edge mask)
      Zmask: [B, T, N_all]    (optional) 1=normal, 0=anomalous/missing (for robust aggregation)
      node_index: [N]         map full columns to the static-graph node subset
    Outputs:
      y_hat: [B, H, N, out_feat]   (H=horizon)
      aux:   diagnostics (fused adjacency, etc.)
    """
    def __init__(
        self,
        num_nodes_hint: Optional[int],
        in_feat: int,
        d_model: int = 128,
        short_kernel: int = 5,             # short-window length (spatial mask mean)
        nhead: int = 4,
        tcn_layers: int = 3,
        dropout: float = 0.1,
        eta: float = 1.0,
        beta: float = 0.6,
        out_feat: Optional[int] = None,    # defaults to in_feat
        horizon: int = 1                   # prediction horizon
    ):
        super().__init__()
        self.in_feat = int(in_feat)
        self.d_model = int(d_model)
        self.horizon = int(horizon)
        self.out_feat = int(out_feat) if out_feat is not None else self.in_feat

        # Temporal branch: create dynamically for current N_mic
        self.feat_proj_t = None  # Lazy init: Linear(N_mic*F -> D)

        # Spatial branch: per-node F -> D projection is done in SpatialEncoder
        self.temporal = TemporalEncoder(self.d_model, nhead=nhead, tcn_layers=tcn_layers, dropout=dropout)
        self.spatial  = SpatialEncoder(in_ch=self.in_feat, d_model=self.d_model, eta=eta, beta=beta, dropout=dropout)
        self.cross    = CrossAttnSync(self.d_model, nhead=nhead, dropout=dropout)

        # Prediction head: decode per node (multi-step, multi-dim)
        self.pred_node = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.horizon * self.out_feat)
        )

        self._num_nodes_hint = num_nodes_hint
        self._short_kernel = int(short_kernel)

    # --- Utilities ---

    def _ensure_time_proj(self, N_mic: int, device):
        in_features = N_mic * self.in_feat
        if (self.feat_proj_t is None) or (self.feat_proj_t.in_features != in_features):
            self.feat_proj_t = nn.Linear(in_features, self.d_model).to(device)

    @staticmethod
    def _gather_nodes(X: torch.Tensor, node_index: torch.Tensor) -> torch.Tensor:
        # X: [B,T,N_all,F], node_index:[N_mic] → [B,T,N_mic,F]
        return X.index_select(dim=2, index=node_index)

    def _build_short_windows(self, X_sel: torch.Tensor, Zm_sel: torch.Tensor, S: int):
        """
        Build short windows without loops:
        X_sel:[B,T,N,F], Zm_sel:[B,T,N]
        Returns:
          X_short_t:[B,T,N,S,F], M_short_t:[B,T,N,S]
        """
        B, T, N, F = X_sel.shape
        device = X_sel.device
        S = max(1, int(S))
        # Left pad S-1
        pads_X = torch.zeros(B, S - 1, N, F, device=device, dtype=X_sel.dtype)
        pads_M = torch.ones(B, S - 1, N, device=device, dtype=Zm_sel.dtype)
        X_pad = torch.cat([pads_X, X_sel], dim=1)   # [B, T+S-1, N, F]
        M_pad = torch.cat([pads_M, Zm_sel], dim=1)  # [B, T+S-1, N]

        # Build [T,S] indices: for each t take t..t+S-1
        idx_t = torch.arange(T, device=device).unsqueeze(1) + torch.arange(S, device=device).unsqueeze(0)  # [T,S]

        X_short_t = X_pad[:, idx_t]                     # [B,T,S,N,F]
        M_short_t = M_pad[:, idx_t]                     # [B,T,S,N]
        X_short_t = X_short_t.permute(0, 1, 3, 2, 4)    # [B,T,N,S,F]
        M_short_t = M_short_t.permute(0, 1, 3, 2)       # [B,T,N,S]
        return X_short_t, M_short_t

    # --- Forward ---

    def forward(self, X: torch.Tensor, A_stat: torch.Tensor, M_mask: torch.Tensor,
                node_index: Optional[torch.Tensor] = None, short_patch: int = 5,
                Zmask: Optional[torch.Tensor] = None):
        """
        X: [B,T,N_all,F]
        A_stat, M_mask: [N,N]
        Zmask (optional): [B,T,N_all] 1=normal, 0=anomalous/missing
        """
        B, T, N_all, F = X.shape
        device = X.device
        N = A_stat.size(0)
        if node_index is None:
            node_index = torch.arange(N, device=device, dtype=torch.long)
        if Zmask is None:
            Zmask = torch.ones(B, T, N_all, device=device, dtype=X.dtype)

        # ---------- Temporal branch ----------
        X_sel = self._gather_nodes(X, node_index)                  # [B,T,N,F]
        Xt = X_sel.reshape(B, T, -1)                               # [B,T,N*F]
        self._ensure_time_proj(N, device)
        Ht_in = self.feat_proj_t(Xt)                               # [B,T,D]
        # Per-time-step cleanliness m_t (mean over nodes)
        m_t = Zmask.index_select(2, node_index).float().mean(dim=2)  # [B,T]
        Ht, m_t_out = self.temporal(Ht_in, m_t)                    # [B,T,D], [B,T]

        # ---------- Spatial branch (vectorized) ----------
        S = int(short_patch) if short_patch else self._short_kernel
        X_short_t, M_short_t = self._build_short_windows(X_sel, Zmask.index_select(2, node_index).float(), S)
        node_rel = M_short_t.float().mean(dim=3)                   # [B,T,N]
        Hs_t, A_fuse_t = self.spatial(X_short_t, M_short_t.float(), A_stat, M_mask, node_rel)  # [B,T,N,D], [B,T,N,N]

        # ---------- Cross-branch sync ----------
        Ht_sync, Hs_sync_t = self.cross(Ht, Hs_t, m_t_out)         # [B,T,D], [B,T,N,D]

        # ---------- Prediction head (last step) ----------
        Hn_last = Hs_sync_t[:, -1]                                 # [B,N,D]
        y_flat = self.pred_node(Hn_last)                           # [B,N,H*out_feat]
        y_hat = y_flat.view(B, N, self.horizon, self.out_feat).permute(0, 2, 1, 3)  # [B,H,N,out_feat]

        aux = {
            "A_fuse": A_fuse_t[:, -1],    # last-step fused adjacency [B,N,N]
            "A_fuse_t": A_fuse_t,         # per-step fused adjacency [B,T,N,N]
            "Ht": Ht_sync,                # [B,T,D]
            "Hs_sync_last": Hn_last,      # [B,N,D]
            "node_index": node_index      # [N_mic]
        }
        return y_hat, aux
