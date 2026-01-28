# -*- coding: utf-8 -*-
# =============================================
# ST-Graph-TCN (enhanced)
# - DropPath (stochastic depth) for residual branches
# - DropNode (node-level dropout, training only)
# - Dynamic graph: temperature scaling, symmetrization, remove self-loops
# - Return full dynamic graph for entropy regularization (high entropy = anti-overfitting)
# - Heteroscedastic head: outputs μ and logσ² (optional)
#   forward(x, A, M) -> y_hat:[B,H,N,1], aux:{pred_logv?, A_fuse, A_fuse_t}
# =============================================
from __future__ import annotations
import math
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------ utils ------------------
class DropPath(nn.Module):
    """Stochastic Depth for residual branches."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


def causal_pad(x: torch.Tensor, kernel: int, dilation: int) -> torch.Tensor:
    pad = (kernel - 1) * dilation
    return F.pad(x, (pad, 0))


# ------------------ Temporal TCN ------------------
class TCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int = 5, dilation: int = 1,
                 dropout: float = 0.1, droppath: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel, dilation=dilation)
        self.bn = nn.BatchNorm1d(c_out)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()
        self.dp = DropPath(droppath)
        self.kernel = kernel
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, C_in, T]
        y = causal_pad(x, self.kernel, self.dilation)
        y = self.conv(y)
        y = self.bn(y)
        y = F.gelu(y)
        y = self.dropout(y)
        res = self.res_proj(x)
        return F.gelu(res + self.dp(y))


class TemporalEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, kernel: int = 7, dropout: float = 0.1,
                 droppath: float = 0.0):
        super().__init__()
        layers = []
        c = d_model
        for i in range(num_layers):
            layers.append(
                TCNBlock(c, c, kernel=kernel, dilation=2 ** i,
                         dropout=dropout, droppath=droppath * 0.5)
            )
        self.net = nn.Sequential(*layers)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: [B, T, N, C]
        B, T, N, C = H.shape
        x = H.permute(0, 2, 3, 1).contiguous().view(B * N, C, T)
        y = self.net(x)
        y = y.view(B, N, C, T).permute(0, 3, 1, 2).contiguous()  # [B,T,N,C]
        return y


# ------------------ Dynamic Graph ------------------
class DynamicGraphAttention(nn.Module):
    """Attention-based dynamic adjacency per timestep."""
    def __init__(self, d_model: int, eta: float = 1.0, tau: float = 1.5, symmetric: bool = True):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.eta = float(eta)
        self.tau = float(tau)
        self.symmetric = bool(symmetric)

    def forward(self, H_t: torch.Tensor, A_stat: torch.Tensor, M_mask: torch.Tensor) -> torch.Tensor:
        # H_t: [B,N,C]; A_stat/M_mask: [N,N]
        B, N, C = H_t.shape
        Q = self.q(H_t)
        K = self.k(H_t)
        logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(C)
        logits = logits / self.tau
        if self.symmetric:
            logits = 0.5 * (logits + logits.transpose(-1, -2))
        # forbid self-loop
        eye = torch.eye(N, device=H_t.device, dtype=torch.bool)
        logits = logits.masked_fill(eye, float('-inf'))
        # bias towards static edges (no self-loop)
        if A_stat is not None:
            A_bias = A_stat.clone()
            A_bias = A_bias.to(H_t.device, dtype=logits.dtype)
            A_bias.fill_diagonal_(0.0)
            logits = logits + self.eta * A_bias.unsqueeze(0)
        # mask invalid edges
        if M_mask is not None:
            mask = (M_mask <= 0).to(torch.bool).unsqueeze(0).expand_as(logits)
            logits = logits.masked_fill(mask, float('-inf'))
        A_dyn = F.softmax(logits, dim=-1)  # row-stochastic
        return A_dyn  # [B,N,N]


class SpatialEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, eta: float = 1.0, tau: float = 1.5):
        super().__init__()
        self.dyn = DynamicGraphAttention(d_model, eta=eta, tau=tau, symmetric=True)
        self.lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, H: torch.Tensor, A_stat: torch.Tensor, M_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # H: [B,T,N,C]
        B, T, N, C = H.shape
        Adyn_t = []
        H_new = []
        for t in range(T):
            Ht = H[:, t]                # [B,N,C]
            Adyn = self.dyn(Ht, A_stat, M_mask)  # [B,N,N]
            Y = torch.matmul(Adyn, Ht)  # [B,N,C]
            Y = self.lin(self.dropout(Y))
            Ht2 = self.ln(Ht + Y)
            H_new.append(Ht2.unsqueeze(1))
            Adyn_t.append(Adyn.unsqueeze(1))
        H_out = torch.cat(H_new, dim=1)            # [B,T,N,C]
        Adyn_all = torch.cat(Adyn_t, dim=1)        # [B,T,N,N]
        A_fuse = Adyn_all[:, -1]                   # [B,N,N]
        return H_out, A_fuse, Adyn_all


# ------------------ Main Model ------------------
class STGraphTCN(nn.Module):
    def __init__(self,
                 in_feat: int = 1,
                 out_feat: int = 1,
                 d_model: int = 64,
                 tcn_layers: int = 4,
                 short_kernel: int = 9,
                 dropout: float = 0.15,
                 horizon: int = 1,
                 eta_bias: float = 1.0,
                 dyn_tau: float = 1.5,
                 node_drop_p: float = 0.05,
                 pred_uncert: bool = True):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.d_model = d_model
        self.horizon = horizon
        self.pred_uncert = bool(pred_uncert)
        self.node_drop_p = float(node_drop_p)

        self.in_proj = nn.Linear(in_feat, d_model)
        self.temporal = TemporalEncoder(d_model, num_layers=tcn_layers,
                                        kernel=short_kernel, dropout=dropout, droppath=dropout*0.5)
        self.spatial = SpatialEncoder(d_model, dropout=dropout, eta=eta_bias, tau=dyn_tau)
        self.norm = nn.LayerNorm(d_model)

        out_dim = horizon * out_feat * (2 if self.pred_uncert else 1)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, out_dim)
        )

    def _apply_dropnode(self, X_sel: torch.Tensor) -> torch.Tensor:
        if self.training and self.node_drop_p > 0.0:
            B, T, N, F = X_sel.shape
            keep = 1.0 - self.node_drop_p
            mask = X_sel.new_empty(B, 1, N, 1).bernoulli_(keep).div_(keep)
            X_sel = X_sel * mask
        return X_sel

    def forward(self, x: torch.Tensor, A: torch.Tensor, M: torch.Tensor):
        """
        x: [B, T, N, in_feat]
        A: [N, N] static adjacency (0..1)
        M: [N, N] mask (1 valid, 0 invalid)
        """
        B, T, N, F = x.shape
        # input proj
        h = self.in_proj(x)         # [B,T,N,C]
        h = self._apply_dropnode(h)
        # temporal
        h = self.temporal(h)        # [B,T,N,C]
        # spatial message passing per step
        h, A_fuse, Adyn_t = self.spatial(h, A, M)  # [B,T,N,C], [B,N,N], [B,T,N,N]
        # last-step representation to predict next step
        h_last = self.norm(h[:, -1])               # [B,N,C]
        y = self.pred_head(h_last)                 # [B,N, H*out_feat*(1 or 2)]
        if self.pred_uncert:
            y = y.view(B, N, self.horizon, self.out_feat, 2)
            mu = y[..., 0]
            logv = y[..., 1]
            mu = mu.permute(0, 2, 1, 3).contiguous()     # [B,H,N,out]
            logv = logv.permute(0, 2, 1, 3).contiguous()
            aux: Dict[str, torch.Tensor] = {"A_fuse": A_fuse, "A_fuse_t": Adyn_t, "pred_logv": logv}
            return mu, aux
        else:
            y = y.view(B, N, self.horizon, self.out_feat)
            y = y.permute(0, 2, 1, 3).contiguous()       # [B,H,N,out]
            aux = {"A_fuse": A_fuse, "A_fuse_t": Adyn_t}
            return y, aux
