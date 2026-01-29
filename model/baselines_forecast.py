# -*- coding: utf-8 -*-
# model/baselines_forecast.py
# Notes:
# - Six pure time-series forecasting baselines: LSTM / GRU / TCN / MLP / CNNLSTM / Transformer
# - Unified interface: forward(x, A=None, M=None) -> (y_hat, extras)
#   Input x: [B, T, N, 1]; Output y_hat: [B, 1, N, 1]
# - Graph info (A, M) is not used, but the arguments remain for compatibility

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_bt_n(x: torch.Tensor) -> torch.Tensor:
    # [B, T, N, 1] -> [B, T, N]
    if x.dim() != 4 or x.size(-1) != 1:
        raise ValueError(f"expect x [B,T,N,1], got {tuple(x.shape)}")
    return x[..., 0]


def _to_out(y: torch.Tensor) -> torch.Tensor:
    # [B, N] -> [B, 1, N, 1]
    return y[:, None, :, None]


class LSTMForecaster(nn.Module):
    def __init__(self, num_nodes: int, lookback: int, d_model: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.lookback = lookback
        self.proj_in = nn.Linear(num_nodes, d_model)
        self.rnn = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=num_layers,
                           dropout=(dropout if num_layers > 1 else 0.0), batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_nodes)
        )

    def forward(self, x: torch.Tensor, A: Optional[torch.Tensor] = None, M: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, dict]:
        xt = _to_bt_n(x)               # [B,T,N]
        h = self.proj_in(xt)           # [B,T,d]
        out, _ = self.rnn(h)           # [B,T,d]
        y = self.head(out[:, -1, :])   # [B,N]
        return _to_out(y), {}          # [B,1,N,1], extras={}


class GRUForecaster(nn.Module):
    def __init__(self, num_nodes: int, lookback: int, d_model: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.lookback = lookback
        self.proj_in = nn.Linear(num_nodes, d_model)
        self.rnn = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=num_layers,
                          dropout=(dropout if num_layers > 1 else 0.0), batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_nodes)
        )

    def forward(self, x: torch.Tensor, A: Optional[torch.Tensor] = None, M: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, dict]:
        xt = _to_bt_n(x)               # [B,T,N]
        h = self.proj_in(xt)           # [B,T,d]
        out, _ = self.rnn(h)           # [B,T,d]
        y = self.head(out[:, -1, :])   # [B,N]
        return _to_out(y), {}


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Trim right padding to preserve causality
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU(inplace=True)
        self.do1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU(inplace=True)
        self.do2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x); y = self.chomp1(y); y = self.relu1(y); y = self.do1(y)
        y = self.conv2(y); y = self.chomp2(y); y = self.relu2(y); y = self.do2(y)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(y + res)


class TCNForecaster(nn.Module):
    def __init__(self, num_nodes: int, lookback: int, d_model: int = 128, levels: int = 4,
                 kernel_size: int = 9, dropout: float = 0.15):
        super().__init__()
        # Treat input as N channels with length T: Conv1d over time
        layers = []
        ch_in = num_nodes
        for i in range(levels):
            ch_out = d_model
            layers += [TemporalBlock(ch_in, ch_out, kernel_size=kernel_size,
                                     dilation=2 ** i, dropout=dropout)]
            ch_in = ch_out
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(d_model, num_nodes)

    def forward(self, x: torch.Tensor, A: Optional[torch.Tensor] = None, M: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, dict]:
        xt = _to_bt_n(x)           # [B,T,N]
        xt = xt.transpose(1, 2)    # [B,N,T]
        h = self.tcn(xt)           # [B,d_model,T]
        last = h[:, :, -1]         # [B,d_model]
        y = self.head(last)        # [B,N]
        return _to_out(y), {}


class MLPForecaster(nn.Module):
    def __init__(self, num_nodes: int, lookback: int, d_model: int = 128, dropout: float = 0.15):
        super().__init__()
        self.num_nodes = num_nodes
        self.lookback = lookback
        in_dim = lookback * num_nodes
        h1 = max(128, 4 * d_model)
        h2 = max(64, 2 * d_model)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(h1, h2),      nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(h2, num_nodes)
        )

    def forward(self, x: torch.Tensor, A: Optional[torch.Tensor] = None, M: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, dict]:
        xt = _to_bt_n(x)                 # [B,T,N]
        y = self.net(xt.reshape(xt.size(0), -1))  # [B,N]
        return _to_out(y), {}


# -------- New: CNN + LSTM --------
class CNNLSTMForecaster(nn.Module):
    """
    Architecture: Conv1d (time) → ReLU/Dropout → (stackable) → transpose to sequence → LSTM → MLP for [B, N]
    - Treat N as channels, convolve over time to extract features, then model long dependencies with LSTM
    """
    def __init__(self, num_nodes: int, lookback: int, d_model: int = 128,
                 conv_levels: int = 2, kernel_size: int = 9, dropout: float = 0.15,
                 rnn_layers: int = 2):
        super().__init__()
        ks = int(kernel_size)
        pad = (ks - 1) // 2
        ch_in = num_nodes
        convs = []
        for _ in range(max(1, conv_levels)):
            conv = nn.Conv1d(ch_in, d_model, kernel_size=ks, padding=pad)
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            convs += [conv, nn.ReLU(inplace=True), nn.Dropout(dropout)]
            ch_in = d_model
        self.conv = nn.Sequential(*convs)
        self.rnn = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=max(1, rnn_layers),
                           dropout=(dropout if rnn_layers > 1 else 0.0), batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_nodes)
        )

    def forward(self, x: torch.Tensor, A: Optional[torch.Tensor] = None, M: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, dict]:
        xt = _to_bt_n(x)            # [B,T,N]
        h = xt.transpose(1, 2)      # [B,N,T]
        h = self.conv(h)            # [B,d_model,T]
        h = h.transpose(1, 2)       # [B,T,d_model]
        out, _ = self.rnn(h)        # [B,T,d_model]
        y = self.head(out[:, -1])   # [B,N]
        return _to_out(y), {}


# -------- New: Transformer --------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,d]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class TransformerForecaster(nn.Module):
    """
    Treat each time step as a token, project N features to d_model, add sinusoidal positional encoding,
    pass through TransformerEncoder, then take the last token and regress to N with an MLP.
    """
    def __init__(self, num_nodes: int, lookback: int, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 4, dropout: float = 0.1, dim_ff: Optional[int] = None):
        super().__init__()
        self.num_nodes = num_nodes
        self.lookback = lookback
        ff = dim_ff if dim_ff is not None else max(256, 4 * d_model)
        self.proj_in = nn.Linear(num_nodes, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=max(1, int(nhead)), dim_feedforward=ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, int(num_layers)))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_nodes)
        )

    def forward(self, x: torch.Tensor, A: Optional[torch.Tensor] = None, M: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, dict]:
        xt = _to_bt_n(x)                 # [B,T,N]
        h = self.proj_in(xt)             # [B,T,d]
        h = self.pos(h)                  # [B,T,d]
        h = self.encoder(h)              # [B,T,d]
        y = self.head(h[:, -1, :])       # [B,N]
        return _to_out(y), {}


def build_baseline(model_type: str, num_nodes: int, lookback: int, d_model: int,
                   tcn_layers: int, short_kernel: int, dropout: float, nhead: int = 8) -> nn.Module:
    mt = (model_type or "lstm").lower()
    if mt == "lstm":
        return LSTMForecaster(num_nodes, lookback, d_model=d_model, num_layers=2, dropout=dropout)
    if mt == "gru":
        return GRUForecaster(num_nodes, lookback, d_model=d_model, num_layers=2, dropout=dropout)
    if mt == "tcn":
        return TCNForecaster(num_nodes, lookback, d_model=d_model, levels=tcn_layers,
                             kernel_size=short_kernel, dropout=dropout)
    if mt == "mlp":
        return MLPForecaster(num_nodes, lookback, d_model=d_model, dropout=dropout)
    if mt == "cnnlstm":
        return CNNLSTMForecaster(num_nodes, lookback, d_model=d_model,
                                 conv_levels=max(1, tcn_layers), kernel_size=short_kernel,
                                 dropout=dropout, rnn_layers=2)
    if mt == "transformer":
        return TransformerForecaster(num_nodes, lookback, d_model=d_model,
                                     nhead=nhead, num_layers=max(1, tcn_layers),
                                     dropout=dropout)
    raise ValueError(f"Unknown model_type={model_type}")
