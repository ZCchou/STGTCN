# -*- coding: utf-8 -*-
"""
UAV Forecast + Anomaly Detection (GPS Attack, Failure 30/70 Split)
- Data directories:
    gpsdata/No_Failure/*.csv        (Benign: train/val; if only 1 file, split by time)
    gpsdata/Failure/*.csv           (per file: first 30% after crop for training; last 70% test only)
- Graph directory: out_static_graph_gpsatt (output of GraphBuild_static_gpsatt.py)
- Label column: label (0/1)
- Timestamp column: timestamp (not used as a feature)
- Output directory: gpsresult/{figures, checkpoints}

Key points:
1) Accept keep_columns.json as list or dict (auto-ignore 'timestamp'/'label').
2) Global head-crop is consistent across train/val/test; Failure 30/70 split uses post-crop indices.
3) Standardization and threshold calibration use training segments only (Benign train + Failure first 30%) to avoid leakage.
4) Evaluation is only on the last 70% of Failure segments with aligned labels.
"""

import os, glob, json, math, random, re, warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.backends.cuda.sdp_kernel.*")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Graph temporal model
from model.st_graph_tcn_pred import STGraphTCN

# ========= Global head crop (consistent for train/val/test) =========
UAD_GLOBAL_HEAD_CROP: int = 1000

DEBUG_NUMERICS: bool = True
HARD_FAIL_ON_NONFINITE: bool = True
NAN_REPLACE = dict(nan=0.0, posinf=1e6, neginf=-1e6)

def uad_set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def uad_makedirs(p: str, exist_ok=True): os.makedirs(p, exist_ok=exist_ok)

def uad_list_csvs(d: str) -> List[str]:
    return sorted(glob.glob(os.path.join(d, "*.csv"))) if d and os.path.isdir(d) else []

def uad_read_lines(fp: str) -> List[str]:
    if not os.path.exists(fp): return []
    with open(fp, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

def uad_configure_sdpa_kernels():
    try:
        from torch.nn.attention import sdpa_kernel
        sdpa_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True)
    except Exception:
        try:
            if torch.cuda.is_available() and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
                torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True)
        except Exception:
            pass
uad_configure_sdpa_kernels()

# ========= Configuration =========
@dataclass
class UADConfig:
    # Data and graph
    uad_use_all_nofail: bool = True
    uad_nofail_dir: str = "gpsdata/No_Failure"
    uad_fail_dir: str = "gpsdata/Failure"
    uad_graph_dir: str = "out_static_graph_gpsatt"

    # Benign single-file split ratio (only when No_Failure has 1 file)
    uad_val_ratio: float = 0.20

    # Training parameters
    uad_lookback: int = 32
    uad_horizon: int = 1
    uad_stride: int = 1
    uad_batch: int = 64
    uad_epochs: int = 200
    uad_patience: int = 10
    uad_lr: float = 5e-4
    uad_weight_decay: float = 3e-4
    uad_loss: str = "huber"        # "mae" | "mse" | "huber"
    uad_huber_delta: float = 1.0
    uad_lambda_delta: float = 0.2

    # Model capacity
    uad_d_model: int = 64
    uad_nhead: int = 8
    uad_tcn_layers: int = 4
    uad_short_kernel: int = 9
    uad_eta_bias: float = 2.0
    uad_beta_fuse: float = 0.5
    uad_dropout: float = 0.15
    uad_horizon_out: int = 1
    uad_use_dynamic_graph: bool = True

    # Visualization output directory (gpsresult)
    uad_topk_nodes_viz: int = 12
    uad_fig_dir: str = os.path.join("gpsresult", "figures")

    # Misc
    uad_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    uad_ckpt_dir: str = os.path.join("gpsresult", "checkpoints")
    uad_seed: int = 42

    # Input smoothing
    uad_enable_smooth: bool = True
    uad_smooth_method: str = "ema"   # "ema" | "ma" | "savgol"
    uad_ema_alpha: float = 0.1
    uad_ma_window: int = 5
    uad_savgol_window: int = 9
    uad_savgol_poly: int = 3
    uad_causal_smooth: bool = True

    # Residual aggregation and thresholds
    uad_use_delta_residual: bool = True
    uad_topk_ratio: float = 0.5
    uad_thr_mode: str = "mad"        # "mad" | "quantile" | "gauss" | "gev"
    uad_thr_quantile: float = 0.995

    # Post-processing
    uad_post_min_run: int = 0
    uad_post_gap: int = 0

    # Label info
    uad_label_col: str = "label"
    uad_label_shift: int = 0

    # Disable extra test head-crop (avoid overlap with 30/70 split)
    uad_test_crop_head: int = 0

    # Dimensional robustness
    uad_dim_filter_mode: str = "skill"  # "skill" | "topq" | "off"
    uad_dim_filter_topq: float = 0.10
    uad_dim_filter_min_keep: int = 24
    uad_dim_weight_mode: str = "none" # "inv_mad" | "none"
    uad_norm_mode: str = "zscore"        # "zscore" | "none"

    # Angle unwrapping and delta residuals
    uad_angle_unwrap: bool = True
    uad_angle_regex: str = r"(heading|yaw|psi)"
    uad_delta_regex: str = r"(wp_dist|alt_error|aspd_error)"

    # Score smoothing
    uad_score_smooth: bool = True
    uad_score_smooth_method: str = "ema"  # "bi_ewma" | "ema" | "ma" | "savgol"
    uad_score_causal: bool = True
    uad_score_ewma_alpha: float = 0.25
    uad_score_ma_window: int = 9
    uad_score_savgol_window: int = 11
    uad_score_savgol_poly: int = 3

    # Top-k aggregator
    uad_topk_agg: str = "trimmed_mean"  # "mean" | "median" | "trimmed_mean"
    uad_topk_trim_ratio: float = 0.15
    uad_topk_trim_high_only: bool = True

    # Per-file adaptive threshold
    uad_enable_file_thr: bool = True
    uad_file_pct: float = 0.25
    uad_file_thr_mode: str = "mad"   # "mad" | "gauss"
    uad_file_thr_k: float = 5
    uad_file_thr_combine: str = "max" # "max" | "mean" | "none"

    # MixUp & EMA & LR
    uad_mixup_alpha: float = 0.2
    uad_ema_decay: float = 0.999
    uad_sched_type: str = "cosine"  # "cosine" | "plateau"
    uad_warmup_epochs: int = 3

    # Per-file adaptive normalization (AdaNorm)
    uad_enable_adaptnorm: bool = True
    uad_adaptnorm_alpha: float = 0.7

# ========= Smoothing =========
def uad_input_smooth(df: pd.DataFrame, cfg: UADConfig) -> pd.DataFrame:
    if not cfg.uad_enable_smooth: return df
    m = (cfg.uad_smooth_method or "ema").lower()
    if m == "ema":
        return df.ewm(alpha=cfg.uad_ema_alpha, adjust=False).mean()
    if m == "ma":
        win = max(1, int(cfg.uad_ma_window))
        return df.rolling(win, min_periods=1, center=not cfg.uad_causal_smooth).mean()
    if m == "savgol":
        try:
            from scipy.signal import savgol_filter
            wl = int(cfg.uad_savgol_window) | 1; po = int(cfg.uad_savgol_poly)
            arr = savgol_filter(df.values, window_length=wl, polyorder=po, axis=0, mode="interp")
            return pd.DataFrame(arr, columns=df.columns, index=df.index)
        except Exception:
            return df.ewm(alpha=cfg.uad_ema_alpha, adjust=False).mean()
    return df

def _ema_1d(x: np.ndarray, a: float) -> np.ndarray:
    if len(x)==0: return x
    y = np.empty_like(x, dtype=np.float64); y[0] = float(x[0]); b=1.0-a
    for i in range(1, len(x)): y[i] = a*float(x[i]) + b*y[i-1]
    return y.astype(np.float32)

def _bi_ewma(x: np.ndarray, a: float) -> np.ndarray:
    if len(x)==0: return x
    f = _ema_1d(x, a); b = _ema_1d(x[::-1], a)[::-1]
    return (0.5*(f+b)).astype(np.float32)

def _ma(x: np.ndarray, win: int, causal=True) -> np.ndarray:
    if len(x)==0: return x
    w = max(1, int(win))
    if causal:
        y = np.empty_like(x, dtype=np.float64); csum = np.cumsum(x.astype(np.float64))
        for i in range(len(x)):
            j0 = max(0, i-w+1); s = csum[i] - (csum[j0-1] if j0>0 else 0.0); y[i] = s/max(1, i-j0+1)
        return y.astype(np.float32)
    k = np.ones(w)/w; return np.convolve(x.astype(np.float64), k, mode="same").astype(np.float32)

def _savgol(x: np.ndarray, win: int, poly: int) -> np.ndarray:
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(x.astype(np.float64), window_length=(int(win)|1), polyorder=int(poly), mode="interp").astype(np.float32)
    except Exception:
        return _bi_ewma(x, 0.25)

def uad_score_smooth(scores: np.ndarray, cfg: UADConfig) -> np.ndarray:
    if not cfg.uad_score_smooth: return scores.astype(np.float32)
    m = (cfg.uad_score_smooth_method or "bi_ewma").lower()
    if m=="ema": return _ema_1d(scores, cfg.uad_score_ewma_alpha)
    if m in ("bi_ewma","bi-ema","biema"): return _bi_ewma(scores, cfg.uad_score_ewma_alpha)
    if m=="ma": return _ma(scores, cfg.uad_score_ma_window, cfg.uad_score_causal)
    if m=="savgol": return _savgol(scores, cfg.uad_score_savgol_window, cfg.uad_score_savgol_poly)
    return scores.astype(np.float32)

# ========= Column preprocessing =========
def _unwrap_angles(series: pd.Series) -> pd.Series:
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if np.nanmax(np.abs(v)) > 3.5: v = np.deg2rad(v)
    v = np.unwrap(v)
    return pd.Series(v, index=series.index)

def uad_preprocess_cols(df: pd.DataFrame, cols: List[str], cfg: UADConfig) -> pd.DataFrame:
    Xdf = df[cols].copy()
    if cfg.uad_angle_unwrap:
        pat = re.compile(cfg.uad_angle_regex, re.IGNORECASE)
        for c in [c for c in cols if pat.search(c)]:
            try: Xdf[c] = _unwrap_angles(Xdf[c])
            except Exception: pass
    return Xdf

# ========= Standardizer =========
class UADStandardizer:
    def __init__(self): self.mu=None; self.sd=None
    def fit(self, X: np.ndarray):
        mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0)
        mu = np.nan_to_num(mu, nan=0.0); sd = np.nan_to_num(sd, nan=0.0); sd[sd<1e-8]=1e-8
        self.mu, self.sd = mu, sd
    def transform(self, X: np.ndarray):   return (X - self.mu) / self.sd
    def inverse(self, Xn: np.ndarray):     return Xn*self.sd + self.mu

# ========= Graph I/O =========
def uad_load_keep_columns(graph_dir: str) -> Optional[List[str]]:
    p = os.path.join(graph_dir, "keep_columns.json")
    if not os.path.exists(p): return None
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)

    ks = None
    if isinstance(obj, list):
        ks = obj
    elif isinstance(obj, dict):
        for key in ("keep_columns", "columns", "names", "cols", "keep"):
            if key in obj and isinstance(obj[key], (list, tuple)):
                ks = list(obj[key]); break
        if ks is None:
            flat = []
            for v in obj.values():
                if isinstance(v, str):
                    flat.append(v)
                elif isinstance(v, (list, tuple)):
                    flat.extend([x for x in v if isinstance(x, str)])
            ks = flat if flat else None

    if not ks: return None
    seen, out = set(), []
    for c in ks:
        c = str(c)
        if c.lower() in ("timestamp", "label"):
            continue
        if c not in seen:
            seen.add(c); out.append(c)
    return out if out else None

def uad_load_graph(graph_dir: str):
    nodes_fp      = os.path.join(graph_dir, "nodes.csv")
    adj_dense_fp  = os.path.join(graph_dir, "adjacency_dense.csv")
    adj_sparse_fp = os.path.join(graph_dir, "A_global_sparse.csv")

    keep_cols = uad_load_keep_columns(graph_dir)
    if keep_cols is None:
        if not os.path.exists(nodes_fp):
            raise FileNotFoundError("Missing keep_columns.json or nodes.csv")
        df_nodes = pd.read_csv(nodes_fp)
        if "name" in df_nodes.columns: keep_cols = df_nodes["name"].astype(str).tolist()
        elif "label" in df_nodes.columns: keep_cols = df_nodes["label"].astype(str).tolist()
        else: keep_cols = [f"n{i}" for i in range(len(df_nodes))]

    keep_cols = [c for c in keep_cols if str(c).lower() not in ("timestamp","label")]
    cols = list(map(str, keep_cols)); N = len(cols)

    A=None
    if os.path.exists(adj_dense_fp):
        try:
            dfA = pd.read_csv(adj_dense_fp, index_col=0, low_memory=False)
            dfA.index = dfA.index.astype(str); dfA.columns=dfA.columns.astype(str)
            for c in cols:
                if c not in dfA.index: dfA.loc[c]=0.0
                if c not in dfA.columns: dfA[c]=0.0
            dfA = dfA.reindex(index=cols, columns=cols).fillna(0.0)
            A = dfA.values.astype(np.float32)
        except Exception:
            try:
                A = pd.read_csv(adj_dense_fp, header=None, low_memory=False).values.astype(np.float32)
                k = min(N, A.shape[0], A.shape[1]); A=A[:k,:k]; cols=cols[:k]; N=k
            except Exception: A=None

    if A is None:
        if not os.path.exists(adj_sparse_fp): raise FileNotFoundError(f"Missing: {adj_dense_fp} & {adj_sparse_fp}")
        dfs = pd.read_csv(adj_sparse_fp)  # row,col,val
        A = np.zeros((N,N), dtype=np.float32)
        r = np.asarray(dfs.get("row", dfs.columns[0]).values, dtype=int)
        c = np.asarray(dfs.get("col", dfs.columns[1]).values, dtype=int)
        v = np.asarray(dfs.get("val", dfs.columns[2]).values, dtype=np.float32)
        r = np.clip(r,0,N-1); c = np.clip(c,0,N-1)
        A[r,c]=v; A[c,r]=v

    np.fill_diagonal(A, 0.0)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    if A.max()>0: A = A/(A.max()+1e-6)
    M = (A>0).astype(np.float32)

    try:
        deg=(A>0).astype(np.float32).sum(axis=1); topk_order=list(np.argsort(-deg))
    except Exception:
        topk_order=list(range(N))
    print(f"[DBG] keep_columns used: {len(cols)}; head={cols[:5]}")
    print(f"[DBG] A shape={A.shape}, M shape={M.shape}, A.max={float(np.max(A)):.4f}")
    return cols, torch.from_numpy(A.astype(np.float32)), torch.from_numpy(M.astype(np.float32)), topk_order

# ========= Dataset =========
class UADDataset(Dataset):
    def __init__(self, files: List[str], cols: List[str], scaler: Optional[UADStandardizer],
                 lookback: int, horizon: int, stride: int = 1, cfg: Optional[UADConfig] = None,
                 ranges: Optional[Dict[str, Tuple[int,int]]] = None):
        self.files=list(files); self.cols=list(cols); self.scaler=scaler
        self.lb=int(lookback); self.hz=int(horizon); self.stride=max(1,int(stride))
        self.cfg = cfg if cfg is not None else UADConfig()
        self.ranges = ranges or {}
        self.index=[]; self.cache=[]
        self._build_index()

    def _load_one(self, fp: str):
        df = pd.read_csv(fp)
        miss=[c for c in self.cols if c not in df.columns]
        if miss: raise ValueError(f"[{os.path.basename(fp)}] Missing columns: {miss[:8]} ...")
        Xdf = uad_preprocess_cols(df, self.cols, self.cfg)
        Xdf = Xdf.apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan).ffill().bfill()
        Xdf = uad_input_smooth(Xdf, self.cfg)

        if len(Xdf) > UAD_GLOBAL_HEAD_CROP:
            Xdf = Xdf.iloc[UAD_GLOBAL_HEAD_CROP:, :]
        else:
            Xdf = Xdf.iloc[0:0, :]

        if self.scaler is not None and self.scaler.mu is not None:
            mu = pd.Series(self.scaler.mu, index=self.cols); Xdf = Xdf.fillna(mu)
        X = np.nan_to_num(Xdf.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if self.scaler is not None:
            X = np.nan_to_num(self.scaler.transform(X), nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(X).all():
            if DEBUG_NUMERICS:
                print(f"[WARN] non-finite in X (file={os.path.basename(fp)})")
            X = np.nan_to_num(X, **NAN_REPLACE)
        return X

    def _build_index(self):
        for fp in self.files:
            X=self._load_one(fp); T=X.shape[0]
            if T<(self.lb+self.hz): continue
            self.cache.append((X, os.path.basename(fp)))
            ci=len(self.cache)-1

            # If range provided, roll only within it (range indices are post head-crop)
            t_start, t_end = 0, T
            base = os.path.basename(fp)
            if base in self.ranges:
                t_start, t_end = self.ranges[base]
                t_start = max(0, min(t_start, T))
                t_end   = max(0, min(t_end, T))
                if t_end <= t_start: continue

            t0_begin = max(self.lb, t_start + self.lb)
            t0_end_ex = min(T - self.hz + 1, t_end - self.hz + 1)
            for t0 in range(t0_begin, t0_end_ex, self.stride):
                self.index.append((ci,t0))
        random.shuffle(self.index)

    def __len__(self): return len(self.index)
    def __getitem__(self, idx):
        ci,t0 = self.index[idx]; X,_ = self.cache[ci]
        x_win = X[t0-self.lb:t0]; y_next = X[t0+self.hz-1]
        return torch.from_numpy(x_win[...,None]), torch.from_numpy(y_next)

def uad_collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs,0), torch.stack(ys,0)

# ========= Residuals / aggregation / thresholds =========
def uad_robust_med_mad(E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    med = np.median(E, axis=0)
    mad = np.median(np.abs(E - med[None,:]), axis=0)
    mad = np.maximum(mad, 1e-12)
    return med.astype(np.float64), mad.astype(np.float64)

def uad_build_delta_mask(node_names: List[str], cfg: UADConfig) -> np.ndarray:
    if not cfg.uad_use_delta_residual:
        return np.zeros((len(node_names),), dtype=bool)
    pat = re.compile(cfg.uad_delta_regex, re.IGNORECASE)
    return np.array([bool(pat.search(nm)) for nm in node_names], dtype=bool)

def uad_perdim_errors(truth: np.ndarray, pred: np.ndarray, delta_mask: np.ndarray) -> np.ndarray:
    E = np.abs(truth - pred)
    if delta_mask.any() and truth.shape[0]>=2:
        dtruth = truth[1:] - truth[:-1]
        dpred  = pred[1:]  - pred[:-1]
        Ed = np.abs(dtruth - dpred)
        Ed = np.vstack([Ed[0:1,:], Ed])  # pad
        E[:, delta_mask] = Ed[:, delta_mask]
    return E

def uad_dim_keep_and_weights(E_model: np.ndarray, E_naive: np.ndarray, cfg: UADConfig
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    T,N = E_model.shape
    med, mad = uad_robust_med_mad(E_model)
    med_n, _ = uad_robust_med_mad(E_naive)
    skill_gain = med_n - med  # >0 better

    keep_mask = np.ones(N, dtype=bool)
    mode=(cfg.uad_dim_filter_mode or "skill").lower()
    if mode=="skill":
        keep_mask = skill_gain > 0.0
        if keep_mask.sum()<cfg.uad_dim_filter_min_keep:
            order = np.argsort(-skill_gain)
            keep_mask[:] = False
            keep_mask[order[:max(cfg.uad_dim_filter_min_keep,1)]] = True
    elif mode=="topq":
        q=min(max(float(cfg.uad_dim_filter_topq),0.0),0.9)
        drop_k=int(round(N*q))
        order_worst=np.argsort(-med)
        keep_mask[:] = True
        keep_mask[order_worst[:drop_k]] = False
        if keep_mask.sum()<cfg.uad_dim_filter_min_keep:
            add=cfg.uad_dim_filter_min_keep-keep_mask.sum()
            keep_mask[order_worst[drop_k:drop_k+add]] = True

    if (cfg.uad_dim_weight_mode or "inv_mad").lower()=="inv_mad":
        w = 1.0 / (mad + 1e-12)
        w = np.where(keep_mask, w, 0.0); s=w.sum()
        if s<=0: w = np.where(keep_mask,1.0,0.0); s=w.sum()
        w = (w/s).astype(np.float32)
    else:
        w = np.where(keep_mask,1.0,0.0).astype(np.float32); w=w/max(w.sum(),1e-12)

    stats=dict(med=med, mad=mad, med_naive=med_n, skill_gain=skill_gain)
    return keep_mask, w, stats

def uad_normalize_errors(E: np.ndarray, stats: Dict[str, np.ndarray], cfg: UADConfig) -> np.ndarray:
    if (cfg.uad_norm_mode or "zscore").lower()=="none": return E.astype(np.float32)
    med = stats["med"].astype(np.float64); mad = stats["mad"].astype(np.float64)
    scale = 1.4826*mad + 1e-12
    Z = (E.astype(np.float64) - med[None,:]) / scale[None,:]
    return np.abs(Z).astype(np.float32)

def _w_mean(vals: np.ndarray, w: Optional[np.ndarray]) -> np.ndarray:
    if w is None: return np.mean(vals, axis=1)
    s = np.sum(w, axis=1, keepdims=True) + 1e-12
    return np.sum(vals * (w/s), axis=1)

def _w_median_row(vals: np.ndarray, w: Optional[np.ndarray]) -> float:
    if w is None: return float(np.median(vals))
    w = w / (w.sum() + 1e-12)
    idx = np.argsort(vals)
    v = vals[idx]; ww = w[idx]
    c = np.cumsum(ww)
    j = int(np.searchsorted(c, 0.5))
    j = min(max(j,0), len(v)-1)
    return float(v[j])

def aggregate_topk_scores(E: np.ndarray, ratio: float, weights: Optional[np.ndarray],
                          keep_mask: Optional[np.ndarray], agg: str = "mean",
                          trim_ratio: float = 0.15, trim_high_only: bool = True) -> np.ndarray:
    T,N = E.shape
    if keep_mask is None: keep_mask = np.ones(N, dtype=bool)
    idx = np.where(keep_mask)[0]
    if len(idx)==0: return np.zeros((T,), dtype=np.float32)

    X = E[:, idx]
    Nk = X.shape[1]
    k = max(1, int(min(max(ratio,0.0),1.0)*Nk))
    part_idx = np.argpartition(X, Nk - k, axis=1)[:, Nk-k:]
    rows = np.arange(T)[:,None]
    top_vals = X[rows, part_idx]

    w_top = None
    if weights is not None:
        w = weights[idx]
        w_top = w[part_idx]

    agg = (agg or "mean").lower()
    if agg == "median":
        out = np.empty((T,), dtype=np.float32)
        if w_top is None:
            out[:] = np.median(top_vals, axis=1).astype(np.float32)
        else:
            for i in range(T):
                out[i] = _w_median_row(top_vals[i], w_top[i])
        return out

    if agg == "trimmed_mean":
        m = int(round(float(trim_ratio)*k))
        if m >= k: m = k-1
        if m > 0:
            order = np.argsort(top_vals, axis=1)  # ascending
            rows = np.arange(T)[:,None]
            if trim_high_only:
                pick = order[:, :k-m]
            else:
                l = m//2; r = m - l
                pick = order[:, l:k-r]
            top_vals = top_vals[rows, pick]
            if w_top is not None: w_top = w_top[rows, pick]
    return _w_mean(top_vals, w_top).astype(np.float32)

def uad_main_loss(y_pred, y_true, loss_type: str, huber_delta: float):
    if loss_type == "mse":   return F.mse_loss(y_pred, y_true)
    if loss_type == "huber": return F.smooth_l1_loss(y_pred, y_true, beta=huber_delta, reduction="mean")
    return F.l1_loss(y_pred, y_true)

# ========= EMA =========
class UADEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def _is_trackable(self, t: torch.Tensor) -> bool:
        return torch.is_floating_point(t)

    @torch.no_grad()
    def _sync_if_needed(self, model: nn.Module):
        if self.shadow: return
        for k, v in model.state_dict().items():
            if self._is_trackable(v):
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        if not self.shadow: self._sync_if_needed(model)
        cur = model.state_dict()
        for k, v in cur.items():
            if not self._is_trackable(v): continue
            vv = v.detach()
            if (k not in self.shadow) or (self.shadow[k].shape != vv.shape) or (self.shadow[k].dtype != vv.dtype):
                self.shadow[k] = vv.clone()
            else:
                self.shadow[k].mul_(self.decay).add_(vv, alpha=1.0 - self.decay)
        gone = [k for k in list(self.shadow.keys()) if k not in cur]
        for k in gone: self.shadow.pop(k, None)

    def store(self, model: nn.Module):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def copy_to(self, model: nn.Module):
        if not self.shadow: self._sync_if_needed(model)
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model: nn.Module):
        if self.backup is not None:
            model.load_state_dict(self.backup, strict=False)
            self.backup = None

# ========= LR schedule =========
class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs=3, min_lr_ratio=0.1, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            scale = (self.last_epoch+1)/max(1,self.warmup_epochs)
            return [base_lr*scale for base_lr in self.base_lrs]
        t = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        cos = 0.5*(1+math.cos(math.pi*min(max(t,0.0),1.0)))
        return [base_lr*(self.min_lr_ratio + (1-self.min_lr_ratio)*cos) for base_lr in self.base_lrs]

def uad_mixup_batch(x, y, alpha=0.4):
    if alpha is None or alpha <= 0: return x, y
    lam = np.random.beta(alpha, alpha)
    B = x.size(0)
    idx = torch.randperm(B, device=x.device)
    x2, y2 = x[idx], y[idx]
    return lam * x + (1 - lam) * x2, lam * y + (1 - lam) * y2

# ========= Plotting =========
def uad_plot_train_curves(hist, out_png):
    plt.figure(); plt.plot(hist["train"], label="train"); plt.plot(hist["val"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Forecast loss"); plt.legend()
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def uad_plot_forecast(times_idx, truth, preds, node_names, topk_order, K, out_png):
    if truth.shape[0]==0: return
    K=min(K, truth.shape[1]); pick=topk_order[:K]
    plt.figure(figsize=(12,2.2*K))
    for r,i in enumerate(pick, start=1):
        ax=plt.subplot(K,1,r)
        ax.plot(times_idx, truth[:,i], label=f"{node_names[i]} - gt")
        ax.plot(times_idx, preds[:,i],  label=f"{node_names[i]} - pred", alpha=0.9)
        ax.set_xlim(times_idx[0], times_idx[-1]); ax.grid(True, alpha=0.2)
        if r==1: ax.set_title("Forecast vs Ground Truth")
        if r==K: ax.set_xlabel("t")
        ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def uad_plot_residual(times_idx, residuals, labels, thr, out_png, thr_name="3σ"):
    plt.figure(figsize=(12,3))
    plt.plot(times_idx, residuals, label="residual")
    plt.axhline(thr, ls="--", color="r", label=thr_name)
    if labels is not None and len(labels)==len(times_idx):
        plt.fill_between(times_idx, 0, 1, where=(labels>0), color="r", alpha=0.15,
                         transform=plt.gca().get_xaxis_transform(), label="anomaly label")
    plt.legend(); plt.title(f"Residual timeline with {thr_name} threshold")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def uad_labels_from_file(file_path, length: int, lookback:int, horizon:int, label_col:str="label",
                         shift:int=0, extra_head_crop:int=0):
    df = pd.read_csv(file_path)
    if label_col in df.columns:
        y = df[label_col].astype(int).values
        start = UAD_GLOBAL_HEAD_CROP + extra_head_crop + lookback
        y = y[start:start+length]
        if shift != 0: y = np.roll(y, int(shift))
        return y
    return np.zeros((length,), dtype=np.int32)

# ========= Rolling forecast =========
@torch.no_grad()
def uad_rolling_forecast(model, file_path, cols, scaler, A, M, device, lookback, horizon, cfg: UADConfig,
                         crop_head_rows: int = 0):
    df = pd.read_csv(file_path)
    miss=[c for c in cols if c not in df.columns]
    if miss: raise ValueError(f"[{os.path.basename(file_path)}] Missing columns: {miss[:8]} ...")

    Xdf = uad_preprocess_cols(df, cols, cfg)
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan).ffill().bfill()
    Xdf = uad_input_smooth(Xdf, cfg)

    if len(Xdf) > UAD_GLOBAL_HEAD_CROP:
        Xdf = Xdf.iloc[UAD_GLOBAL_HEAD_CROP:, :]
    else:
        return np.empty((0, len(cols))), np.empty((0, len(cols))), np.array([], dtype=int), np.empty((0, len(cols)))

    if scaler is not None and scaler.mu is not None:
        mu=pd.Series(scaler.mu, index=cols); Xdf=Xdf.fillna(mu)

    X = np.nan_to_num(Xdf.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if scaler is not None:
        X = np.nan_to_num(scaler.transform(X), nan=0.0, posinf=0.0, neginf=0.0)

    # AdaNorm (early part of file) optional
    if cfg.uad_enable_adaptnorm and X.shape[0] >= 50:
        L = max(50, int(round(X.shape[0]*cfg.uad_file_pct)))
        mu_f = np.nanmean(X[:L], axis=0); sd_f = np.nanstd(X[:L], axis=0); sd_f = np.maximum(sd_f, 1e-8)
        alpha = float(cfg.uad_adaptnorm_alpha)
        mu = (1-alpha)*mu_f
        sd = (1-alpha)*sd_f + alpha*1.0
        X = (X - mu[None,:]) / sd[None,:]

    if crop_head_rows and crop_head_rows>0:
        if crop_head_rows>=X.shape[0]:
            return np.empty((0,X.shape[1])), np.empty((0,X.shape[1])), np.array([],dtype=int), np.empty((0,X.shape[1]))
        X = X[crop_head_rows:, :]

    T,N = X.shape
    preds,truth,t_idx,naives=[],[],[],[]
    for t in tqdm(range(lookback, T-horizon+1), desc=f"roll:{os.path.basename(file_path)[:24]}", ncols=100):
        x_win=X[t-lookback:t]; naive=x_win[-1].copy()
        x=torch.from_numpy(x_win[...,None]).unsqueeze(0).to(device)
        y_hat,_=model(x,A,M); y_pred=y_hat[0,0,:,0].detach().cpu().numpy()
        if not np.isfinite(y_pred).all(): y_pred=np.nan_to_num(y_pred, **NAN_REPLACE)
        preds.append(y_pred); truth.append(X[t+horizon-1]); naives.append(naive); t_idx.append(t+horizon-1)
    if not preds:
        return np.empty((0,N)), np.empty((0,N)), np.array([],dtype=int), np.empty((0,N))
    return np.vstack(preds), np.vstack(truth), np.array(t_idx,dtype=int), np.vstack(naives)

# ========= Post-processing =========
def uad_postprocess(bits: np.ndarray, min_run: int = 8, gap: int = 4) -> np.ndarray:
    x = bits.astype(np.int32).copy()
    i=0
    while i<len(x):
        if x[i]==0:
            j=i
            while j<len(x) and x[j]==0: j+=1
            if i>0 and j<len(x) and (j-i)<=gap and x[i-1]==1 and x[j]==1: x[i:j]=1
            i=j
        else: i+=1
    i=0
    while i<len(x):
        if x[i]==1:
            j=i
            while j<len(x) and x[j]==1: j+=1
            if (j-i)<min_run: x[i:j]=0
            i=j
        else: i+=1
    return x

# ========= Trainer =========
class UADTrainer:
    def __init__(self, cfg: UADConfig):
        self.cfg=cfg
        uad_makedirs(cfg.uad_fig_dir, True); uad_makedirs(cfg.uad_ckpt_dir, True)
        uad_set_seed(cfg.uad_seed)

        # graph
        self.node_names, self.A, self.M, self.topk_order = uad_load_graph(cfg.uad_graph_dir)
        self.A=self.A.to(cfg.uad_device); self.M=self.M.to(cfg.uad_device)
        print(f"[DBG] keep_columns -> {len(self.node_names)} features (sample: {self.node_names[:5]})")

        # ---- Benign (No_Failure) build ----
        all_files = uad_list_csvs(cfg.uad_nofail_dir)
        if not all_files:
            raise FileNotFoundError("No No_Failure files found (Benign Flight.csv).")

        self.single_file_ranges_train: Dict[str, Tuple[int,int]] = {}
        self.single_file_ranges_val: Dict[str, Tuple[int,int]] = {}

        if len(all_files) > 1:
            rng=random.Random(cfg.uad_seed)
            files=all_files[:]; rng.shuffle(files)
            k=int(min(len(files), max(1, round(len(files)*(1.0-cfg.uad_val_ratio)))))
            tr_files=files[:k]; va_files=files[k:]
            if len(va_files)==0 and len(tr_files)>1: va_files=tr_files[-1:]; tr_files=tr_files[:-1]
            self.train_paths = tr_files[:]; self.val_paths = va_files[:]
            print(f"[DBG] total no-failure files={len(all_files)} | train={len(tr_files)} | val={len(va_files)}")

            Xs=[]
            for p in tr_files:
                Xi = self._load_for_scaler(p)
                Xs.append(Xi)
            Xcat = np.concatenate(Xs, axis=0) if Xs else np.empty((0, len(self.node_names)), dtype=np.float32)
        else:
            only = all_files[0]
            Xi = self._load_for_scaler(only)
            T = Xi.shape[0]
            if T < (cfg.uad_lookback + cfg.uad_horizon) * 3:
                raise RuntimeError(
                    f"Single file too short to split: effective length={T}. "
                    "Reduce UAD_GLOBAL_HEAD_CROP or shorten lookback."
                )
            t_split = int((1.0 - cfg.uad_val_ratio) * T)
            need = cfg.uad_lookback + cfg.uad_horizon + 16
            t_split = max(need, min(T - need, t_split))
            self.train_paths = [only]
            self.val_paths   = [only]
            base = os.path.basename(only)
            self.single_file_ranges_train = {base: (0, t_split)}
            self.single_file_ranges_val   = {base: (t_split, T)}
            print(f"[DBG] single benign file -> internal split: train=[0,{t_split}) | val=[{t_split},{T}) (len={T})")
            Xcat = Xi[:t_split, :]

        if Xcat.size == 0:
            raise RuntimeError(
                "Training data empty after global crop/split; check UAD_GLOBAL_HEAD_CROP or data length."
            )

        # ---- Failure: first 30% for training; last 70% for testing ----
        fail_files = uad_list_csvs(cfg.uad_fail_dir)
        self.failure_ranges_train: Dict[str, Tuple[int,int]] = {}
        self.failure_ranges_test:  Dict[str, Tuple[int,int]] = {}
        self.fail_base2path: Dict[str, str] = {os.path.basename(p): p for p in fail_files}

        for p in fail_files:
            Xi_fail = self._load_for_scaler(p)   # globally head-cropped & preprocessed
            T = Xi_fail.shape[0]
            if T < (cfg.uad_lookback + cfg.uad_horizon) * 2:
                print(f"[WARN] Failure file too short, skipping: {os.path.basename(p)} (len={T})")
                continue
            t30 = int(round(0.30 * T))
            need = cfg.uad_lookback + cfg.uad_horizon + 8
            t30 = max(need, min(T - need, t30))  # avoid segments that are too short

            base = os.path.basename(p)
            self.failure_ranges_train[base] = (0, t30)
            self.failure_ranges_test[base]  = (t30, T)

            # Include Failure training segment in scaler statistics
            if t30 > 0:
                Xcat = np.concatenate([Xcat, Xi_fail[:t30, :]], axis=0)

        # Drop constant columns at runtime & trim A/M accordingly
        var=np.nanstd(Xcat,axis=0); mask_keep=(var>1e-12)
        if not mask_keep.all():
            dropped=[self.node_names[i] for i,b in enumerate(mask_keep) if not b]
            print(f"[WARN] Dropped constant columns at runtime: {dropped[:10]}{'...' if len(dropped)>10 else ''}")
            self.node_names=[c for c,b in zip(self.node_names, mask_keep) if b]
            Xcat=Xcat[:,mask_keep]
            idx=torch.from_numpy(np.where(mask_keep)[0].astype(np.int64)).to(cfg.uad_device)
            self.A=self.A.index_select(0,idx).index_select(1,idx); self.M=self.M.index_select(0,idx).index_select(1,idx)
            deg=(self.A>0).float().sum(dim=1).detach().cpu().numpy(); self.topk_order=list(np.argsort(-deg))

        self.scaler=UADStandardizer(); self.scaler.fit(Xcat)

        # —— DataLoaders: merge Failure first-30% ranges into training set ——
        ranges_train = dict(self.single_file_ranges_train)
        ranges_train.update(self.failure_ranges_train)
        self.tr_loader=DataLoader(
            UADDataset(self.train_paths + fail_files, self.node_names, self.scaler,
                       cfg.uad_lookback, cfg.uad_horizon,
                       stride=cfg.uad_stride, cfg=cfg, ranges=ranges_train),
            batch_size=cfg.uad_batch, shuffle=True, num_workers=0, collate_fn=uad_collate
        )
        self.va_loader=DataLoader(
            UADDataset(self.val_paths, self.node_names, self.scaler,
                       cfg.uad_lookback, cfg.uad_horizon,
                       stride=cfg.uad_stride, cfg=cfg, ranges=self.single_file_ranges_val),
            batch_size=cfg.uad_batch, shuffle=False, num_workers=0, collate_fn=uad_collate
        )
        print(f"[Data] train windows: {len(self.tr_loader.dataset):,} | val windows: {len(self.va_loader.dataset):,}")

        if cfg.uad_d_model % cfg.uad_nhead != 0: raise ValueError("d_model must be divisible by nhead.")

        # model
        self.model = STGraphTCN(num_nodes_hint=len(self.node_names), in_feat=1,
                                d_model=cfg.uad_d_model, short_kernel=cfg.uad_short_kernel,
                                nhead=cfg.uad_nhead, tcn_layers=cfg.uad_tcn_layers,
                                dropout=cfg.uad_dropout, eta=cfg.uad_eta_bias, beta=cfg.uad_beta_fuse,
                                out_feat=1, horizon=cfg.uad_horizon_out).to(cfg.uad_device)

        # optimizer & scheduler
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.uad_lr, weight_decay=cfg.uad_weight_decay)
        self.sched = (CosineWithWarmup(self.opt, total_epochs=cfg.uad_epochs,
                                       warmup_epochs=cfg.uad_warmup_epochs, min_lr_ratio=0.1)
                      if cfg.uad_sched_type == "cosine"
                      else torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode="min", patience=6, factor=0.5))

        # dry-run & EMA
        with torch.no_grad():
            for xb, _ in self.tr_loader:
                xb = xb.to(self.cfg.uad_device)
                _ = self.model(xb, self.A, self.M)
                break
        self.ema = UADEMA(self.model, decay=cfg.uad_ema_decay)

        # Test pairs (Failure last 70%)
        self.test_pairs: List[Tuple[str, Tuple[int,int]]] = []
        for base, rg in self.failure_ranges_test.items():
            p = self.fail_base2path.get(base, None)
            if p is None: continue
            if rg[1]-rg[0] >= (cfg.uad_lookback + cfg.uad_horizon):
                self.test_pairs.append((p, rg))
        print(f"[DBG] test files: {len(self.test_pairs)} -> "
              f"{[os.path.basename(p) + f'[{r[0]}:{r[1]} )' for p,r in self.test_pairs]}")

        # Placeholders
        self.keep_mask=None; self.dim_weights=None; self.resid_stats=None
        self.delta_mask = uad_build_delta_mask(self.node_names, self.cfg)

    def _load_for_scaler(self, p: str) -> np.ndarray:
        df = pd.read_csv(p)
        miss=[c for c in self.node_names if c not in df.columns]
        if miss: raise ValueError(f"[{os.path.basename(p)}] Missing columns: {miss[:8]} ...")
        Xi=uad_preprocess_cols(df, self.node_names, self.cfg)
        Xi=Xi.apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan).ffill().bfill()
        Xi=uad_input_smooth(Xi, self.cfg)
        if len(Xi) > UAD_GLOBAL_HEAD_CROP:
            Xi = Xi.iloc[UAD_GLOBAL_HEAD_CROP:, :]
        else:
            Xi = Xi.iloc[0:0, :]
        return Xi.values.astype(np.float32)

    def _step_train_epoch(self):
        self.model.train(); tot=n=0
        pbar = tqdm(self.tr_loader, desc="train", ncols=100)
        for x,y in pbar:
            x=x.to(self.cfg.uad_device); y=y.to(self.cfg.uad_device)
            if self.cfg.uad_mixup_alpha > 0:
                x, y = uad_mixup_batch(x, y, self.cfg.uad_mixup_alpha)

            y_hat,_=self.model(x,self.A,self.M); y_pred=y_hat[:,0,:,0]
            if (not torch.isfinite(y).all()) or (not torch.isfinite(y_pred).all()):
                if HARD_FAIL_ON_NONFINITE: raise RuntimeError("Non-finite detected.")
                y_pred=torch.nan_to_num(y_pred, **NAN_REPLACE); y=torch.nan_to_num(y, **NAN_REPLACE)

            base=uad_main_loss(y_pred,y,self.cfg.uad_loss,self.cfg.uad_huber_delta)
            x_last=x[:,-1,:,0]; d_true=y-x_last; d_pred=y_pred-x_last
            loss=base + self.cfg.uad_lambda_delta*F.l1_loss(d_pred,d_true)
            self.opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.opt.step()
            self.ema.update(self.model)

            bs=x.size(0); tot+=float(loss.detach())*bs; n+=bs
            pbar.set_postfix(loss=f"{tot/max(1,n):.5f}")
        return tot/max(1,n)

    @torch.no_grad()
    def _step_val_epoch(self, use_ema: bool = True):
        self.model.eval(); tot=n=0
        if use_ema:
            self.ema.store(self.model); self.ema.copy_to(self.model)
        pbar=tqdm(self.va_loader, desc="val  ", ncols=100)
        for x,y in pbar:
            x=x.to(self.cfg.uad_device); y=y.to(self.cfg.uad_device)
            y_hat,_=self.model(x,self.A,self.M); y_pred=y_hat[:,0,:,0]
            if (not torch.isfinite(y).all()) or (not torch.isfinite(y_pred).all()):
                if HARD_FAIL_ON_NONFINITE: raise RuntimeError("Non-finite detected (val).")
                y_pred=torch.nan_to_num(y_pred, **NAN_REPLACE); y=torch.nan_to_num(y, **NAN_REPLACE)
            base=uad_main_loss(y_pred,y,self.cfg.uad_loss,self.cfg.uad_huber_delta)
            x_last=x[:,-1,:,0]; d_true=y-x_last; d_pred=y_pred-x_last
            loss=base + self.cfg.uad_lambda_delta*F.l1_loss(d_pred,d_true)
            bs=x.size(0); tot+=float(loss.detach())*bs; n+=bs
            pbar.set_postfix(loss=f"{tot/max(1,n):.5f}")
        if use_ema:
            self.ema.restore(self.model)
        return tot/max(1,n) if n>0 else float('inf')

    def fit(self):
        hist={"train":[], "val":[]}; best=1e9; bad=0
        ckpt=os.path.join(self.cfg.uad_ckpt_dir,"stgraphtcn_forecast_best.pt")
        for ep in range(1, self.cfg.uad_epochs+1):
            tr=self._step_train_epoch()
            va=self._step_val_epoch(use_ema=True)
            if isinstance(self.sched, CosineWithWarmup):
                self.sched.step()
                lr = self.opt.param_groups[0]["lr"]
            else:
                self.sched.step(va)
                lr = self.opt.param_groups[0]["lr"]
            hist["train"].append(tr); hist["val"].append(va)
            print(f"[Epoch {ep:03d}] train={tr:.5f} | val={va:.5f} | lr={lr:.2e}")
            if va+1e-6<best:
                best=va; bad=0
                self.ema.store(self.model); self.ema.copy_to(self.model)
                torch.save(self.model.state_dict(), ckpt)
                self.ema.restore(self.model)
            else:
                bad+=1
                if bad>=self.cfg.uad_patience:
                    print("Early stopping."); break
        uad_plot_train_curves(hist, os.path.join(self.cfg.uad_fig_dir,"forecast_training.png"))
        if os.path.exists(ckpt): self.model.load_state_dict(torch.load(ckpt, map_location=self.cfg.uad_device), strict=False)

    @torch.no_grad()
    def _collect_train_residuals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect training residuals for threshold calibration:
        - Benign multi-file: iterate training files (head-cropped)
        - Single benign file: only the training segment
        - Failure: only the first 30% training segment per file
        """
        E_model_all, E_naive_all = [], []
        range_map = {}
        range_map.update(self.single_file_ranges_train)   # Benign training segment (if any)
        range_map.update(self.failure_ranges_train)       # Failure first 30%

        # Build training sources: all benign training files + failures with training segments
        train_sources = set(self.train_paths)
        for base in self.failure_ranges_train.keys():
            p = self.fail_base2path.get(base, None)
            if p is not None:
                train_sources.add(p)

        for p in sorted(train_sources):
            preds,truth,t_idx,naive = uad_rolling_forecast(self.model,p,self.node_names,self.scaler,
                                                           self.A,self.M,self.cfg.uad_device,
                                                           self.cfg.uad_lookback,self.cfg.uad_horizon,cfg=self.cfg,
                                                           crop_head_rows=0)
            if preds.shape[0]==0: continue

            base = os.path.basename(p)
            if base in range_map:
                t_start, t_end = range_map[base]
                mask = (t_idx >= t_start) & (t_idx < t_end)
                if not np.any(mask):
                    continue
                preds = preds[mask]; truth = truth[mask]; naive = naive[mask]

            E_model_all.append(uad_perdim_errors(truth,preds,self.delta_mask))
            E_naive_all.append(uad_perdim_errors(truth,naive,self.delta_mask))

        if not E_model_all: raise RuntimeError("Failed to collect training residuals.")
        return np.vstack(E_model_all), np.vstack(E_naive_all)

    @torch.no_grad()
    def compute_thresholds(self):
        print(">> Calibrating per-dim stats & global threshold ...")
        E_model, E_naive = self._collect_train_residuals()

        self.keep_mask, self.dim_weights, self.resid_stats = uad_dim_keep_and_weights(E_model, E_naive, self.cfg)
        print(f"[DBG] kept dims: {int(self.keep_mask.sum())}/{len(self.keep_mask)}")

        Z = uad_normalize_errors(E_model, self.resid_stats, self.cfg)
        R_train = aggregate_topk_scores(
            Z, self.cfg.uad_topk_ratio, self.dim_weights, self.keep_mask,
            agg=self.cfg.uad_topk_agg, trim_ratio=self.cfg.uad_topk_trim_ratio, trim_high_only=self.cfg.uad_topk_trim_high_only
        )
        R_train = uad_score_smooth(R_train, self.cfg)

        mode=self.cfg.uad_thr_mode.lower()
        if mode=="mad":
            med=float(np.median(R_train)); mad=float(np.median(np.abs(R_train-med))+1e-12)
            thr = med + 3.0*(1.4826*mad); thr_name="robust-3σ"
        elif mode=="quantile":
            thr=float(np.quantile(R_train,self.cfg.uad_thr_quantile)); thr_name=f"q{int(self.cfg.uad_thr_quantile*1000)/10:.1f}%"
        elif mode=="gauss":
            mu=float(np.mean(R_train)); sigma=float(np.std(R_train)+1e-12); thr=mu+3.0*sigma; thr_name="3σ"
        elif mode=="gev":
            try:
                from scipy.stats import genextreme
                k = max(30, int(0.05*len(R_train)))
                tail = np.sort(R_train)[-k:]
                c, loc, scale = genextreme.fit(tail)
                alpha = 0.99
                thr = float(genextreme.ppf(alpha, c, loc=loc, scale=scale))
                thr_name=f"GEV({alpha:.2f})"
            except Exception:
                mu=float(np.mean(R_train)); sigma=float(np.std(R_train)+1e-12); thr=mu+3.0*sigma; thr_name="3σ"
        else:
            mu=float(np.mean(R_train)); sigma=float(np.std(R_train)+1e-12); thr=mu+3.0*sigma; thr_name="3σ"

        plt.figure(); plt.hist(R_train, bins=80, alpha=0.85); plt.axvline(thr, ls="--", color="r")
        plt.title("Scalar residuals (train, smoothed) & threshold"); plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.uad_fig_dir,"train_scalar_residual_hist_threshold.png")); plt.close()
        print(f"[DBG] global threshold={thr:.6f} ({thr_name}) | topk_agg={self.cfg.uad_topk_agg}")
        return thr, thr_name

    @staticmethod
    def _file_adaptive_thr(scores: np.ndarray, cfg: UADConfig) -> Optional[float]:
        if (not cfg.uad_enable_file_thr) or len(scores)<20: return None
        L = max(20, int(round(len(scores)*float(cfg.uad_file_pct))))
        early = scores[:L]
        if cfg.uad_file_thr_mode.lower()=="mad":
            med=float(np.median(early)); mad=float(np.median(np.abs(early-med))+1e-12); sigma=1.4826*mad
            return med + float(cfg.uad_file_thr_k)*sigma
        else:
            mu=float(np.mean(early)); sigma=float(np.std(early)+1e-12)
            return mu + float(cfg.uad_file_thr_k)*sigma

    @torch.no_grad()
    def evaluate_file(self, tag: str, file_path: str, thr_global: float, thr_name: str,
                      eval_range: Optional[Tuple[int,int]] = None):
        base=f"{tag}__{os.path.basename(file_path).replace('.csv','')}"
        preds_full,truth_full,t_idx,naive_full = uad_rolling_forecast(
            self.model,file_path,self.node_names,self.scaler,
            self.A,self.M,self.cfg.uad_device,
            self.cfg.uad_lookback,self.cfg.uad_horizon,cfg=self.cfg,
            crop_head_rows=0
        )
        if preds_full.shape[0]==0:
            print(f"[Skip] {base}: data too short"); return None

        labels_full = uad_labels_from_file(
            file_path, length=preds_full.shape[0],
            lookback=self.cfg.uad_lookback, horizon=self.cfg.uad_horizon,
            label_col=self.cfg.uad_label_col, shift=self.cfg.uad_label_shift,
            extra_head_crop=0
        )

        if eval_range is not None:
            a,b = eval_range
            mask = (t_idx >= a) & (t_idx < b)
            if not np.any(mask):
                print(f"[Skip] {base}: no valid windows in the specified eval range")
                return None
            preds = preds_full[mask]; truth = truth_full[mask]; naive = naive_full[mask]
            times = t_idx[mask]; labels = labels_full[mask]
        else:
            preds, truth, naive, times, labels = preds_full, truth_full, naive_full, t_idx, labels_full

        uad_plot_forecast(times, truth, preds, self.node_names, self.topk_order,
                          K=self.cfg.uad_topk_nodes_viz,
                          out_png=os.path.join(self.cfg.uad_fig_dir, f"forecast_vs_truth__{base}.png"))

        E_model = uad_perdim_errors(truth, preds, self.delta_mask)
        Z = uad_normalize_errors(E_model, self.resid_stats, self.cfg)

        scores_list = []
        tta_cfgs = [("bi_ewma", self.cfg.uad_score_ewma_alpha),
                    ("ema", 0.18),
                    ("ma", max(5, self.cfg.uad_score_ma_window))]
        for mode, param in tta_cfgs:
            s = aggregate_topk_scores(Z, self.cfg.uad_topk_ratio, self.dim_weights, self.keep_mask,
                                      agg=self.cfg.uad_topk_agg,
                                      trim_ratio=self.cfg.uad_topk_trim_ratio,
                                      trim_high_only=self.cfg.uad_topk_trim_high_only)
            if mode == "ma":
                s = _ma(s, win=int(param), causal=self.cfg.uad_score_causal)
            elif mode == "ema":
                s = _ema_1d(s, a=float(param))
            else:
                s = _bi_ewma(s, a=float(param))
            scores_list.append(s.astype(np.float32))
        scores = np.mean(np.stack(scores_list, axis=0), axis=0).astype(np.float32)

        thr_file = self._file_adaptive_thr(scores, self.cfg)
        if thr_file is not None:
            combine = (self.cfg.uad_file_thr_combine or "max").lower()
            if combine=="max": thr = max(thr_global, thr_file)
            elif combine=="mean": thr = 0.5*(thr_global + thr_file)
            else: thr = thr_global
            thr_used_name = f"{thr_name} + adapt({combine})"
        else:
            thr = thr_global; thr_used_name = thr_name

        uad_plot_residual(times, scores, labels, thr,
                          out_png=os.path.join(self.cfg.uad_fig_dir, f"residual_timeline__{base}.png"),
                          thr_name=thr_used_name)

        pred_bin = (scores >= thr).astype(np.int32)
        pred_bin = uad_postprocess(pred_bin, min_run=self.cfg.uad_post_min_run, gap=self.cfg.uad_post_gap)

        if labels.sum()==0:
            tn=int(((pred_bin==0)&(labels==0)).sum()); fp=int(((pred_bin==1)&(labels==0)).sum())
            fpr=fp/max(1,fp+tn); print(f"[{base}] All negatives. TN={tn}, FP={fp}, FPR={fpr:.4f}")
            return {"file":base,"tp":0,"fp":fp,"tn":tn,"fn":0,"f1":None,"prec":None,"rec":None,"fpr":fpr}

        tp=int(((pred_bin==1)&(labels==1)).sum())
        fp=int(((pred_bin==1)&(labels==0)).sum())
        tn=int(((pred_bin==0)&(labels==0)).sum())
        fn=int(((pred_bin==0)&(labels==1)).sum())
        prec=tp/max(1,tp+fp)
        rec=tp/max(1,tp+fn)
        f1 = 2*prec*rec/max(1e-12,prec+rec)
        print(f"[{base}] P={prec:.4f} R={rec:.4f} F1={f1:.4f} (tp={tp}, fp={fp}, tn={tn}, fn={fn})  thr_file={thr_file if thr_file is not None else 'NA'}")
        return {"file":base,"tp":tp,"fp":fp,"tn":tn,"fn":fn,"f1":f1,"prec":prec,"rec":rec,"fpr":None}

    def run(self):
        self.fit()
        thr, thr_name = self.compute_thresholds()
        results=[]
        for path, rg in self.test_pairs:
            r=self.evaluate_file("failure", path, thr, thr_name, eval_range=rg)
            if r is not None: results.append(r)
        f1_list=[r["f1"] for r in results if r.get("f1") is not None]
        if f1_list:
            print(f"\n=== Test summary over {len(f1_list)} labelled files ===")
            print(f"mean F1 = {np.mean(f1_list):.4f}  median F1 = {np.median(f1_list):.4f}")
        else:
            print("\n=== Test summary: no labelled positives found (only FPR) ===")
        if results:
            out_fp=os.path.join(self.cfg.uad_fig_dir,"test_results_summary.csv")
            pd.DataFrame(results).to_csv(out_fp, index=False)
            print(f"[DBG] saved test summary to {out_fp}")

# ========= main =========
def main():
    cfg=UADConfig()
    uad_makedirs(cfg.uad_fig_dir, True); uad_makedirs(cfg.uad_ckpt_dir, True)
    trainer=UADTrainer(cfg)
    trainer.run()

if __name__ == "__main__":
    main()
