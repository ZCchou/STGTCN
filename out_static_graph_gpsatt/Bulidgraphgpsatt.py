# -*- coding: utf-8 -*-
"""
GraphBuild_static_gpsatt.py  (isolated-safe)
Based on GraphBuild: build a feature graph from a single input CSV (default Benign Flight.csv).
- Column filtering: numeric columns only; explicitly exclude timestamp* and label; drop all-NaN and **constant columns**.
- Row sampling: up to ROW_SAMPLE_MAX rows (reproducible sampling).
- Candidate prefilter: top-K by Spearman |ρ| (up to K per feature, must be ≥ PREFILTER_R_MIN).
- Metric computation: prefer minepy MIC (MINE); fallback to Spearman |ρ| if unavailable.
- Parallelism: multiprocessing + memmap shared sample matrix.
- Threshold: upper-triangle quantile with fallback and density-tightening strategy.
- **Isolated-node guard: after thresholding, drop nodes with degree < MIN_DEGREE (optional)**.
- Outputs: write GraphBuild-style filenames into out_dir.

Usage example:
    python GraphBuild_static_gpsatt.py \
        --in_csv "Benign Flight.csv" \
        --out_dir "./out_static_graph_gpsatt" \
        --row_sample_max 100000 \
        --prefilter_topk 20 \
        --prefilter_r_min 0.15 \
        --quantile 0.975 \
        --mic_sample_max 20000 \
        --processes 0 \
        --drop_isolated 1 \
        --min_degree 1 \
        --random_seed 42
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

# Try to import minepy (preferred for MIC)
try:
    from minepy import MINE
    HAVE_MINEPY = True
except Exception:
    HAVE_MINEPY = False

# ---------- CLI arguments ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="./preprocessed/Benign Flight.csv", help="Input data CSV")
    ap.add_argument("--out_dir", default="./out_static_graph_gpsatt", help="Output directory")

    # Column filtering
    ap.add_argument("--drop_constant", type=int, default=1, help="Drop constant columns (1=yes, 0=no)")

    # Row sampling
    ap.add_argument("--row_sample_max", type=int, default=100000, help="Max sampled rows for graph build")
    ap.add_argument("--random_seed", type=int, default=42, help="Random seed")

    # Candidate prefilter (Spearman)
    ap.add_argument("--prefilter_topk", type=int, default=20, help="Top-K candidates per feature")
    ap.add_argument("--prefilter_r_min", type=float, default=0.15, help="Minimum candidate |ρ|")

    # MIC computation
    ap.add_argument("--mic_sample_max", type=int, default=20000, help="Max sampled rows per MIC pair")

    # Threshold strategy
    ap.add_argument("--quantile", type=float, default=0.975, help="Upper-triangle quantile threshold q")
    ap.add_argument("--max_density", type=float, default=0.30, help="Max edge density for tightening")

    # Parallelism
    ap.add_argument("--processes", type=int, default=0, help="Number of processes (0=cpu_count-1)")

    # Isolated node handling
    ap.add_argument("--drop_isolated", type=int, default=0, help="Drop isolated nodes after threshold (1=yes, 0=no)")
    ap.add_argument("--min_degree", type=int, default=1, help="Minimum degree to keep nodes (default 1)")

    return ap.parse_args()


# ---------- Column filtering ----------
def filter_columns(df: pd.DataFrame, drop_constant: bool = True):
    drop_report = []

    # Explicitly exclude: timestamp* and label
    explicit_exclude = set()
    for c in df.columns:
        lc = str(c).lower()
        if lc == "timestamp" or lc.startswith("timestamp"):
            explicit_exclude.add(c)
        if lc == "label":
            explicit_exclude.add(c)

    # Numeric columns only
    num_df = df.select_dtypes(include=[np.number]).copy()
    kept_numeric_cols = list(num_df.columns)

    cols_to_drop = set()
    for c in kept_numeric_cols:
        if c in explicit_exclude:
            cols_to_drop.add(c)
            drop_report.append((c, "explicit_exclude"))

    # All-NaN columns
    for c in kept_numeric_cols:
        if c in cols_to_drop:
            continue
        if num_df[c].isna().all():
            cols_to_drop.add(c)
            drop_report.append((c, "all_nan"))

    # **Constant columns** (must be dropped per requirement)
    if drop_constant:
        for c in kept_numeric_cols:
            if c in cols_to_drop:
                continue
            s = num_df[c].dropna()
            if s.empty:
                continue
            if s.nunique(dropna=True) <= 1:
                cols_to_drop.add(c)
                drop_report.append((c, "constant"))

    keep_cols = [c for c in kept_numeric_cols if c not in cols_to_drop]
    return keep_cols, drop_report, num_df[keep_cols].astype(np.float32)


# ---------- Sampling ----------
def sample_rows(X: pd.DataFrame, row_sample_max: int, seed: int) -> pd.DataFrame:
    if len(X) <= row_sample_max:
        return X.copy()
    return X.sample(n=row_sample_max, random_state=seed).copy()


# ---------- Prefilter: Spearman absolute correlation Top-K ----------
def prefilter_candidates_spearman(X: pd.DataFrame, topk: int, r_min: float) -> List[Tuple[int, int]]:
    # Fill missing values (Spearman is rank-based; pairwise NaN reduces samples, so fill with median)
    Xf = X.fillna(X.median(numeric_only=True))
    corr = Xf.corr(method="spearman").abs().fillna(0.0).values  # ndarray
    n = corr.shape[0]
    pairs = set()
    for i in range(n):
        r = corr[i].copy()
        r[i] = -1.0  # exclude self-correlation
        # Take Top-K
        k = min(topk, n-1)
        if k <= 0:
            continue
        idx = np.argpartition(-r, kth=k-1)[:k]
        for j in idx:
            if j == i:
                continue
            if r[j] >= r_min:
                a, b = (i, j) if i < j else (j, i)
                pairs.add((a, b))
    return sorted(pairs)


# ---------- Global memmap for parallelism ----------
_MEMMAP_PATH = None
_MEMMAP_SHAPE = None
_MEMMAP_DTYPE = np.float32

def _init_worker(memmap_path: str, shape_tuple: Tuple[int, int]):
    global _MEMMAP_PATH, _MEMMAP_SHAPE
    _MEMMAP_PATH = memmap_path
    _MEMMAP_SHAPE = shape_tuple

def _load_memmap():
    assert _MEMMAP_PATH is not None and _MEMMAP_SHAPE is not None
    mm = np.memmap(_MEMMAP_PATH, mode="r", dtype=_MEMMAP_DTYPE, shape=_MEMMAP_SHAPE)
    return mm

def _compute_metric_for_pair(args):
    """
    Worker task: compute MIC (minepy) for (i, j), otherwise fallback to Spearman |ρ|.
    Limit to mic_sample_max rows, prefer uniform subsampling (cache-friendly).
    """
    i, j, mic_sample_max, seed = args
    Xmm = _load_memmap()
    n_rows = Xmm.shape[0]
    # Uniform subsampling
    if n_rows > mic_sample_max:
        step = n_rows / mic_sample_max
        idx = (np.floor(np.arange(mic_sample_max) * step)).astype(int)
    else:
        idx = np.arange(n_rows, dtype=int)

    xi = Xmm[idx, i].astype(np.float64, copy=False)
    xj = Xmm[idx, j].astype(np.float64, copy=False)

    # Drop NaNs
    mask = np.isfinite(xi) & np.isfinite(xj)
    xi = xi[mask]; xj = xj[mask]
    if len(xi) < 5:
        return (i, j, 0.0)

    if HAVE_MINEPY:
        try:
            mine = MINE(alpha=0.6, c=15)
            mine.compute_score(xi, xj)
            mic = float(mine.mic())
            return (i, j, mic if np.isfinite(mic) else 0.0)
        except Exception:
            pass

    # Fallback to Spearman
    try:
        r = pd.Series(xi).corr(pd.Series(xj), method="spearman")
        return (i, j, float(abs(r)) if np.isfinite(r) else 0.0)
    except Exception:
        return (i, j, 0.0)


# ---------- Threshold selection ----------
def choose_threshold_upper_quantile(A: np.ndarray, q: float, max_density: float):
    """
    Choose a threshold on adjacency matrix A (symmetric, upper-triangle edges, diagonal=1):
    - Start at quantile q; if degenerate (NaN/>=1/no edges), fall back downwards;
    - If too dense (edge density > max_density), tighten the threshold.
    """
    n = A.shape[0]
    if n <= 1:
        return 1.0, 0, 0
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = A[mask]
    if vals.size == 0:
        return 1.0, 0, 0

    thr = float(np.nanquantile(vals, q))

    def edge_count_at(t):
        return int((A[mask] >= t).sum())

    # Fallback
    if not np.isfinite(thr) or thr >= 1.0 or edge_count_at(thr) == 0:
        for cand in [0.999, 0.995, 0.99, 0.98, 0.95, 0.90, 0.80, 0.70, 0.60]:
            if edge_count_at(cand) > 0:
                thr = cand
                break
        else:
            thr = 0.0  # keep all

    total_possible = int(mask.sum())
    edges_now = edge_count_at(thr)

    # Tighten to control density
    if total_possible > 0 and edges_now / total_possible > max_density:
        for cand in [0.99, 0.995, 0.999]:
            if cand > thr and edge_count_at(cand) / total_possible <= max_density and edge_count_at(cand) > 0:
                thr = cand
                edges_now = edge_count_at(cand)
                break

    return thr, edges_now, total_possible


# ---------- Main flow ----------
def main():
    args = parse_args()
    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read data
    df = pd.read_csv(in_csv)

    # Column filtering (including constant columns)
    keep_cols, drop_report, Xnum = filter_columns(df, drop_constant=bool(args.drop_constant))

    # Edge case: fewer than two columns (export empty artifacts)
    if len(keep_cols) < 2:
        (out_dir / "keep_columns.json").write_text(json.dumps(keep_cols, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(drop_report, columns=["column","reason"]).to_csv(out_dir / "drop_columns_report.csv", index=False)
        pd.DataFrame({"id": range(len(keep_cols)), "name": keep_cols}).to_csv(out_dir / "nodes.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "adjacency_dense.csv", index=False)
        pd.DataFrame(columns=["src","dst","weight"]).to_csv(out_dir / "edges_mic.csv", index=False)
        pd.DataFrame(columns=["i","j","weight"]).to_csv(out_dir / "A_global_sparse.csv", index=False)
        pd.DataFrame(columns=["node","degree","weighted_degree"]).to_csv(out_dir / "degrees.csv", index=False)
        (out_dir / "train_files.txt").write_text(f"{in_csv.name}\n", encoding="utf-8")
        (out_dir / "heldout_files.txt").write_text("", encoding="utf-8")
        print(f"[WARN] Not enough columns after filtering: {len(keep_cols)}")
        return

    # Row sampling
    Xs = sample_rows(Xnum, row_sample_max=args.row_sample_max, seed=args.random_seed)

    # Prefilter candidates
    print("[INFO] Prefilter candidates by Spearman...")
    cand_pairs = prefilter_candidates_spearman(Xs, topk=args.prefilter_topk, r_min=args.prefilter_r_min)
    print(f"[INFO] Candidate pairs: {len(cand_pairs)} (from {len(keep_cols)} features)")

    # Build memmap for parallelism
    mm_path = str(out_dir / "sample.memmap")
    Xarr = Xs.to_numpy(dtype=np.float32, copy=True)
    with open(mm_path, "wb"):
        pass
    mm = np.memmap(mm_path, mode="w+", dtype=np.float32, shape=Xarr.shape)
    mm[:] = Xarr[:]
    del mm  # close write view

    # Parallel config
    if args.processes <= 0:
        try:
            import multiprocessing as mp
            procs = max(1, mp.cpu_count() - 1)
        except Exception:
            procs = 1
    else:
        procs = args.processes

    # Compute metrics (MIC or Spearman fallback)
    import multiprocessing as mp
    print(f"[INFO] Compute pair metrics with {procs} process(es); minepy={HAVE_MINEPY}")
    tasks = [(i, j, int(args.mic_sample_max), int(args.random_seed)) for (i, j) in cand_pairs]
    with mp.get_context("spawn").Pool(processes=procs, initializer=_init_worker,
                                      initargs=(mm_path, Xarr.shape)) as pool:
        results = list(pool.imap_unordered(_compute_metric_for_pair, tasks, chunksize=256))

    # Assemble adjacency (symmetric; diag set to 1.0 for downstream, training will set to 0)
    n = len(keep_cols)
    A = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(A, 1.0)
    for i, j, w in results:
        if w > 0:
            if 0 <= i < n and 0 <= j < n:
                if i != j:
                    A[i, j] = max(A[i, j], w)
                    A[j, i] = max(A[j, i], w)

    # Threshold selection (upper triangle)
    thr, edges_now, total_possible = choose_threshold_upper_quantile(A, q=args.quantile, max_density=args.max_density)
    print(f"[INFO] Threshold={thr:.6f}  edges={edges_now}/{total_possible}")

    # ====== Drop isolated nodes (critical) ======
    # After thresholding, compute binary degree and filter by --min_degree
    if int(args.drop_isolated) == 1:
        deg = np.zeros(n, dtype=np.int32)
        wdeg = np.zeros(n, dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                w = A[i, j]
                if w >= thr and w > 0.0:
                    deg[i] += 1
                    wdeg[i] += w
        min_deg = max(0, int(args.min_degree))
        keep_mask = deg >= min_deg
        removed_idx = np.where(~keep_mask)[0].tolist()
        if len(removed_idx) > 0:
            removed_names = [keep_cols[i] for i in removed_idx]
            print(f"[INFO] Remove isolated/low-degree nodes: {len(removed_idx)} -> {removed_names[:10]}{' ...' if len(removed_idx)>10 else ''}")
            # Record in drop_report
            for nm in removed_names:
                drop_report.append((nm, "isolated_after_threshold"))

            # Apply filtering
            A = A[keep_mask][:, keep_mask]
            keep_cols = [c for c, m in zip(keep_cols, keep_mask) if m]
            n = len(keep_cols)

    # ====== If fewer than two columns remain, output empty artifacts to avoid training crashes ======
    if n < 2:
        (out_dir / "keep_columns.json").write_text(json.dumps(keep_cols, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(drop_report, columns=["column","reason"]).to_csv(out_dir / "drop_columns_report.csv", index=False)
        pd.DataFrame({"id": range(n), "name": keep_cols}).to_csv(out_dir / "nodes.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "adjacency_dense.csv", index=False)
        pd.DataFrame(columns=["src","dst","weight"]).to_csv(out_dir / "edges_mic.csv", index=False)
        pd.DataFrame(columns=["i","j","weight"]).to_csv(out_dir / "A_global_sparse.csv", index=False)
        pd.DataFrame(columns=["node","degree","weighted_degree"]).to_csv(out_dir / "degrees.csv", index=False)
        (out_dir / "train_files.txt").write_text(f"{in_csv.name}\n", encoding="utf-8")
        (out_dir / "heldout_files.txt").write_text("", encoding="utf-8")
        try:
            os.remove(mm_path)
        except Exception:
            pass
        print(f"[WARN] Too few columns after isolation filtering: {n}")
        return

    # ====== Export artifacts (based on post-isolation column set) ======
    # keep_columns.json (write last to ensure final columns)
    (out_dir / "keep_columns.json").write_text(json.dumps(keep_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(drop_report, columns=["column","reason"]).to_csv(out_dir / "drop_columns_report.csv", index=False)

    # nodes
    nodes_df = pd.DataFrame({"id": np.arange(n, dtype=int), "name": keep_cols})
    nodes_df.to_csv(out_dir / "nodes.csv", index=False)

    # adjacency_dense (with row/column names)
    adj_df = pd.DataFrame(A, index=keep_cols, columns=keep_cols)
    adj_df.to_csv(out_dir / "adjacency_dense.csv", index=True)

    # Generate edges_mic / sparse triplets / degree files from final columns and threshold
    edges = []
    deg = np.zeros(n, dtype=np.int32)
    wdeg = np.zeros(n, dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            w = float(A[i, j])
            if w >= thr and w > 0.0:
                edges.append((keep_cols[i], keep_cols[j], w))
                deg[i] += 1; deg[j] += 1
                wdeg[i] += w; wdeg[j] += w

    edges_df = pd.DataFrame(edges, columns=["src","dst","weight"]).sort_values("weight", ascending=False)
    edges_df.to_csv(out_dir / "edges_mic.csv", index=False)

    # Sparse triplets (include diagonal self-loop = 1)
    trips = [(i, i, 1.0) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            w = float(A[i, j])
            if w >= thr and w > 0.0:
                trips.append((i, j, w))
                trips.append((j, i, w))
    sparse_df = pd.DataFrame(trips, columns=["i","j","weight"])
    sparse_df.to_csv(out_dir / "A_global_sparse.csv", index=False)

    # Degree and weighted degree
    deg_df = pd.DataFrame({"node": keep_cols, "degree": deg, "weighted_degree": wdeg})
    deg_df.to_csv(out_dir / "degrees.csv", index=False)

    # Metadata (GraphBuild-style)
    (out_dir / "train_files.txt").write_text(f"{in_csv.name}\n", encoding="utf-8")
    (out_dir / "heldout_files.txt").write_text("", encoding="utf-8")

    # Clean up memmap temp file
    try:
        os.remove(mm_path)
    except Exception:
        pass

    print(f"[OK] Graph built at: {out_dir}")
    print(f"Nodes(after drop)={n}  Edges={len(edges_df)}  thr={thr:.6f}")


if __name__ == "__main__":
    main()
