# -*- coding: utf-8 -*-
"""
GraphBuild_static_alfa.py  (dataset-friendly)
从 ALFA 目录读取 No_Failure 的 80% CSV 合并构图（可调），
保留原 GraphBuild 的 MIC/阈值/孤立点等逻辑与产物格式。

用法示例：
  python GraphBuild_static_alfa.py \
    --alfadata_dir ./alfadata \
    --out_dir ./out_static_graph_alfa \
    --no_failure_fraction 0.8 \
    --row_sample_max 100000 \
    --prefilter_topk 20 --prefilter_r_min 0.15 \
    --quantile 0.975 --mic_sample_max 20000 \
    --drop_isolated 1 --min_degree 1 \
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

# 尝试导入 minepy（优先用于 MIC）
try:
    from minepy import MINE
    HAVE_MINEPY = True
except Exception:
    HAVE_MINEPY = False

# ---------- 命令行参数 ----------
def parse_args():
    ap = argparse.ArgumentParser()
    # 单文件模式（向后兼容）
    ap.add_argument("--in_csv", default=None, help="输入数据 CSV（单文件模式）")

    # 数据集模式（优先生效）
    ap.add_argument("--alfadata_dir", default="../alfadata2", help="ALFA 数据集根目录（含 No_Failure/ 与 Failure/）")
    ap.add_argument("--no_failure_subdir", default="No_Failure", help="良性子目录名称（默认 No_Failure）")
    ap.add_argument("--file_glob", default="*.csv", help="CSV 文件通配（默认 *.csv）")
    ap.add_argument("--no_failure_fraction", type=float, default=0.8, help="用于构图的 No_Failure 文件比例")
    ap.add_argument("--min_select", type=int, default=1, help="最少选择文件数（防止过小样本）")

    ap.add_argument("--out_dir", default="./out_static_graph_alfa", help="输出目录")

    # 列筛选
    ap.add_argument("--drop_constant", type=int, default=1, help="是否丢弃常量列（1=是，0=否）")

    # 行采样（对合并后的总样本）
    ap.add_argument("--row_sample_max", type=int, default=100000, help="用于构图的最大采样行数")
    ap.add_argument("--random_seed", type=int, default=42, help="随机种子")

    # 候选预筛（Spearman）
    ap.add_argument("--prefilter_topk", type=int, default=20, help="每个特征保留的候选上限 K")
    ap.add_argument("--prefilter_r_min", type=float, default=0.15, help="候选最小 |ρ|")

    # MIC 计算
    ap.add_argument("--mic_sample_max", type=int, default=20000, help="单对 MIC 的最大采样行数")

    # 阈值策略
    ap.add_argument("--quantile", type=float, default=0.975, help="上三角分位阈值 q")
    ap.add_argument("--max_density", type=float, default=0.30, help="过密加严的边密度上限")

    # 并行
    ap.add_argument("--processes", type=int, default=0, help="并行进程数（0=cpu_count-1）")

    # 孤立点处理
    ap.add_argument("--drop_isolated", type=int, default=0, help="是否在阈值后剔除孤立点（1=是，0=否）")
    ap.add_argument("--min_degree", type=int, default=1, help="保留节点的最小度阈值（默认 1）")

    return ap.parse_args()


# ---------- 列筛选 ----------
def filter_columns(df: pd.DataFrame, drop_constant: bool = True):
    drop_report = []

    # 显式排除：timestamp* 与 label
    explicit_exclude = set()
    for c in df.columns:
        lc = str(c).lower()
        if lc == "timestamp" or lc.startswith("timestamp"):
            explicit_exclude.add(c)
        if lc == "label":
            explicit_exclude.add(c)

    # 仅数值列
    num_df = df.select_dtypes(include=[np.number]).copy()
    kept_numeric_cols = list(num_df.columns)

    cols_to_drop = set()
    for c in kept_numeric_cols:
        if c in explicit_exclude:
            cols_to_drop.add(c)
            drop_report.append((c, "explicit_exclude"))

    # 全 NaN 列
    for c in kept_numeric_cols:
        if c in cols_to_drop:
            continue
        if num_df[c].isna().all():
            cols_to_drop.add(c)
            drop_report.append((c, "all_nan"))

    # 常量列
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


# ---------- 采样 ----------
def sample_rows(X: pd.DataFrame, row_sample_max: int, seed: int) -> pd.DataFrame:
    if len(X) <= row_sample_max:
        return X.copy()
    return X.sample(n=row_sample_max, random_state=seed).copy()


# ---------- 预筛：Spearman 绝对相关 Top-K ----------
def prefilter_candidates_spearman(X: pd.DataFrame, topk: int, r_min: float) -> List[Tuple[int, int]]:
    Xf = X.fillna(X.median(numeric_only=True))
    corr = Xf.corr(method="spearman").abs().fillna(0.0).values
    n = corr.shape[0]
    pairs = set()
    for i in range(n):
        r = corr[i].copy()
        r[i] = -1.0
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


# ---------- 全局 memmap 用于并行 ----------
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
    i, j, mic_sample_max, seed = args
    Xmm = _load_memmap()
    n_rows = Xmm.shape[0]
    if n_rows > mic_sample_max:
        step = n_rows / mic_sample_max
        idx = (np.floor(np.arange(mic_sample_max) * step)).astype(int)
    else:
        idx = np.arange(n_rows, dtype=int)

    xi = Xmm[idx, i].astype(np.float64, copy=False)
    xj = Xmm[idx, j].astype(np.float64, copy=False)
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

    try:
        r = pd.Series(xi).corr(pd.Series(xj), method="spearman")
        return (i, j, float(abs(r)) if np.isfinite(r) else 0.0)
    except Exception:
        return (i, j, 0.0)


# ---------- 阈值选择 ----------
def choose_threshold_upper_quantile(A: np.ndarray, q: float, max_density: float):
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

    if not np.isfinite(thr) or thr >= 1.0 or edge_count_at(thr) == 0:
        for cand in [0.999, 0.995, 0.99, 0.98, 0.95, 0.90, 0.80, 0.70, 0.60]:
            if edge_count_at(cand) > 0:
                thr = cand
                break
        else:
            thr = 0.0

    total_possible = int(mask.sum())
    edges_now = edge_count_at(thr)

    if total_possible > 0 and edges_now / total_possible > max_density:
        for cand in [0.99, 0.995, 0.999]:
            if cand > thr and edge_count_at(cand) / total_possible <= max_density and edge_count_at(cand) > 0:
                thr = cand
                edges_now = edge_count_at(cand)
                break

    return thr, edges_now, total_possible


# ---------- 数据集模式：读取 No_Failure 80% 合并 ----------
def load_no_failure_aggregate(alfadata_dir: Path,
                              subdir_name: str,
                              file_glob: str,
                              frac: float,
                              min_select: int,
                              seed: int) -> Tuple[pd.DataFrame, List[str], List[str]]:
    nf_dir = alfadata_dir / subdir_name
    files = sorted([p for p in nf_dir.glob(file_glob) if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No files found under {nf_dir} with pattern {file_glob}")

    rng = np.random.RandomState(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)

    k = max(min_select, int(math.ceil(frac * len(files))))
    k = min(k, len(files))
    sel_idx = idx[:k]
    hold_idx = idx[k:]

    sel_files = [files[i] for i in sel_idx]
    hold_files = [files[i] for i in hold_idx]

    # 先扫一遍，取“公共数值列集合”（排除 timestamp*/label）
    def numeric_cols_exclude(df: pd.DataFrame) -> List[str]:
        explicit_exclude = set()
        for c in df.columns:
            lc = str(c).lower()
            if lc == "timestamp" or lc.startswith("timestamp"):
                explicit_exclude.add(c)
            if lc == "label":
                explicit_exclude.add(c)
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in explicit_exclude]
        return num_cols

    common_cols = None
    for fp in sel_files:
        try:
            df = pd.read_csv(fp, low_memory=False)
            cols = set(numeric_cols_exclude(df))
            if common_cols is None:
                common_cols = cols
            else:
                common_cols &= cols
        except Exception as e:
            print(f"[WARN] Read failed (skip for common set): {fp.name} | {e}")

    if not common_cols:
        raise RuntimeError("No common numeric columns across selected No_Failure files after excluding timestamp/label.")

    common_cols = sorted(list(common_cols))

    # 汇总数据（仅取公共数值列），并记录来源文件名列（可选）
    chunks = []
    used_files = []
    for fp in sel_files:
        try:
            df = pd.read_csv(fp, low_memory=False)
            sub = df[common_cols].copy()
            # 清理极端 NaN 列（容错：若某文件该公共列全 NaN，就跳过该文件）
            if sub.isna().all(axis=0).sum() > 0 and sub.shape[0] > 0:
                # 若任一公共列在该文件全 NaN，放弃该文件，避免污染
                if any(sub[c].isna().all() for c in common_cols):
                    print(f"[WARN] Drop file (column all-NaN in this file): {fp.name}")
                    continue
            sub["_source_file"] = fp.name
            chunks.append(sub)
            used_files.append(fp.name)
        except Exception as e:
            print(f"[WARN] Read failed (skip): {fp.name} | {e}")

    if not chunks:
        raise RuntimeError("No data rows loaded from selected files.")

    X = pd.concat(chunks, axis=0, ignore_index=True)

    return X, used_files, [f.name for f in hold_files]


# ---------- 主流程 ----------
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_mode = args.alfadata_dir is not None and Path(args.alfadata_dir).exists()

    # ===== 读取数据 =====
    if dataset_mode:
        alfadata_dir = Path(args.alfadata_dir).resolve()
        print(f"[INFO] Dataset mode: {alfadata_dir} | use {args.no_failure_fraction*100:.1f}% No_Failure files")
        Xall, used_files, heldout_files = load_no_failure_aggregate(
            alfadata_dir=alfadata_dir,
            subdir_name=args.no_failure_subdir,
            file_glob=args.file_glob,
            frac=float(args.no_failure_fraction),
            min_select=int(args.min_select),
            seed=int(args.random_seed),
        )
        # 列筛选（含常量列丢弃）
        keep_cols, drop_report, Xnum = filter_columns(Xall.drop(columns=["_source_file"]), drop_constant=bool(args.drop_constant))
        train_files_text = "\n".join(used_files) + "\n"
        heldout_files_text = "\n".join(heldout_files) + ("\n" if heldout_files else "")
    else:
        if not args.in_csv:
            raise ValueError("Either --alfadata_dir must exist or --in_csv must be provided.")
        in_csv = Path(args.in_csv)
        if not in_csv.exists():
            raise FileNotFoundError(f"in_csv not found: {in_csv}")
        print(f"[INFO] Single-file mode: {in_csv}")
        df = pd.read_csv(in_csv, low_memory=False)
        keep_cols, drop_report, Xnum = filter_columns(df, drop_constant=bool(args.drop_constant))
        train_files_text = f"{in_csv.name}\n"
        heldout_files_text = ""

    # 极端：不足两列（直接导出空产物）
    if len(keep_cols) < 2:
        (out_dir / "keep_columns.json").write_text(json.dumps(keep_cols, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(drop_report, columns=["column","reason"]).to_csv(out_dir / "drop_columns_report.csv", index=False)
        pd.DataFrame({"id": range(len(keep_cols)), "name": keep_cols}).to_csv(out_dir / "nodes.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "adjacency_dense.csv", index=False)
        pd.DataFrame(columns=["src","dst","weight"]).to_csv(out_dir / "edges_mic.csv", index=False)
        pd.DataFrame(columns=["i","j","weight"]).to_csv(out_dir / "A_global_sparse.csv", index=False)
        pd.DataFrame(columns=["node","degree","weighted_degree"]).to_csv(out_dir / "degrees.csv", index=False)
        (out_dir / "train_files.txt").write_text(train_files_text, encoding="utf-8")
        (out_dir / "heldout_files.txt").write_text(heldout_files_text, encoding="utf-8")
        print(f"[WARN] Not enough columns after filtering: {len(keep_cols)}")
        return

    # 行采样（在筛列之后）
    Xs = sample_rows(Xnum, row_sample_max=args.row_sample_max, seed=args.random_seed)

    # 预筛候选
    print("[INFO] Prefilter candidates by Spearman...")
    cand_pairs = prefilter_candidates_spearman(Xs, topk=args.prefilter_topk, r_min=args.prefilter_r_min)
    print(f"[INFO] Candidate pairs: {len(cand_pairs)} (from {len(keep_cols)} features)")

    # 构建 memmap 供并行
    mm_path = str(out_dir / "sample.memmap")
    Xarr = Xs.to_numpy(dtype=np.float32, copy=True)
    with open(mm_path, "wb"):
        pass
    mm = np.memmap(mm_path, mode="w+", dtype=np.float32, shape=Xarr.shape)
    mm[:] = Xarr[:]
    del mm  # 关闭写视图

    # 并行配置
    if args.processes <= 0:
        try:
            import multiprocessing as mp
            procs = max(1, mp.cpu_count() - 1)
        except Exception:
            procs = 1
    else:
        procs = args.processes

    # 计算指标（MIC 或 Spearman 回退）
    import multiprocessing as mp
    print(f"[INFO] Compute pair metrics with {procs} process(es); minepy={HAVE_MINEPY}")
    tasks = [(i, j, int(args.mic_sample_max), int(args.random_seed)) for (i, j) in cand_pairs]
    with mp.get_context("spawn").Pool(processes=procs, initializer=_init_worker,
                                      initargs=(mm_path, Xarr.shape)) as pool:
        results = list(pool.imap_unordered(_compute_metric_for_pair, tasks, chunksize=256))

    # 组装邻接矩阵
    n = len(keep_cols)
    A = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(A, 1.0)
    for i, j, w in results:
        if w > 0:
            if 0 <= i < n and 0 <= j < n and i != j:
                if w > A[i, j]:
                    A[i, j] = w
                    A[j, i] = w

    # 阈值选择（上三角）
    thr, edges_now, total_possible = choose_threshold_upper_quantile(A, q=args.quantile, max_density=args.max_density)
    print(f"[INFO] Threshold={thr:.6f}  edges={edges_now}/{total_possible}")

    # 孤立点剔除
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
            for nm in removed_names:
                drop_report.append((nm, "isolated_after_threshold"))
            A = A[keep_mask][:, keep_mask]
            keep_cols = [c for c, m in zip(keep_cols, keep_mask) if m]
            n = len(keep_cols)

    # 极端：剔除后不足两列
    if n < 2:
        (out_dir / "keep_columns.json").write_text(json.dumps(keep_cols, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(drop_report, columns=["column","reason"]).to_csv(out_dir / "drop_columns_report.csv", index=False)
        pd.DataFrame({"id": range(n), "name": keep_cols}).to_csv(out_dir / "nodes.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "adjacency_dense.csv", index=False)
        pd.DataFrame(columns=["src","dst","weight"]).to_csv(out_dir / "edges_mic.csv", index=False)
        pd.DataFrame(columns=["i","j","weight"]).to_csv(out_dir / "A_global_sparse.csv", index=False)
        pd.DataFrame(columns=["node","degree","weighted_degree"]).to_csv(out_dir / "degrees.csv", index=False)
        (out_dir / "train_files.txt").write_text(train_files_text, encoding="utf-8")
        (out_dir / "heldout_files.txt").write_text(heldout_files_text, encoding="utf-8")
        try:
            os.remove(mm_path)
        except Exception:
            pass
        print(f"[WARN] Too few columns after isolation filtering: {n}")
        return

    # ====== 导出产物 ======
    (out_dir / "keep_columns.json").write_text(json.dumps(keep_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(drop_report, columns=["column","reason"]).to_csv(out_dir / "drop_columns_report.csv", index=False)

    nodes_df = pd.DataFrame({"id": np.arange(n, dtype=int), "name": keep_cols})
    nodes_df.to_csv(out_dir / "nodes.csv", index=False)

    adj_df = pd.DataFrame(A, index=keep_cols, columns=keep_cols)
    adj_df.to_csv(out_dir / "adjacency_dense.csv", index=True)

    # edges / 稀疏三元组 / 度
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

    trips = [(i, i, 1.0) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            w = float(A[i, j])
            if w >= thr and w > 0.0:
                trips.append((i, j, w))
                trips.append((j, i, w))
    sparse_df = pd.DataFrame(trips, columns=["i","j","weight"])
    sparse_df.to_csv(out_dir / "A_global_sparse.csv", index=False)

    deg_df = pd.DataFrame({"node": keep_cols, "degree": deg, "weighted_degree": wdeg})
    deg_df.to_csv(out_dir / "degrees.csv", index=False)

    # 元数据（记录用于构图/保留文件）
    (out_dir / "train_files.txt").write_text(train_files_text, encoding="utf-8")
    (out_dir / "heldout_files.txt").write_text(heldout_files_text, encoding="utf-8")

    # 清理 memmap
    try:
        os.remove(mm_path)
    except Exception:
        pass

    print(f"[OK] Graph built at: {out_dir}")
    print(f"Nodes(after drop)={n}  Edges={len(edges_df)}  thr={thr:.6f}")


if __name__ == "__main__":
    main()
