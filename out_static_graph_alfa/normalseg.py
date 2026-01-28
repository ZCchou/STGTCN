# -*- coding: utf-8 -*-
"""
cut_normal_segments.py
从 alldata_labeled/ 中切出所有连续正常片段（label==0），保存到新目录。

功能要点
- 按连续的 label==0 片段分割，每段单独保存 CSV。
- 可选：最小长度筛选（按行数或按秒数）。
- 可选：把很短的异常“缝隙”（label==1）并回到正常（按行数或按秒数）。
- 支持 _dt（datetime）或自动解析 %time（整数 ns/us/ms/s 或字符串时间）。
- 输出文件名包含片段起止时间戳，便于溯源（自动避开 Windows 不能用的字符）。

用法示例：
    python cut_normal_segments.py --in_dir ./alldata_labeled --out_dir ./alldata_normals --min_rows 30 --merge_gap_rows 5
或：
    python cut_normal_segments.py --in_dir ./alldata_labeled --out_dir ./alldata_normals --min_seconds 2 --merge_gap_seconds 0.2
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import math
import re
import sys

def detect_time_unit_from_int(x: int) -> str:
    d = len(str(abs(int(x))))
    if d >= 18:
        return "ns"
    elif d >= 16:
        return "us"
    elif d >= 13:
        return "ms"
    else:
        return "s"

def to_datetime_series(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    # 若大多数可转数字，则按整数时间戳处理
    if s_num.notna().mean() > 0.8:
        med = s_num.dropna().median()
        unit = detect_time_unit_from_int(int(med))
        dt = pd.to_datetime(s_num.astype("Int64"), unit=unit, utc=True).dt.tz_convert(None)
        return dt
    # 否则按字符串时间解析
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, utc=False)
    if dt.isna().all():
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    return dt

def ensure_datetime(df: pd.DataFrame) -> pd.Series:
    """
    返回可用于时长计算与命名的 datetime 序列：
    - 优先使用 _dt（若已存在且为 datetime）
    - 否则尝试解析 %time
    - 都不可用时，返回全 NaT
    """
    if "_dt" in df.columns and np.issubdtype(df["_dt"].dtype, np.datetime64):
        return df["_dt"]
    if "%time" in df.columns:
        dt = to_datetime_series(df["%time"])
        if dt.notna().any():
            return dt
    return pd.to_datetime(pd.Series([pd.NaT] * len(df)))

def fmt_dt_for_fname(ts) -> str:
    """
    将时间戳格式化用于文件名（避免 : 等非法字符）。
    NaT -> 'NaT'
    """
    if pd.isna(ts):
        return "NaT"
    # 统一到毫秒级，形如 20180911-185938-100
    return pd.Timestamp(ts).strftime("%Y%m%d-%H%M%S-%f")[:-3]  # 去掉到毫秒

def merge_short_gaps(labels: pd.Series,
                     dt: pd.Series | None,
                     merge_gap_rows: int = 0,
                     merge_gap_seconds: float = 0.0) -> pd.Series:
    """
    将很短的异常缝隙（label==1 的短段）并回到 0。
    - 若提供 merge_gap_seconds>0 且有 dt，则按持续时间判断；否则按行数判断。
    """
    y = labels.astype(int).to_numpy().copy()
    if len(y) == 0:
        return labels

    # 计算分段
    run_id = (labels != labels.shift(1)).cumsum().to_numpy()
    # 获取每段的 (value, start_idx, end_idx)
    segments = []
    start = 0
    for i in range(1, len(y)+1):
        if i == len(y) or run_id[i] != run_id[i-1]:
            end = i  # [start, end)
            val = y[start]
            segments.append((val, start, end))
            start = i

    # 遍历 label==1 的段，满足阈值则置 0
    y_out = y.copy()
    use_time = (merge_gap_seconds > 0.0) and (dt is not None) and np.issubdtype(dt.dtype, np.datetime64)
    for val, s, e in segments:
        if val != 1:
            continue
        length = e - s
        ok = False
        if use_time:
            # 时间长度（秒）
            t0 = dt.iloc[s]
            t1 = dt.iloc[e-1]
            if pd.notna(t0) and pd.notna(t1):
                dur = (t1 - t0).total_seconds()
            else:
                dur = float("inf")
            if dur <= merge_gap_seconds + 1e-12:
                ok = True
        else:
            if length <= merge_gap_rows:
                ok = True
        if ok:
            y_out[s:e] = 0

    return pd.Series(y_out, index=labels.index)

def cut_segments(df: pd.DataFrame,
                 min_rows: int,
                 min_seconds: float,
                 merge_gap_rows: int,
                 merge_gap_seconds: float) -> list[tuple[int, int]]:
    """
    返回满足条件的正常片段索引区间列表 [(s,e), ...]，半开区间 [s,e)。
    """
    if "label" not in df.columns:
        return []

    # 确保顺序稳定；若有 _dt 则按时间排序
    work = df.copy()
    if "_dt" in work.columns and np.issubdtype(work["_dt"].dtype, np.datetime64):
        work = work.sort_values("_dt").reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)

    # 合并短异常缝隙
    dt = work["_dt"] if ("_dt" in work.columns and np.issubdtype(work["_dt"].dtype, np.datetime64)) else None
    labels = merge_short_gaps(work["label"], dt, merge_gap_rows, merge_gap_seconds)

    # 重新分段
    run_id = (labels != labels.shift(1)).cumsum()
    segs = []
    for rid, g in work.groupby(run_id):
        lab = int(labels.loc[g.index[0]])
        if lab != 0:
            continue
        s = g.index.min()
        e = g.index.max() + 1  # 半开
        # 长度筛选
        ok = True
        if min_rows > 0 and (e - s) < min_rows:
            ok = False
        if ok and min_seconds > 0.0 and dt is not None:
            t0 = dt.iloc[s]; t1 = dt.iloc[e-1]
            if pd.notna(t0) and pd.notna(t1):
                dur = (t1 - t0).total_seconds()
                if dur < min_seconds - 1e-12:
                    ok = False
        if ok:
            segs.append((s, e))
    return segs, work

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="./alldata_labeled", help="输入目录（含打好 label 的 CSV）")
    ap.add_argument("--out_dir", type=str, default="./alldata_normals", help="输出目录（切片 CSV 将写到这里）")
    # 片段最小长度（任选一种，若两者都给，二者都需满足）
    ap.add_argument("--min_rows", type=int, default=1, help="正常片段的最少行数（默认1）")
    ap.add_argument("--min_seconds", type=float, default=0.0, help="正常片段的最少持续时间（秒，默认0=不限制）")
    # 合并短异常缝隙（任选一种；若两者都给，优先按秒数；否则按行数）
    ap.add_argument("--merge_gap_rows", type=int, default=0, help="将长度≤该行数的异常段并回正常（默认0=不并）")
    ap.add_argument("--merge_gap_seconds", type=float, default=0.0, help="将持续≤该秒数的异常段并回正常（默认0=不并）")
    ap.add_argument("--overwrite", action="store_true", help="若目标存在是否覆盖")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"[ERROR] 输入目录不存在：{in_dir}")
        sys.exit(1)

    files = sorted(in_dir.glob("*.csv"))
    if not files:
        print(f"[WARN] 未在 {in_dir} 找到 CSV")
        sys.exit(0)

    total_in = len(files)
    total_out = 0
    print(f"[INFO] 将处理 {total_in} 个文件，输出至：{out_dir}")

    for i, fp in enumerate(files, 1):
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception as e:
            print(f"[ERROR] 读取失败：{fp.name} | {e}")
            continue

        if "label" not in df.columns:
            print(f"[WARN] 跳过（无 label 列）：{fp.name}")
            continue

        # 准备 _dt（用于命名和秒数约束）
        if "_dt" not in df.columns or not np.issubdtype(df["_dt"].dtype, np.datetime64):
            dt = ensure_datetime(df)
            df["_dt"] = dt

        segs, df_sorted = cut_segments(
            df, min_rows=max(1, int(args.min_rows)),
            min_seconds=float(args.min_seconds),
            merge_gap_rows=max(0, int(args.merge_gap_rows)),
            merge_gap_seconds=float(args.merge_gap_seconds)
        )

        if not segs:
            print(f"[{i}/{total_in}] {fp.name} -> 无满足条件的正常片段")
            continue

        stem = fp.stem
        written = 0
        for k, (s, e) in enumerate(segs, 1):
            part = df_sorted.iloc[s:e].copy()

            # 片段起止时间（用于命名）
            dt_start = part["_dt"].iloc[0] if "_dt" in part.columns else pd.NaT
            dt_end = part["_dt"].iloc[-1] if "_dt" in part.columns else pd.NaT
            t0 = fmt_dt_for_fname(dt_start)
            t1 = fmt_dt_for_fname(dt_end)

            out_name = f"{stem}__normal_seg{k:03d}__{t0}__{t1}.csv"
            out_fp = out_dir / out_name

            if out_fp.exists() and not args.overwrite:
                # 避免覆盖，简单加序号后缀
                out_name = f"{stem}__normal_seg{k:03d}__{t0}__{t1}__dup.csv"
                out_fp = out_dir / out_name

            try:
                part.to_csv(out_fp, index=False)
                written += 1
                total_out += 1
            except Exception as e:
                print(f"[ERROR] 保存失败：{out_name} | {e}")

        print(f"[{i}/{total_in}] {fp.name} -> 正常片段 {written} 段")

    print(f"[DONE] 共处理 {total_in} 个文件，导出正常片段 {total_out} 段到：{out_dir}")

if __name__ == "__main__":
    main()
