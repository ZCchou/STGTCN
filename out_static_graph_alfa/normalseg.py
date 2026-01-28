# -*- coding: utf-8 -*-
"""
cut_normal_segments.py
Cut all continuous normal segments (label==0) from alldata_labeled/ and save to a new directory.

Key features
- Split by continuous label==0 segments; save each segment as a CSV.
- Optional: minimum length filter (by rows or by seconds).
- Optional: merge short anomaly "gaps" (label==1) back to normal (by rows or seconds).
- Supports _dt (datetime) or auto-parse %time (integer ns/us/ms/s or string time).
- Output filenames include segment start/end timestamps (avoids Windows-invalid characters).

Usage examples:
    python cut_normal_segments.py --in_dir ./alldata_labeled --out_dir ./alldata_normals --min_rows 30 --merge_gap_rows 5
or:
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
    # If most values are numeric, treat as integer timestamps
    if s_num.notna().mean() > 0.8:
        med = s_num.dropna().median()
        unit = detect_time_unit_from_int(int(med))
        dt = pd.to_datetime(s_num.astype("Int64"), unit=unit, utc=True).dt.tz_convert(None)
        return dt
    # Otherwise parse as string time
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, utc=False)
    if dt.isna().all():
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    return dt

def ensure_datetime(df: pd.DataFrame) -> pd.Series:
    """
    Return a datetime series usable for duration and naming:
    - Prefer _dt if present and datetime
    - Otherwise parse %time
    - If unavailable, return all NaT
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
    Format timestamp for filenames (avoid illegal characters like ':').
    NaT -> 'NaT'
    """
    if pd.isna(ts):
        return "NaT"
    # Normalize to millisecond precision, e.g., 20180911-185938-100
    return pd.Timestamp(ts).strftime("%Y%m%d-%H%M%S-%f")[:-3]  # truncate to milliseconds

def merge_short_gaps(labels: pd.Series,
                     dt: pd.Series | None,
                     merge_gap_rows: int = 0,
                     merge_gap_seconds: float = 0.0) -> pd.Series:
    """
    Merge short anomaly gaps (short label==1 runs) back to 0.
    - If merge_gap_seconds>0 and dt is available, use duration; otherwise use row count.
    """
    y = labels.astype(int).to_numpy().copy()
    if len(y) == 0:
        return labels

    # Compute segments
    run_id = (labels != labels.shift(1)).cumsum().to_numpy()
    # Get (value, start_idx, end_idx) for each segment
    segments = []
    start = 0
    for i in range(1, len(y)+1):
        if i == len(y) or run_id[i] != run_id[i-1]:
            end = i  # [start, end)
            val = y[start]
            segments.append((val, start, end))
            start = i

    # Traverse label==1 segments; set to 0 if under threshold
    y_out = y.copy()
    use_time = (merge_gap_seconds > 0.0) and (dt is not None) and np.issubdtype(dt.dtype, np.datetime64)
    for val, s, e in segments:
        if val != 1:
            continue
        length = e - s
        ok = False
        if use_time:
            # Duration in seconds
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
    Return list of normal segment index ranges [(s,e), ...] with half-open intervals [s,e).
    """
    if "label" not in df.columns:
        return []

    # Ensure stable order; sort by time if _dt exists
    work = df.copy()
    if "_dt" in work.columns and np.issubdtype(work["_dt"].dtype, np.datetime64):
        work = work.sort_values("_dt").reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)

    # Merge short anomaly gaps
    dt = work["_dt"] if ("_dt" in work.columns and np.issubdtype(work["_dt"].dtype, np.datetime64)) else None
    labels = merge_short_gaps(work["label"], dt, merge_gap_rows, merge_gap_seconds)

    # Re-segment
    run_id = (labels != labels.shift(1)).cumsum()
    segs = []
    for rid, g in work.groupby(run_id):
        lab = int(labels.loc[g.index[0]])
        if lab != 0:
            continue
        s = g.index.min()
        e = g.index.max() + 1  # half-open
        # Length filter
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
    ap.add_argument("--in_dir", type=str, default="./alldata_labeled", help="Input dir (CSV files with labels)")
    ap.add_argument("--out_dir", type=str, default="./alldata_normals", help="Output dir (segments written here)")
    # Minimum segment length (choose either; if both set, both must pass)
    ap.add_argument("--min_rows", type=int, default=1, help="Minimum rows in normal segment (default 1)")
    ap.add_argument("--min_seconds", type=float, default=0.0, help="Minimum duration in seconds (default 0=none)")
    # Merge short anomaly gaps (choose either; if both set, seconds take priority)
    ap.add_argument("--merge_gap_rows", type=int, default=0, help="Merge anomaly runs with length ≤ rows (default 0=off)")
    ap.add_argument("--merge_gap_seconds", type=float, default=0.0, help="Merge anomaly runs with duration ≤ seconds (default 0=off)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing targets")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"[ERROR] Input directory does not exist: {in_dir}")
        sys.exit(1)

    files = sorted(in_dir.glob("*.csv"))
    if not files:
        print(f"[WARN] No CSV files found under {in_dir}")
        sys.exit(0)

    total_in = len(files)
    total_out = 0
    print(f"[INFO] Processing {total_in} files, output to: {out_dir}")

    for i, fp in enumerate(files, 1):
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception as e:
            print(f"[ERROR] Failed to read: {fp.name} | {e}")
            continue

        if "label" not in df.columns:
            print(f"[WARN] Skipping (no label column): {fp.name}")
            continue

        # Prepare _dt (for naming and seconds constraints)
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
            print(f"[{i}/{total_in}] {fp.name} -> no qualifying normal segments")
            continue

        stem = fp.stem
        written = 0
        for k, (s, e) in enumerate(segs, 1):
            part = df_sorted.iloc[s:e].copy()

            # Segment start/end times (for naming)
            dt_start = part["_dt"].iloc[0] if "_dt" in part.columns else pd.NaT
            dt_end = part["_dt"].iloc[-1] if "_dt" in part.columns else pd.NaT
            t0 = fmt_dt_for_fname(dt_start)
            t1 = fmt_dt_for_fname(dt_end)

            out_name = f"{stem}__normal_seg{k:03d}__{t0}__{t1}.csv"
            out_fp = out_dir / out_name

            if out_fp.exists() and not args.overwrite:
                # Avoid overwrite: add suffix
                out_name = f"{stem}__normal_seg{k:03d}__{t0}__{t1}__dup.csv"
                out_fp = out_dir / out_name

            try:
                part.to_csv(out_fp, index=False)
                written += 1
                total_out += 1
            except Exception as e:
                print(f"[ERROR] Save failed: {out_name} | {e}")

        print(f"[{i}/{total_in}] {fp.name} -> normal segments {written}")

    print(f"[DONE] Processed {total_in} files; exported {total_out} normal segments to: {out_dir}")

if __name__ == "__main__":
    main()
