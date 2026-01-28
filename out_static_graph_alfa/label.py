# -*- coding: utf-8 -*-
"""
Label ALFA flights in `alldata/` using failure_status under `ALFA/processed/<flight>/`.

Usage:
    python label_alfa_from_failure_status.py \
        --alldata_dir ./alldata \
        --alfa_processed_dir ./ALFA/processed \
        --out_dir ./alldata_labeled

Notes:
- Automatically detect and parse two timestamp formats:
  * numeric (auto-detect ns/us/ms/s)
  * string (e.g., '2018-09-11 18:59:38.100')
- Alignment: treat failure_status as a status series, align to alldata timestamps, forward-fill (pad leading with 0).
- If multiple failure_status files exist for a flight, merge by max (union).
- no_failure files (filename contains 'no_failure') are labeled all 0.
- Interpolate missing data over time + ffill/bfill fallback.
"""

import argparse
from pathlib import Path
import sys
import re
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # quiet mode

def detect_time_unit_from_int(x: int) -> str:
    """
    Infer unit from integer timestamp scale: ns/us/ms/s.
    """
    # Digit length of absolute value
    d = len(str(abs(int(x))))
    # Common ranges (rough heuristic)
    if d >= 18:   # 1_000_000_000_000_000_000 (ns)
        return "ns"
    elif d >= 16: # 1_000_000_000_000_000 (likely us)
        return "us"
    elif d >= 13: # 1_000_000_000_000 (ms)
        return "ms"
    else:
        return "s"

def to_datetime_series(s: pd.Series) -> pd.Series:
    """
    Convert %time column to pandas datetime (naive, nanosecond precision).
    Compatible with:
      - string timestamps (e.g., '2018-09-11 18:59:38.100')
      - numeric timestamps (ns/us/ms/s), auto-detect unit
    """
    # Try numeric parsing first
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() > 0.8:
        # Infer unit by median
        med = s_num.dropna().median()
        unit = detect_time_unit_from_int(int(med))
        dt = pd.to_datetime(s_num.astype("Int64"), unit=unit, utc=True)
        return dt.dt.tz_convert(None)  # drop timezone, make naive
    else:
        # Parse strings
        dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, utc=False)
        # If all NaT, try dayfirst/format fallback
        if dt.isna().all():
            dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
        return dt

def read_alldata_csv(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, low_memory=False)
    if "%time" not in df.columns:
        raise ValueError(f"[ERROR] {fp.name} is missing '%time' column")
    # Parse time column
    df["_dt"] = to_datetime_series(df["%time"])
    if df["_dt"].isna().all():
        raise ValueError(f"[ERROR] {fp.name} '%time' cannot be parsed to datetime")
    # Keep original %time string column, add _dt for alignment/interpolation
    return df

def read_failure_status_csv(fp: Path) -> pd.Series:
    """
    Read a failure_status CSV and return a datetime-indexed status series (0/1 int).
    Expected columns: %time, field.data
    """
    df = pd.read_csv(fp, low_memory=False, comment="#")
    # Some files may have BOM or weird header whitespace: normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    # Tolerance: allow slight field column name variants
    field_col = None
    for cand in ["field.data", "field.data[0]", "field.data[1]", "data", "value", "status"]:
        if cand in df.columns:
            field_col = cand
            break
    if field_col is None and len(df.columns) >= 2:
        # Fallback: use second column as status
        field_col = df.columns[1]

    if "%time" not in df.columns:
        # Fallback: try matching columns containing time
        tcol = [c for c in df.columns if "%time" in c or "time" == c.lower()]
        if not tcol:
            raise ValueError(f"[ERROR] {fp} missing time column")
        tcol = tcol[0]
    else:
        tcol = "%time"

    # Parse time column
    df["_dt"] = to_datetime_series(df[tcol])
    df = df.loc[df["_dt"].notna()].copy()
    if df.empty:
        raise ValueError(f"[ERROR] {fp} time column parsed to empty")

    # Convert status to 0/1
    s = df[field_col].copy()
    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower().map({
            "1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0
        }).fillna(0)
    else:
        s = pd.to_numeric(s, errors="coerce").fillna(0)

    s = (s > 0).astype(int)
    s.index = df["_dt"].values
    s = s.sort_index()
    # Deduplicate (same timestamp, take max conservatively)
    s = s.groupby(level=0).max()
    return s

def collect_failure_status_series(flight_dir: Path) -> pd.Series:
    """
    Aggregate all failure_status files under a flight directory, merging by per-timestamp max.
    Return an empty Series if none are found.
    """
    patterns = [
        "*-failure_status*.csv",       # common pattern
        "*failure_status*.csv",        # fallback
    ]
    files = []
    for pat in patterns:
        files.extend(sorted(flight_dir.glob(pat)))
    files = list(dict.fromkeys(files))  # dedupe, keep order

    combined = None
    for f in files:
        try:
            s = read_failure_status_csv(f)
            combined = s if combined is None else pd.concat([combined, s], axis=1).max(axis=1)
        except Exception as e:
            print(f"[WARN] Failed to read {f.name}: {e}")

    if combined is None:
        return pd.Series(dtype="int64")  # empty
    return combined.astype(int)

def align_status_to_alldata_times(status_s: pd.Series, alldata_times: pd.Series) -> pd.Series:
    """
    Align merged failure_status (datetime index) to alldata timestamps (datetime).
    Strategy: reindex -> ffill; leading NaN -> 0.
    """
    if status_s.empty:
        return pd.Series(0, index=alldata_times, dtype="int64")
    status_s = status_s.sort_index()
    # Align to alldata timestamps
    s_aligned = status_s.reindex(alldata_times, method="pad")  # ffill
    s_aligned = s_aligned.fillna(0).astype(int)
    s_aligned.index = alldata_times
    return s_aligned

def sanitize_and_interpolate(df: pd.DataFrame, time_col: str = "_dt") -> pd.DataFrame:
    """
    Interpolate numeric columns over time; then ffill/bfill fallback;
    do not touch 'label' and original '%time' columns.
    """
    work = df.copy()
    if time_col not in work.columns:
        return work
    work = work.sort_values(time_col)
    work = work.set_index(time_col)

    protected = {"%time", "label"}
    numeric_cols = [c for c in work.columns if c not in protected and pd.api.types.is_numeric_dtype(work[c])]
    # Leave non-numeric columns unchanged (could be string sensor states)
    if numeric_cols:
        try:
            work[numeric_cols] = work[numeric_cols].interpolate(method="time", limit_direction="both")
        except Exception:
            # If index isn't DatetimeIndex or interpolation fails, fall back to linear
            work[numeric_cols] = work[numeric_cols].interpolate(limit_direction="both")
        work[numeric_cols] = work[numeric_cols].fillna(method="ffill").fillna(method="bfill")

    work = work.reset_index()
    return work

def main():
    ap = argparse.ArgumentParser(description="Label ALFA alldata using failure_status under ALFA/processed.")
    ap.add_argument("--alldata_dir", type=str, default="./alldata", help="Dir: processed per-flight CSVs")
    ap.add_argument("--alfa_processed_dir", type=str, default="./ALFA/processed", help="Dir: raw ALFA processed")
    ap.add_argument("--out_dir", type=str, default="./alldata_labeled", help="Output directory")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing targets")
    args = ap.parse_args()

    alldata_dir = Path(args.alldata_dir).resolve()
    processed_dir = Path(args.alfa_processed_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not alldata_dir.exists():
        print(f"[ERROR] alldata_dir does not exist: {alldata_dir}")
        sys.exit(1)
    if not processed_dir.exists():
        print(f"[ERROR] alfa_processed_dir does not exist: {processed_dir}")
        sys.exit(1)

    csvs = sorted(alldata_dir.glob("*.csv"))
    if not csvs:
        print(f"[WARN] No CSVs found under {alldata_dir}")
        sys.exit(0)

    print(f"[INFO] Processing {len(csvs)} flight files, output to: {out_dir}")

    for i, fp in enumerate(csvs, 1):
        flight_name = fp.stem  # corresponds to processed/<flight_name>/
        out_fp = out_dir / f"{flight_name}.csv"

        if out_fp.exists() and not args.overwrite:
            print(f"[SKIP {i}/{len(csvs)}] {fp.name} → target exists (skipped).")
            continue

        print(f"[{i}/{len(csvs)}] Processing flight: {fp.name}")

        try:
            df = read_alldata_csv(fp)
        except Exception as e:
            print(f"[ERROR] Failed to read alldata: {fp.name} | {e}")
            continue

        # no_failure -> all zeros
        is_no_failure = ("no_failure" in flight_name.lower())
        flight_dir = processed_dir / flight_name

        if is_no_failure:
            print("  - Detected no_failure, labeling entire segment as 0")
            label = pd.Series(0, index=df["_dt"], dtype="int64")
        else:
            if not flight_dir.exists():
                print(f"  [WARN] Missing source directory: {flight_dir.name}, labeling all 0")
                label = pd.Series(0, index=df["_dt"], dtype="int64")
            else:
                status_s = collect_failure_status_series(flight_dir)
                if status_s.empty:
                    print("  [WARN] No failure_status files found, labeling all 0")
                    label = pd.Series(0, index=df["_dt"], dtype="int64")
                else:
                    print(f"  - Aggregated failure_status with {status_s.shape[0]} status timestamps")
                    label = align_status_to_alldata_times(status_s, df["_dt"])

        # Add label column (0/1)
        df["label"] = label.values.astype(np.int64)

        # Handle missing values (leave %time and label untouched)
        df_out = sanitize_and_interpolate(df, time_col="_dt")

        # Preserve original column order: keep or drop _dt (optional)
        # Here we keep %time (original string) and _dt (alignment/interp), but you can drop _dt if desired.
        df_out = df_out.drop(columns=["_dt"])

        # Save
        try:
            df_out.to_csv(out_fp, index=False)
            n1 = int(df_out["label"].sum())
            print(f"  -> Saved: {out_fp.name} | rows={len(df_out)} | anomaly count(1)={n1}")
        except Exception as e:
            print(f"[ERROR] Save failed: {out_fp} | {e}")

    print("[DONE] All processing complete.")

if __name__ == "__main__":
    main()
