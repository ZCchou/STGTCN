# -*- coding: utf-8 -*-
"""
Clean three merged CSVs (Benign Flight / GPS Jamming / GPS Spoofing),
keep only columns common to all three, and output to the local preprocessed/ folder.
Extra: normalize label to 0/1 (benign→0, malicious→1).

Default inputs (same conventions as the previous script):
  <BASE_DIR>/<SubFolder>/CSVs/Condensed/gpsonly.csv
  or <BASE_DIR>/<SubFolder>/CSVs/Condensed/<SubFolder>.csv
  if neither exists, use the first CSV under Condensed

Outputs:
  ./preprocessed/Benign Flight.csv
  ./preprocessed/GPS Jamming.csv
  ./preprocessed/GPS Spoofing.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Dict

# ======== Config: base dir and subfolders ========
BASE_DIR = r"Live GPS Spoofing and Jamming"
SUBFOLDERS = ["Benign Flight", "GPS Jamming", "GPS Spoofing"]
# =====================================

def find_condensed_csv(folder: Path) -> Optional[Path]:
    """
    Under <folder>/CSVs/Condensed, try in order:
      1) gpsonly.csv
      2) <folder.name>.csv
      3) first *.csv in the directory
    Return None if not found.
    """
    con = folder / "Processed"
    if not con.exists():
        return None
    cand1 = con / "gpsonly.csv"
    if cand1.exists():
        return cand1
    cand2 = con / f"{folder.name}.csv"
    if cand2.exists():
        return cand2
    csvs = sorted(con.glob("*.csv"))
    return csvs[0] if csvs else None

def as_int_ts(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def coerce_bool_like(col: pd.Series) -> pd.Series:
    """
    Normalize booleans/bool-like strings to 0/1; keep other types or coerce to numeric if possible.
    """
    if col.dtype == bool:
        return col.astype(np.int8)
    if col.dtype == object:
        lower = col.astype(str).str.strip().str.lower()
        mask_true  = lower.isin(["true","t","yes","y","1"])
        mask_false = lower.isin(["false","f","no","n","0"])
        if (mask_true | mask_false).any():
            out = col.copy()
            out[mask_true]  = 1
            out[mask_false] = 0
            try:
                return pd.to_numeric(out, errors="ignore")
            except Exception:
                return out
    return col

def coerce_numeric_except_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    For columns except 'label', try to convert object columns to numeric and booleans to 0/1.
    """
    for c in df.columns:
        if c == "label":
            continue
        df[c] = coerce_bool_like(df[c])
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly map label to 0/1:
      - 'benign','normal','0','false','no','n' -> 0
      - 'malicious','attack','1','true','yes','y' -> 1
      - numeric strings -> nonzero=1, zero=0
      - unknown values -> default to 0 (conservative)
    If no label column, return as-is.
    """
    if "label" not in df.columns:
        return df

    s = df["label"]
    # Numeric already: nonzero=1
    if pd.api.types.is_numeric_dtype(s):
        df["label"] = (pd.to_numeric(s, errors="coerce").fillna(0) != 0).astype(np.int8)
        return df

    # String mapping
    lower = s.astype(str).str.strip().str.lower()
    mapping = {
        "benign": 0, "normal": 0, "0": 0, "false": 0, "f": 0, "no": 0, "n": 0,
        "malicious": 1, "attack": 1, "1": 1, "true": 1, "t": 1, "yes": 1, "y": 1
    }
    m = lower.map(mapping)

    # For unmapped values, try numeric parsing
    num = pd.to_numeric(lower, errors="coerce")
    m = m.where(~m.isna(), num)
    # Default unknown → 0 (conservative)
    df["label"] = m.fillna(0).astype(np.int8)
    return df

def clean_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Gentle cleaning:
      - normalize timestamp to numeric, dedupe, sort
      - replace inf/-inf
      - drop all-null columns
      - coerce numeric / normalize booleans
      - label → 0/1
    """
    if "timestamp" not in df.columns:
        raise ValueError(f"{name} is missing the 'timestamp' column")

    # Normalize timestamp
    df["timestamp"] = as_int_ts(df["timestamp"])
    before_rows = len(df)
    df = df.dropna(subset=["timestamp"]).copy()
    if len(df) < before_rows:
        print(f"[{name}] Dropped rows with NaN timestamp: {before_rows - len(df)}")

    # Dedupe + sort
    dup_cnt = df.duplicated(subset=["timestamp"]).sum()
    if dup_cnt > 0:
        print(f"[{name}] Dropped duplicate timestamp rows: {dup_cnt}")
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Replace inf, then drop all-NaN columns
    df = df.replace([np.inf, -np.inf], np.nan)
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        print(f"[{name}] Dropped all-null columns ({len(all_nan_cols)}): "
              f"{all_nan_cols[:8]}{'...' if len(all_nan_cols)>8 else ''}")
        df = df.drop(columns=all_nan_cols)

    # Coerce types (except label)
    df = coerce_numeric_except_label(df)

    # label → 0/1
    df = normalize_label_column(df)

    # Drop columns that became all-null after cleaning
    all_nan_cols2 = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols2:
        print(f"[{name}] Dropped all-null columns again ({len(all_nan_cols2)}): "
              f"{all_nan_cols2[:8]}{'...' if len(all_nan_cols2)>8 else ''}")
        df = df.drop(columns=all_nan_cols2)

    return df

def main():
    base = Path(BASE_DIR)
    assert base.exists(), f"BASE_DIR does not exist: {base}"

    # Read three CSVs
    raw: Dict[str, pd.DataFrame] = {}
    paths: Dict[str, Path] = {}

    for sub in SUBFOLDERS:
        folder = base / sub
        p = find_condensed_csv(folder)
        if p is None:
            raise FileNotFoundError(f"No CSV found under {folder}/CSVs/Condensed")
        print(f"[LOAD] {sub}: {p}")
        df = pd.read_csv(p)
        raw[sub] = df
        paths[sub] = p

    # Clean
    cleaned: Dict[str, pd.DataFrame] = {}
    for sub in SUBFOLDERS:
        cleaned[sub] = clean_df(raw[sub], name=sub)
        # Print label distribution for verification
        if "label" in cleaned[sub].columns:
            vc = cleaned[sub]["label"].value_counts(dropna=False).to_dict()
            print(f"[LABEL] {sub} distribution: {vc}")
        print(f"[CLEAN] {sub}: rows={len(cleaned[sub])}, cols={cleaned[sub].shape[1]}")

    # Get common columns (intersection)
    cols_sets = [set(cleaned[sub].columns) for sub in SUBFOLDERS]
    common_cols = set.intersection(*cols_sets)
    if not common_cols:
        raise RuntimeError("No common columns after cleaning. Check upstream merge or cleaning rules.")

    # Align column order: follow Benign Flight order (timestamp first)
    benign_cols = list(cleaned["Benign Flight"].columns)
    ordered_common = [c for c in benign_cols if c in common_cols]
    if "timestamp" in ordered_common:
        ordered_common = ["timestamp"] + [c for c in ordered_common if c != "timestamp"]

    print(f"[COMMON] common columns={len(ordered_common)}, first 20={ordered_common[:20]}")

    # Output directory: local preprocessed/
    try:
        script_dir = Path(__file__).parent
    except NameError:
        script_dir = Path.cwd()
    out_dir = script_dir / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # For each dataset: keep common columns and write in consistent order
    for sub in SUBFOLDERS:
        df = cleaned[sub].copy()
        # Defensive: ensure all common columns exist
        for c in ordered_common:
            if c not in df.columns:
                df[c] = np.nan
        df = df[ordered_common]

        out_path = out_dir / f"{sub}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[SAVE] {sub} → {out_path} rows={len(df)}, cols={df.shape[1]}")

if __name__ == "__main__":
    main()
