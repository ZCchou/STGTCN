# -*- coding: utf-8 -*-


from pathlib import Path
import pandas as pd
import numpy as np

# Base directory (adjust as needed)
BASE_DIR = r"Live GPS Spoofing and Jamming"

# Subfolders to process
SUBFOLDERS = ["Benign Flight", "GPS Jamming", "GPS Spoofing"]

# Label threshold (degrees), matching the original repo (about kilometer scale)
DEG_THRESHOLD = 0.03


def read_csv_sorted(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path.name} is missing the 'timestamp' column")
    return df.sort_values("timestamp").reset_index(drop=True)


def find_topic_csv(folder: Path, keys: list[str]) -> Path | None:
    """
    Find a CSV in folder containing any key (e.g., *vehicle_gps_position_0*.csv).
    Return the first match; return None if not found.
    """
    for key in keys:
        for p in sorted(folder.glob(f"*{key}*.csv")):
            return p
    return None


def merge_chain_outer(dfs: list[pd.DataFrame | None]) -> pd.DataFrame:
    """
    Chain outer-merge(on='timestamp') in order.
    """
    final = pd.DataFrame()
    for df in dfs:
        if df is None or df.empty:
            continue
        if final.empty:
            final = df
        else:
            final = final.merge(df, how="outer", on="timestamp")
    if final.empty:
        raise RuntimeError("Merged result is empty; check whether the four topic CSVs exist.")
    return final


def linear_interpolate_on_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by timestamp → set index → linear interpolation (both directions) → restore columns
    """
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df.interpolate(axis=0, method="linear", limit_direction="both")
    return df.reset_index()


def drop_extra_timestamp_like_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns containing 'timestamp' except the main one (e.g., timestamp_x from merges).
    """
    droplist = [c for c in df.columns if ("timestamp" in c and c != "timestamp")]
    if droplist:
        df = df.drop(columns=droplist, errors="ignore")
    return df


def label_like_original(df: pd.DataFrame,
                        lat_col: str = "lat_x",
                        lon_col: str = "lon_x",
                        deg_threshold: float = DEG_THRESHOLD) -> pd.DataFrame:
    """
    Use first-row lat_x/lon_x as reference; if any row deviates by ±deg_threshold → malicious, else benign.
    If lat_x/lon_x is missing (or all NaN), label all as benign with a warning.
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        print(f"[WARN] Missing {lat_col}/{lon_col}; labeling all as 'benign'")
        df["label"] = "benign"
        return df

    lat = pd.to_numeric(df[lat_col], errors="coerce").ffill().bfill()
    lon = pd.to_numeric(df[lon_col], errors="coerce").ffill().bfill()

    if lat.isna().all() or lon.isna().all():
        print(f"[WARN] {lat_col}/{lon_col} are all NaN; labeling all as 'benign'")
        df["label"] = "benign"
        return df

    lat0, lon0 = lat.iloc[0], lon.iloc[0]
    cond = (
        (lat > lat0 + deg_threshold) | (lat < lat0 - deg_threshold) |
        (lon > lon0 + deg_threshold) | (lon < lon0 - deg_threshold)
    )
    df["label"] = np.where(cond, "malicious", "benign")
    vc = df["label"].value_counts(dropna=False)
    print(f"[INFO] Label distribution: {vc.to_dict()}")
    return df


def process_one_folder(folder: Path) -> None:
    print(f"\n==== Processing folder: {folder} ====")
    if not folder.exists():
        print(f"[WARN] Folder not found, skipping: {folder}")
        return

    # Topic keywords (compatible with vehicle_* and prefix-less)
    topic_keys = {
        "att":  ["vehicle_attitude_0", "attitude_0"],
        "gpos": ["vehicle_global_position_0", "global_position_0"],
        "gps":  ["vehicle_gps_position_0", "gps_position_0"],
        "lpos": ["vehicle_local_position_0", "local_position_0"],
    }

    # Find 4 CSVs (strict order: att → gpos → gps → lpos)
    att_path  = find_topic_csv(folder, topic_keys["att"])
    gpos_path = find_topic_csv(folder, topic_keys["gpos"])
    gps_path  = find_topic_csv(folder, topic_keys["gps"])
    lpos_path = find_topic_csv(folder, topic_keys["lpos"])

    for name, p in [("attitude", att_path),
                    ("global_position", gpos_path),
                    ("gps_position", gps_path),
                    ("local_position", lpos_path)]:
        print(f"[INFO] {name:16s}: {p.name if p else 'not found'}")

    # Read (missing ones are None; outer merge continues)
    att_df  = read_csv_sorted(att_path)  if att_path  else None
    gpos_df = read_csv_sorted(gpos_path) if gpos_path else None
    gps_df  = read_csv_sorted(gps_path)  if gps_path  else None
    lpos_df = read_csv_sorted(lpos_path) if lpos_path else None

    # Outer-join merge
    final_df = merge_chain_outer([att_df, gpos_df, gps_df, lpos_df])

    # Sort + linear interpolate
    final_df = linear_interpolate_on_timestamp(final_df)

    # Drop extra timestamp* columns
    final_df = drop_extra_timestamp_like_cols(final_df)

    # Label per original rule (lat_x/lon_x)
    final_df = label_like_original(final_df, lat_col="lat_x", lon_col="lon_x", deg_threshold=DEG_THRESHOLD)

    # Output
    out_dir = folder / "CSVs" / "Condensed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{folder.name}.csv"   # Keep original naming style: <folder_name>.csv
    final_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Wrote: {out_path}  rows={len(final_df)}, cols={final_df.shape[1]}")


def main():
    base = Path(BASE_DIR)
    for sub in SUBFOLDERS:
        process_one_folder(base / sub)


if __name__ == "__main__":
    main()
